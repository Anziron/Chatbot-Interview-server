import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uuid
import re
import logging
import traceback
import time

from _agents.image_mailer_agent._agent import run_agent as run_image_mailer_agent
from _agents.invoice_extractor_agent._agent import run_agent as run_invoice_extractor_agent
from _agents.meeting_notes_agent._agent import run_agent as run_meeting_notes_agent
from _tools._identify.invoice import identify_invoice
from _tools._identify.meeting_minutes import identify_meeting_minutes

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'api_errors.log'))  # 输出到文件
    ]
)
logger = logging.getLogger('agent_api')

# 创建路由
router = APIRouter()

# 图片上传目录 - 使用绝对路径
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"uploads"))
logger.info(f"图片上传目录: {UPLOAD_DIR}")

# 确保上传目录存在
try:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    logger.info(f"确保上传目录已创建: {UPLOAD_DIR}")
except Exception as e:
    logger.error(f"创建上传目录失败: {str(e)}")
    logger.error(traceback.format_exc())

# 请求模型定义
class AgentRequest(BaseModel):
    query: str
    image_path: Optional[str] = None

# 用于跟踪正在处理的请求
active_requests: Dict[str, Dict[str, Any]] = {}

# 安全调用agent函数的包装器
def safe_agent_call(agent_func, query, max_retries=3):
    """安全地调用agent函数，包括重试逻辑"""
    retry_count = 0
    retry_delay = 1  # 初始延迟1秒
    
    while retry_count < max_retries:
        try:
            start_time = time.time()
            result = agent_func(query)
            elapsed_time = time.time() - start_time
            logger.info(f"Agent调用成功，耗时: {elapsed_time:.2f}秒")
            return result
        except Exception as e:
            retry_count += 1
            logger.error(f"Agent调用失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                logger.info(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数级回退
            else:
                logger.error(f"达到最大重试次数，放弃: {str(e)}")
                logger.error(traceback.format_exc())
                # 返回错误信息而不是抛出异常
                return {
                    "output": f"处理您的请求时出现错误: {str(e)[:100]}...",
                    "intermediate_steps": []
                }
    
    # 不应该到达这里，但以防万一
    return {
        "output": "系统暂时无法处理您的请求，请稍后再试。",
        "intermediate_steps": []
    }

# 保存上传的图片
async def save_uploaded_image(image: UploadFile = None):
    if not image:
        return None
        
    try:
        logger.info(f"开始处理上传文件: {image.filename}")
        
        # 确认上传目录存在
        if not os.path.exists(UPLOAD_DIR):
            logger.info(f"上传目录不存在，尝试创建: {UPLOAD_DIR}")
            os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # 生成唯一文件名
        file_extension = os.path.splitext(image.filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        logger.info(f"将保存文件到: {file_path}")
        
        # 读取文件内容
        file_content = await image.read()
        logger.info(f"读取到文件内容，大小: {len(file_content)} 字节")
        
        # 保存文件
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        logger.info(f"文件已保存到: {file_path}")
        
        # 验证文件是否已创建
        if os.path.exists(file_path):
            logger.info(f"已确认文件存在: {file_path}")
            file_size = os.path.getsize(file_path)
            logger.info(f"文件大小: {file_size} 字节")
        else:
            logger.warning(f"警告: 文件保存后不存在: {file_path}")
        
        # 返回访问URL
        return f"/uploads/{unique_filename}"
    except Exception as e:
        logger.error(f"保存图片出错: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# 图片上传API - 保留此API以兼容旧版本
@router.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    try:
        image_url = await save_uploaded_image(file)
        if image_url:
            return {
                "success": True,
                "filename": os.path.basename(image_url),
                "file_path": os.path.join(UPLOAD_DIR, os.path.basename(image_url)),
                "url": image_url
            }
        else:
            return {
                "success": False,
                "error": "无法保存图片"
            }
    except Exception as e:
        logger.error(f"上传图片出错: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

# 获取图片绝对路径
def get_absolute_image_path(image_path):
    if not image_path:
        return None
        
    if image_path.startswith("/uploads/"):
        filename = os.path.basename(image_path)
        return os.path.join(UPLOAD_DIR, filename)
    return image_path

# 预处理图片（根据Agent类型调用不同的识别工具）
def preprocess_image(agent_type, image_path):
    if not image_path:
        return None
        
    absolute_path = get_absolute_image_path(image_path)
    logger.info(f"预处理图片: {absolute_path}, Agent类型: {agent_type}")
    
    try:
        if agent_type == "invoice_extractor":
            logger.info("调用发票识别工具")
            result = identify_invoice(absolute_path)
            return {"type": "invoice", "data": result}
        elif agent_type == "meeting_notes":
            logger.info("调用会议纪要识别工具")
            result = identify_meeting_minutes(absolute_path)
            return {"type": "meeting_notes", "data": result}
        else:
            return None
    except Exception as e:
        logger.error(f"图片预处理失败: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# 后台任务：跟踪并清理完成的请求
def cleanup_completed_requests():
    """清理已完成的请求"""
    current_time = time.time()
    keys_to_remove = []
    
    for req_id, req_info in active_requests.items():
        # 如果请求完成超过10分钟，从跟踪列表中删除
        if req_info.get("completed", False) and (current_time - req_info.get("completion_time", 0)) > 600:
            keys_to_remove.append(req_id)
    
    for key in keys_to_remove:
        active_requests.pop(key, None)
    
    logger.info(f"清理了 {len(keys_to_remove)} 个已完成的请求，当前活跃请求数: {len(active_requests)}")

# 图片邮件Agent
@router.post("/agents/image_mailer")
async def image_mailer_endpoint(
    background_tasks: BackgroundTasks,
    query: str = Form(...), 
    image: Optional[UploadFile] = File(None)
):
    request_id = f"image_mailer_{uuid.uuid4().hex}"
    active_requests[request_id] = {
        "type": "image_mailer",
        "start_time": time.time(),
        "completed": False
    }
    
    try:
        # 图片邮件助手不需要预处理图片，直接调用Agent
        full_query = query
        
        # 调用Agent
        logger.info(f"开始调用image_mailer_agent，请求ID: {request_id}")
        result = safe_agent_call(run_image_mailer_agent, full_query)
        
        # 更新请求状态
        active_requests[request_id]["completed"] = True
        active_requests[request_id]["completion_time"] = time.time()
        active_requests[request_id]["success"] = True
        
        # 清理已完成的请求
        background_tasks.add_task(cleanup_completed_requests)
        
        return {
            "success": True,
            "result": result.get("output", ""),
            "steps": [step for step in result.get("intermediate_steps", [])]
        }
    except Exception as e:
        logger.error(f"image_mailer_agent处理失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 更新请求状态
        active_requests[request_id]["completed"] = True
        active_requests[request_id]["completion_time"] = time.time()
        active_requests[request_id]["success"] = False
        active_requests[request_id]["error"] = str(e)
        
        # 清理已完成的请求
        background_tasks.add_task(cleanup_completed_requests)
        
        return {
            "success": False,
            "error": str(e)
        }

# 发票提取Agent
@router.post("/agents/invoice_extractor")
async def invoice_extractor_endpoint(
    background_tasks: BackgroundTasks,
    query: str = Form(...), 
    image: Optional[UploadFile] = File(None)
):
    request_id = f"invoice_extractor_{uuid.uuid4().hex}"
    active_requests[request_id] = {
        "type": "invoice_extractor",
        "start_time": time.time(),
        "completed": False
    }
    
    try:
        # 构建查询字符串
        full_query = query
        image_result = None
        image_path = None
        
        # 如果有图片，先保存再进行预处理
        if image:
            image_path = await save_uploaded_image(image)
            if image_path:
                absolute_path = get_absolute_image_path(image_path)
                logger.info(f"发票提取 - 图片路径: {absolute_path}")
                
                # 预处理图片
                image_result = preprocess_image("invoice_extractor", absolute_path)
                
                if image_result:
                    logger.info(f"图片预处理成功，识别结果长度: {len(image_result['data'])}")
                    # 将图片识别结果添加到查询中
                    full_query = f"[已识别发票数据：{image_result['data']}] {query}"
                else:
                    # 如果预处理失败，使用原始路径
                    full_query = f"[图片内容：{absolute_path}] {query}"
        
        # 调用Agent
        logger.info(f"开始调用invoice_extractor_agent，请求ID: {request_id}")
        result = safe_agent_call(run_invoice_extractor_agent, full_query)
        
        # 更新请求状态
        active_requests[request_id]["completed"] = True
        active_requests[request_id]["completion_time"] = time.time()
        active_requests[request_id]["success"] = True
        
        # 清理已完成的请求
        background_tasks.add_task(cleanup_completed_requests)
        
        return {
            "success": True,
            "result": result.get("output", ""),
            "steps": [step for step in result.get("intermediate_steps", [])],
            "image_result": image_result
        }
    except Exception as e:
        logger.error(f"invoice_extractor_agent处理失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 更新请求状态
        active_requests[request_id]["completed"] = True
        active_requests[request_id]["completion_time"] = time.time()
        active_requests[request_id]["success"] = False
        active_requests[request_id]["error"] = str(e)
        
        # 清理已完成的请求
        background_tasks.add_task(cleanup_completed_requests)
        
        return {
            "success": False,
            "error": str(e)
        }

# 会议笔记Agent
@router.post("/agents/meeting_notes")
async def meeting_notes_endpoint(
    background_tasks: BackgroundTasks,
    query: str = Form(...), 
    image: Optional[UploadFile] = File(None)
):
    request_id = f"meeting_notes_{uuid.uuid4().hex}"
    active_requests[request_id] = {
        "type": "meeting_notes",
        "start_time": time.time(),
        "completed": False
    }
    
    try:
        # 构建查询字符串
        full_query = query
        image_result = None
        image_path = None
        
        # 如果有图片，先保存再进行预处理
        if image:
            image_path = await save_uploaded_image(image)
            if image_path:
                absolute_path = get_absolute_image_path(image_path)
                logger.info(f"会议笔记 - 图片路径: {absolute_path}")
                
                # 预处理图片
                image_result = preprocess_image("meeting_notes", absolute_path)
                
                if image_result:
                    logger.info(f"图片预处理成功，识别结果长度: {len(image_result['data'])}")
                    # 将图片识别结果添加到查询中
                    full_query = f"[已识别会议纪要：{image_result['data']}] {query}"
                else:
                    # 如果预处理失败，使用原始路径
                    full_query = f"[图片内容：{absolute_path}] {query}"
        
        # 调用Agent
        logger.info(f"开始调用meeting_notes_agent，请求ID: {request_id}")
        result = safe_agent_call(run_meeting_notes_agent, full_query)
        
        # 更新请求状态
        active_requests[request_id]["completed"] = True
        active_requests[request_id]["completion_time"] = time.time()
        active_requests[request_id]["success"] = True
        
        # 清理已完成的请求
        background_tasks.add_task(cleanup_completed_requests)
        
        return {
            "success": True,
            "result": result.get("output", ""),
            "steps": [step for step in result.get("intermediate_steps", [])],
            "image_result": image_result
        }
    except Exception as e:
        logger.error(f"meeting_notes_agent处理失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 更新请求状态
        active_requests[request_id]["completed"] = True
        active_requests[request_id]["completion_time"] = time.time()
        active_requests[request_id]["success"] = False
        active_requests[request_id]["error"] = str(e)
        
        # 清理已完成的请求
        background_tasks.add_task(cleanup_completed_requests)
        
        return {
            "success": False,
            "error": str(e)
        }

# 健康检查API
@router.get("/agents/health")
async def health_check():
    """提供agent服务的健康状态检查"""
    try:
        return {
            "status": "ok",
            "active_requests": len(active_requests),
            "uptime": "unknown"  # 可以扩展为真实的服务启动时间
        }
    except Exception as e:
        logger.error(f"健康检查API出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
