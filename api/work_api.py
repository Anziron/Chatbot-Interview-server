import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from _workflow._work import chat
from _workflow._database import Database
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import time
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('work_api')

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    enable_web: bool = False
    enable_illation: bool = False
    thread_id: str = "1"

@router.post("/chat")
async def chatbot(
    query: str = Form(...),
    enable_web: bool = Form(False),
    enable_illation: bool = Form(False),
    thread_id: str = Form("1"),
):
    start_time = time.time()
    request_id = f"chat_{int(time.time())}"
    logger.info(f"接收到聊天请求 ID: {request_id}, 线程ID: {thread_id}")
    
    try:
        # 处理查询
        full_query = query
        
        # 调用chat函数获取响应
        res, price_info, illation = chat(full_query, enable_web, enable_illation, thread_id)
        
        # 记录处理时间
        processing_time = time.time() - start_time
        logger.info(f"请求 {request_id} 处理完成，耗时: {processing_time:.2f}秒")
        
        # 构建响应对象
        response = {
            "res": res,
            "status": price_info.get("status", "completed"),
            "processing_time": processing_time,
            "request_id": request_id,
            "price": price_info.get("price", 0),
            "tokens": price_info.get("tokens", 0)
        }
        
        # 只有在启用推理的情况下才返回推理过程
        if enable_illation and illation is not None:
            response["illation"] = illation
            
        return response
        
    except Exception as e:
        logger.error(f"处理请求 {request_id} 时出错: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 返回错误响应
        error_response = {
            "res": f"系统处理请求时出错: {str(e)[:100]}...",
            "status": "error",
            "processing_time": time.time() - start_time,
            "request_id": request_id,
            "error": str(e)
        }
        
        return error_response

class DeleteThreadRequest(BaseModel):
    thread_id: str

@router.post("/delete_thread_memory")
def delete_thread_memory(req: DeleteThreadRequest):
    try:
        # 调用Database类的delete_thread方法删除线程记忆
        Database.delete_thread(req.thread_id)
        return {"success": True, "message": f"成功删除线程ID {req.thread_id} 的记忆数据"}
    except Exception as e:
        return {"success": False, "message": f"删除线程ID {req.thread_id} 的记忆数据失败: {str(e)}"}

# 健康检查端点
@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": time.time(),
        "service": "chat_api"
    }
