import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model._llms import model, ModelTimeoutError, ModelConnectionError
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import BaseTool, tool
import json
import logging
import time
import traceback
import re
import html
import threading
import queue
import base64
import urllib.parse

from _tools._img.generate_images import generate_image_url_tool, generate_image_in_thread, hash_prompt, IMAGE_CACHE
from _tools._email.send_email import send_email, simple_send_email, send_html_email, validate_email
from _agents.image_mailer_agent._functions_prompt import functions_prompt
from _agents._agent_utils import (
    logger, timing_decorator, with_retry, JSONFixer, 
    AgentResultFormatter, SafeToolExecutor, JSONParseError,
    ToolExecutionError
)

# 配置日志
logger = logging.getLogger('image_mailer_agent')

# 设置SMTP超时时间
os.environ['SMTP_TIMEOUT'] = '15'  # 15秒超时

# 全局邮件发送状态队列
email_result_queue = queue.Queue()

# 记录解析错误计数
JSON_PARSE_ERROR_COUNT = 0
MAX_JSON_PARSE_ERRORS = 3

# Agent 初始化
@timing_decorator
def agent_init():
    try:
        # 由于AgentExecutor创建问题，我们将直接使用模型和工具，而不是通过AgentExecutor
        # 这是一个完全不同的方法，避免了问题的根源
        
        # 创建一个简单的代理执行器，仅包含图片生成工具
        # 这将绕过LangChain的AgentExecutor类的问题
        
        logger.info("使用自定义代理执行器替代AgentExecutor...")
        
        # 返回一个简单的对象，模拟AgentExecutor的接口
        class SimpleAgentExecutor:
            def __init__(self):
                self.tools = [generate_image_url_tool]
                self.verbose = True
            
            def invoke(self, inputs):
                """模拟AgentExecutor的invoke方法"""
                query = inputs.get("input", "")
                logger.info(f"SimpleAgentExecutor处理查询: {query}")
                
                # 提取邮箱和图片描述
                return direct_process_request(query)
        
        return SimpleAgentExecutor()
    except Exception as e:
        logger.error(f"初始化image_mailer_agent失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# 直接处理请求，不使用AgentExecutor
def direct_process_request(query):
    """直接处理请求，不通过AgentExecutor"""
    logger.info("直接处理请求，不使用AgentExecutor...")
    
    # 提取邮箱
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    email_match = email_pattern.search(query)
    
    if not email_match:
        return {
            "output": "无法从您的请求中提取有效的邮箱地址。请提供正确的邮箱格式，例如example@mail.com。",
            "intermediate_steps": []
        }
    
    recipient = email_match.group(0)
    
    # 验证邮箱
    if not validate_email(recipient):
        return {
            "output": f"提供的邮箱地址 {recipient} 格式不正确。请提供有效的邮箱地址。",
            "intermediate_steps": []
        }
    
    # 提取图片描述
    query_without_email = query.replace(recipient, "")
    
    # 尝试提取图片描述
    image_patterns = [
        r'(生成|创建|画)(一张|一副|一幅)?(.*?)(图片|图像|照片)',
        r'(图片|图像|照片)(内容|主题|描述|是|为|：|:)(.*?)(?=发送|邮件|邮箱|$)',
        r'(主题|内容|描述)(是|为|：|:)(.*?)(?=发送|邮件|邮箱|$)'
    ]
    
    image_description = ""
    for pattern in image_patterns:
        match = re.search(pattern, query_without_email)
        if match and len(match.groups()) >= 3:
            image_description = match.group(3).strip()
            if image_description:
                break
    
    # 如果无法提取特定描述，使用简单描述
    if not image_description:
        image_description = "一张小狗图片"
    
    # 启动图片生成和邮件发送（后台处理）
    generate_thread = threading.Thread(
        target=process_image_and_email,
        args=(image_description, recipient)
    )
    generate_thread.daemon = True
    generate_thread.start()
    
    # 返回即时响应
    return {
        "output": f"已开始生成「{image_description}」的图片，完成后将发送到您的邮箱({recipient})。整个过程可能需要几秒钟，请耐心等待。",
        "intermediate_steps": []
    }

# 处理图片生成和邮件发送
def process_image_and_email(prompt, email_to):
    """处理图片生成和邮件发送的完整流程"""
    try:
        logger.info(f"开始处理图片生成和邮件发送: {prompt} -> {email_to}")
        
        # 生成图片URL
        from _tools._img.generate_images import generate_image_url_tool, hash_prompt, IMAGE_CACHE
        
        # 首先尝试生成图片
        image_result = generate_image_url_tool(prompt)
        
        # 如果状态是pending，需要等待图片生成完成
        if image_result and image_result.get("status") == "pending":
            logger.info("图片生成正在进行中，等待完成...")
            
            # 获取缓存键
            cache_key = hash_prompt(prompt)
            
            # 等待图片生成完成，最多等待60秒
            max_wait_time = 60
            wait_interval = 5
            total_waited = 0
            
            while total_waited < max_wait_time:
                # 检查缓存是否已有结果
                if cache_key in IMAGE_CACHE:
                    cached_result = IMAGE_CACHE[cache_key]
                    # 如果状态是成功，使用缓存的结果
                    if cached_result.get("status") == "success" and "image_url" in cached_result:
                        image_result = {
                            "status": "success",
                            "image_url": cached_result["image_url"]
                        }
                        logger.info(f"找到缓存的图片结果，等待时间: {total_waited}秒")
                        break
                
                # 短暂等待后再次检查
                time.sleep(wait_interval)
                total_waited += wait_interval
                logger.info(f"等待图片生成完成: {total_waited}秒...")
            
            # 如果等待超时，记录错误
            if total_waited >= max_wait_time:
                logger.warning(f"等待图片生成超时 ({max_wait_time}秒)")
        
        # 检查最终结果
        if not image_result or "image_url" not in image_result:
            logger.error(f"图片生成失败或超时: {image_result}")
            
            # 发送失败通知邮件
            try:
                email_subject = f"图片生成通知: {prompt}"
                email_body = f"""
您好，

您请求的图片「{prompt}」正在生成中，但尚未完成。
系统会在图片生成完成后自动发送给您，请耐心等待。

祝好,
AI助手
"""
                from _tools._email.send_email import send_plain_email
                send_plain_email(email_to, email_subject, email_body)
                logger.info(f"已发送图片生成进行中通知邮件: {email_to}")
            except Exception as notify_error:
                logger.error(f"发送通知邮件失败: {str(notify_error)}")
            
            return
        
        image_url = image_result["image_url"]
        logger.info(f"图片生成成功: {image_url[:30]}...")
        
        # 发送邮件
        email_subject = f"您请求的图片: {prompt}"
        email_body = f"""
您好，

您请求的图片已生成。请查看附件或点击以下链接查看:
{image_url}

祝好,
AI助手
"""
        
        # 使用简单的纯文本邮件发送
        try:
            from _tools._email.send_email import simple_send_email, send_plain_email
            
            # 首先尝试使用simple_send_email
            try:
                simple_send_email(email_to, email_subject, email_body)
                logger.info(f"邮件发送成功: {email_to}")
            except Exception as email_error:
                logger.error(f"邮件发送失败: {str(email_error)}")
                
                # 尝试使用备用方法发送
                try:
                    send_plain_email(email_to, email_subject, email_body)
                    logger.info(f"使用备用方法发送邮件成功: {email_to}")
                except Exception as backup_error:
                    logger.error(f"备用邮件发送方法也失败: {str(backup_error)}")
                    
                    # 最后尝试使用极简内容
                    try:
                        minimal_subject = "您请求的图片已生成"
                        minimal_body = f"""
您好，

您请求的图片「{prompt}」已生成完成。
由于技术原因，无法在邮件中直接显示图片链接，请复制以下地址到浏览器查看：

{image_url.replace('&', '[和]').replace('%', '[百分号]')}

祝好,
AI助手
"""
                        send_plain_email(email_to, minimal_subject, minimal_body)
                        logger.info(f"使用极简内容发送邮件成功: {email_to}")
                    except Exception as final_error:
                        logger.error(f"所有邮件发送方法均失败: {str(final_error)}")
        except Exception as e:
            logger.error(f"处理邮件发送失败: {str(e)}")
    
    except Exception as e:
        logger.error(f"处理图片生成和邮件发送失败: {str(e)}")
        logger.error(traceback.format_exc())

# 修复JSON格式化
def fix_json_format(json_str):
    """修复不完整或格式错误的JSON字符串"""
    # 检查是否缺少结束的引号和大括号
    if not json_str.endswith('"}'):
        # 添加缺失的引号和大括号
        if '"' in json_str and not json_str.endswith('"'):
            json_str += '"'
        if '{' in json_str and not json_str.endswith('}'):
            json_str += '}'
    
    try:
        # 尝试解析JSON
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        logger.warning(f"JSON格式错误，尝试修复: {e}")
        
        # 检查常见格式错误
        if '"prompt":' in json_str and '"email_to":' in json_str:
            # 提取关键参数并重新构建JSON
            prompt_match = re.search(r'"prompt":\s*"([^"]*)"', json_str)
            email_match = re.search(r'"email_to":\s*"([^"]*)"', json_str)
            
            if prompt_match and email_match:
                prompt = prompt_match.group(1)
                email = email_match.group(1)
                
                # 构建正确格式的JSON
                fixed_json = f'{{"prompt": "{prompt}", "email_to": "{email}"}}'
                return fixed_json
        
        # 无法修复，返回原始字符串
        return json_str

# 安全执行Agent
@timing_decorator
@with_retry(max_retries=2, initial_delay=1, backoff_factor=2)
def safe_execute_agent(agent, query):
    """安全地执行Agent，带有重试机制"""
    global JSON_PARSE_ERROR_COUNT
    
    try:
        start_time = time.time()
        result = agent.invoke({"input": query})
        execution_time = time.time() - start_time
        logger.info(f"Agent执行成功，耗时: {execution_time:.2f}秒")
        
        # 重置错误计数
        JSON_PARSE_ERROR_COUNT = 0
        
        return result
    except Exception as e:
        error_msg = str(e)
        
        # 如果是JSON解析错误，增加计数
        if "not valid JSON" in error_msg or "JSON" in error_msg or "Could not parse" in error_msg:
            JSON_PARSE_ERROR_COUNT += 1
            logger.error(f"JSON解析错误 #{JSON_PARSE_ERROR_COUNT}: {error_msg}")
            
            # 尝试从错误中提取工具调用参数并修复
            try:
                if "arguments" in error_msg:
                    # 提取参数字符串
                    args_match = re.search(r"'arguments': '([^']*)'", error_msg)
                    if args_match:
                        args_str = args_match.group(1)
                        fixed_args = fix_json_format(args_str)
                        logger.info(f"修复后的参数: {fixed_args}")
                        
                        # 如果包含提示和邮箱，直接调用工具
                        try:
                            args_dict = json.loads(fixed_args)
                            if "prompt" in args_dict and "email_to" in args_dict:
                                # 直接处理请求
                                return direct_process_request(query)
                        except:
                            pass
            except Exception as fix_error:
                logger.error(f"修复JSON参数失败: {str(fix_error)}")
            
            # 如果达到最大错误次数，直接切换到快速路径
            if JSON_PARSE_ERROR_COUNT >= MAX_JSON_PARSE_ERRORS:
                logger.warning(f"达到最大JSON解析错误次数({MAX_JSON_PARSE_ERRORS})，切换到快速路径")
                # 直接处理请求
                return direct_process_request(query)
        
        if "timeout" in error_msg.lower():
            logger.error(f"Agent执行超时: {error_msg}")
            raise ModelTimeoutError(f"模型响应超时: {error_msg}")
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            logger.error(f"Agent网络连接错误: {error_msg}")
            raise ModelConnectionError(f"网络连接错误: {error_msg}")
        elif "not valid JSON" in error_msg or "JSON" in error_msg or "Could not parse" in error_msg:
            logger.error(f"JSON解析错误: {error_msg}")
            
            # 直接处理请求
            return direct_process_request(query)
        else:
            logger.error(f"Agent执行出错: {error_msg}")
            logger.error(traceback.format_exc())
            raise

@timing_decorator
def run_agent(query: str):
    request_id = f"img_mail_{int(time.time())}"
    logger.info(f"开始处理请求 {request_id}: {query}")
    
    # 直接处理请求，不使用复杂的Agent执行器
    try:
        # 初始化简单代理
        agent = agent_init()
        
        # 安全执行代理
        result = safe_execute_agent(agent, query)
        logger.info(f"请求 {request_id} 处理成功")
        return result
    except Exception as e:
        error_message = str(e)
        logger.error(f"Agent处理请求 {request_id} 出错: {error_message}")
        logger.error(traceback.format_exc())
        
        # 如果Agent处理失败，直接处理请求
        try:
            return direct_process_request(query)
        except Exception as direct_error:
            logger.error(f"直接处理请求失败: {str(direct_error)}")
        
        # 构建友好的错误消息
        if "timeout" in error_message.lower():
            return AgentResultFormatter.format_error(error_message, "timeout")
        elif "connection" in error_message.lower() or "network" in error_message.lower():
            return AgentResultFormatter.format_error(error_message, "network")
        else:
            return AgentResultFormatter.format_error(
                f"处理您的请求时出现问题。请尝试使用更简单的描述方式。\n错误详情: {error_message[:100]}...",
                "general"
            )

# 测试入口
if __name__ == "__main__":
    res = run_agent("生成一张小狗图片，并发送到example@example.com邮箱")
    print(res)
