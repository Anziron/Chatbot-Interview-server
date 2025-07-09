import os
import json
import time
import hashlib
import requests
import logging
import threading
import queue
from typing import Optional, Dict, Any, Callable, List, Union
import asyncio
import threading
import traceback
import base64
import re
from langchain_core.tools import tool

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('image_generator')

# 图片缓存
IMAGE_CACHE = {}  # 哈希值 -> 图片结果

# 最大缓存项数
MAX_CACHE_ITEMS = 100
# 过期时间（小时）
CACHE_EXPIRY_HOURS = 24

# 图片生成队列
image_queue = queue.Queue()
# 线程池
MAX_THREADS = 3
active_threads = []
thread_semaphore = threading.Semaphore(MAX_THREADS)

# 正在处理的请求
PENDING_REQUESTS = {}

# 邮件请求信息
EMAIL_REQUESTS = {}

def hash_prompt(prompt: str) -> str:
    """为提示词生成一个哈希值作为缓存键"""
    return hashlib.md5(prompt.encode('utf-8')).hexdigest()

def clean_expired_cache():
    """清理过期的缓存项"""
    now = time.time()
    expiry_seconds = CACHE_EXPIRY_HOURS * 3600
    
    # 获取所有过期的键
    expired_keys = [
        key for key, value in IMAGE_CACHE.items()
        if 'generated_at' in value and now - value['generated_at'] > expiry_seconds
    ]
    
    # 删除过期项
    for key in expired_keys:
        del IMAGE_CACHE[key]
    
    # 如果缓存超过最大项数，删除最旧的
    if len(IMAGE_CACHE) > MAX_CACHE_ITEMS:
        # 按生成时间排序
        sorted_items = sorted(
            IMAGE_CACHE.items(),
            key=lambda x: x[1].get('last_access', 0)
        )
        
        # 删除最旧的项，直到达到最大数量
        for key, _ in sorted_items[:len(sorted_items) - MAX_CACHE_ITEMS]:
            del IMAGE_CACHE[key]
    
    return len(expired_keys)

# 提取用户请求中的邮箱
def extract_email_from_query(query: str) -> Optional[str]:
    """从查询字符串中提取邮箱地址"""
    if not query:
        return None
    
    # 尝试提取邮箱
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    email_match = email_pattern.search(query)
    
    if email_match:
        return email_match.group(0)
    
    return None

# 检查环境变量并发送邮件
def auto_send_email_after_generation(prompt: str, image_url: str):
    """图片生成成功后，自动发送邮件（如果有相关请求）"""
    try:
        # 检查是否有对应的邮件请求
        cache_key = hash_prompt(prompt)
        if cache_key not in EMAIL_REQUESTS:
            # 没有关联的邮件请求
            logger.info(f"没有找到与提示词关联的邮件请求: {prompt[:30]}...")
            return
        
        email_info = EMAIL_REQUESTS[cache_key]
        recipient = email_info.get('recipient')
        original_query = email_info.get('query', '')
        
        if not recipient:
            # 尝试从原始查询中提取
            recipient = extract_email_from_query(original_query)
        
        if not recipient:
            logger.warning(f"找不到收件人邮箱，无法自动发送邮件")
            return
        
        # 导入邮件发送模块
        sys_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if sys_path not in sys.path:
            import sys
            sys.path.append(sys_path)
        
        # 动态导入邮件发送模块
        try:
            from _tools._email.send_email import simple_send_email, send_plain_email
            
            # 准备邮件内容
            email_body = f"""您好，

您请求的图片已生成。根据描述「{prompt}」，AI助手为您创建了图片。
您可以通过以下链接查看图片：

{image_url}

祝您使用愉快！
AI助手"""
            
            # 发送邮件
            logger.info(f"图片生成成功，自动发送邮件到: {recipient}")
            
            # 在新线程中发送邮件，避免阻塞
            def send_email_thread():
                try:
                    # 首先尝试使用simple_send_email
                    try:
                        result = simple_send_email(
                            to=recipient,
                            subject=f"AI助手为您生成的图片",
                            body=email_body
                        )
                        logger.info(f"自动邮件发送结果: {result[:100] if isinstance(result, str) else str(result)[:100]}...")
                    except Exception as email_err:
                        logger.error(f"自动发送邮件失败，尝试备用方法: {str(email_err)}")
                        
                        # 尝试使用备用方法
                        try:
                            # 使用send_plain_email作为备用
                            result = send_plain_email(
                                to=recipient,
                                subject=f"AI助手为您生成的图片",
                                body=email_body
                            )
                            logger.info(f"使用备用方法发送邮件成功: {result[:100] if isinstance(result, str) else str(result)[:100]}...")
                        except Exception as backup_err:
                            logger.error(f"备用邮件发送方法也失败: {str(backup_err)}")
                            
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
                                send_plain_email(recipient, minimal_subject, minimal_body)
                                logger.info(f"使用极简内容发送邮件成功: {recipient}")
                            except Exception as final_error:
                                logger.error(f"所有邮件发送方法均失败: {str(final_error)}")
                except Exception as e:
                    logger.error(f"自动发送邮件失败: {str(e)}")
                    logger.error(traceback.format_exc())
            
            # 启动邮件发送线程
            email_thread = threading.Thread(target=send_email_thread)
            email_thread.daemon = True
            email_thread.start()
            
            # 清理邮件请求记录
            if cache_key in EMAIL_REQUESTS:
                del EMAIL_REQUESTS[cache_key]
                
        except ImportError as imp_err:
            logger.error(f"导入邮件模块失败: {str(imp_err)}")
        except Exception as e:
            logger.error(f"自动发送邮件时出错: {str(e)}")
            logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"处理自动邮件发送时出错: {str(e)}")
        logger.error(traceback.format_exc())

def generate_image_in_thread(prompt: str, cache_key: str, callback: Callable = None, email_to: str = None, original_query: str = None):
    """在单独线程中生成图片"""
    # 保存邮件信息
    if email_to:
        EMAIL_REQUESTS[cache_key] = {
            'recipient': email_to,
            'query': original_query,
            'timestamp': time.time()
        }
    
    def thread_worker():
        try:
            thread_semaphore.acquire()
            logger.info(f"线程开始生成图片: {prompt[:30]}...")
            
            # 首先检查缓存
            if cache_key in IMAGE_CACHE:
                result = IMAGE_CACHE[cache_key]
                # 更新最后访问时间
                result['last_access'] = time.time()
                logger.info(f"缓存命中: {prompt[:30]}...")
                
                # 如果图片生成成功，尝试自动发送邮件
                if result.get('status') == 'success' and 'image_url' in result:
                    auto_send_email_after_generation(prompt, result['image_url'])
                
                # 回调函数通知结果
                if callback:
                    callback(result)
                return
            
            # 调用图片生成API
            try:
                # Dashscope API配置
                api_key = os.environ.get('DASHSCOPE_API_KEY')
                if not api_key:
                    logger.error("未设置DASHSCOPE_API_KEY环境变量")
                    result = {
                        "status": "error",
                        "message": "未配置DASHSCOPE_API_KEY环境变量"
                    }
                    
                    if callback:
                        callback(result)
                    return
                
                # 使用官方SDK而不是直接调用API
                try:
                    from http import HTTPStatus
                    from dashscope import ImageSynthesis
                    
                    # 调用DashScope API
                    start_time = time.time()
                    logger.info(f"开始调用DashScope生成图片: {prompt[:30]}...")
                    
                    rsp = ImageSynthesis.call(
                        api_key=api_key,
                        model="wanx-v1",
                        prompt=prompt,
                        n=1,
                        size='1024*1024'
                    )
                    
                    elapsed_time = time.time() - start_time
                    logger.info(f"DashScope响应完成，耗时: {elapsed_time:.2f}秒")
                    
                    if rsp.status_code == HTTPStatus.OK:
                        # 提取图片URL
                        image_url = rsp.output.results[0].url
                        
                        # 存储到缓存
                        cache_result = {
                            "status": "success",
                            "image_url": image_url,
                            "generated_at": time.time(),
                            "last_access": time.time()
                        }
                        
                        IMAGE_CACHE[cache_key] = cache_result
                        
                        # 清理过期缓存
                        clean_expired_cache()
                        
                        logger.info(f"图片生成成功，已加入缓存")
                        
                        # 自动发送邮件
                        auto_send_email_after_generation(prompt, image_url)
                        
                        # 回调通知结果
                        if callback:
                            callback(cache_result)
                        return
                    else:
                        logger.error(f"API调用失败: {rsp.status_code} - {rsp.message}")
                        result = {
                            "status": "error",
                            "message": f"API调用失败: {rsp.status_code}"
                        }
                except ImportError:
                    # 如果没有DashScope SDK，使用请求库
                    logger.warning("无法导入DashScope SDK，使用备用请求方法")
                    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}"
                    }
                    
                    payload = {
                        "model": "wanx-v1",
                        "input": {
                            "prompt": prompt
                        },
                        "parameters": {
                            "style": "photographic",
                            "n": 1
                        }
                    }
                    
                    response = requests.post(url, headers=headers, json=payload, timeout=30)
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        
                        # 提取图片URL
                        if 'output' in response_data and 'results' in response_data['output']:
                            image_url = response_data['output']['results'][0]['url']
                            
                            # 存储到缓存
                            cache_result = {
                                "status": "success",
                                "image_url": image_url,
                                "generated_at": time.time(),
                                "last_access": time.time()
                            }
                            
                            IMAGE_CACHE[cache_key] = cache_result
                            
                            # 清理过期缓存
                            clean_expired_cache()
                            
                            logger.info(f"图片生成成功: {prompt[:30]}...")
                            
                            # 自动发送邮件
                            auto_send_email_after_generation(prompt, image_url)
                            
                            # 回调通知结果
                            if callback:
                                callback(cache_result)
                            return
                        else:
                            logger.error(f"API响应格式不正确: {response_data}")
                            result = {
                                "status": "error",
                                "message": "API响应格式不正确"
                            }
                    else:
                        logger.error(f"API调用失败: {response.status_code} - {response.text}")
                        result = {
                            "status": "error",
                            "message": f"API调用失败: {response.status_code}"
                        }
            except Exception as e:
                logger.error(f"生成图片时出错: {str(e)}")
                logger.error(traceback.format_exc())
                result = {
                    "status": "error",
                    "message": f"生成图片时出错: {str(e)}"
                }
            
            # 存储错误结果到缓存
            result['generated_at'] = time.time()
            result['last_access'] = time.time()
            IMAGE_CACHE[cache_key] = result
            
            # 回调通知结果
            if callback:
                callback(result)
                
        except Exception as e:
            logger.error(f"图片生成线程出错: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 尝试通知错误
            if callback:
                error_result = {
                    "status": "error",
                    "message": f"图片生成线程出错: {str(e)}"
                }
                callback(error_result)
                
        finally:
            # 释放信号量
            thread_semaphore.release()
            
    # 创建并启动线程
    generate_thread = threading.Thread(target=thread_worker)
    generate_thread.daemon = True
    generate_thread.start()
    
    # 记录正在处理的请求
    PENDING_REQUESTS[cache_key] = {
        "prompt": prompt,
        "thread": generate_thread,
        "start_time": time.time()
    }
    
    return generate_thread

@tool
def generate_image_url_tool(prompt: str, email_to: str = None) -> Dict[str, Any]:
    """
    根据文本描述生成图片，返回图片URL
    
    参数:
    - prompt: 图片描述文本，例如："一只可爱的小猫"
    - email_to: 可选参数，如果提供则在图片生成后自动发送到该邮箱
    
    返回:
    - 包含图片URL的字典，例如：{"status": "success", "image_url": "https://example.com/image.jpg"}
    """
    logger.info(f"开始生成图片: {prompt[:30]}..." + (f" 并发送到: {email_to}" if email_to else ""))
    start_time = time.time()
    
    # 检查参数
    if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
        return {
            "status": "error",
            "message": "图片描述不能为空"
        }
    
    # 生成缓存键
    cache_key = hash_prompt(prompt)
    
    # 提取原始查询
    original_query = f"生成图片: {prompt}" + (f" 发送到: {email_to}" if email_to else "")
    
    # 检查缓存
    if cache_key in IMAGE_CACHE:
        cached_result = IMAGE_CACHE[cache_key]
        # 更新最后访问时间
        cached_result['last_access'] = time.time()
        
        # 检查是否有URL字段并且状态是成功
        if cached_result.get('status') == 'success' and 'image_url' in cached_result:
            logger.info(f"从缓存返回图片: {prompt[:30]}...")
            
            # 如果提供了邮箱，尝试自动发送邮件
            if email_to:
                auto_send_email_after_generation(prompt, cached_result['image_url'])
            
            # 返回干净的结果（不包含内部字段）
            return {
                "status": "success",
                "image_url": cached_result.get("image_url", "")
            }
    
    # 使用结果队列来等待结果
    result_queue = queue.Queue()
    
    # 启动图片生成线程
    generate_image_in_thread(
        prompt, 
        cache_key, 
        lambda result: result_queue.put(result),
        email_to,
        original_query
    )
    
    try:
        # 等待结果，最多等待15秒
        result = result_queue.get(timeout=15)
        
        # 返回干净的结果（不包含内部字段）
        clean_result = {
            "status": result.get("status", "error"),
        }
        
        if "image_url" in result:
            clean_result["image_url"] = result["image_url"]
        if "message" in result:
            clean_result["message"] = result["message"]
            
        elapsed_time = time.time() - start_time
        logger.info(f"图片生成完成，耗时: {elapsed_time:.2f}秒")
        
        # 如果图片生成成功且提供了邮箱，提示用户邮件正在发送
        if clean_result.get("status") == "success" and email_to:
            clean_result["email_message"] = f"图片已生成并正在发送到您的邮箱 {email_to}"
        
        return clean_result
    except queue.Empty:
        # 超时，返回处理中状态
        logger.warning(f"等待图片生成超时 (15秒)，已在后台继续处理")
        
        # 在后台继续处理，但立即返回
        pending_message = f"图片生成仍在进行中，请稍后再查询"
        if email_to:
            pending_message += f"。生成完成后，将自动发送到您的邮箱 {email_to}"
        
        return {
            "status": "pending",
            "message": pending_message,
            "estimated_time": 10
        }
    except Exception as e:
        logger.error(f"等待图片生成结果时出错: {str(e)}")
        logger.error(traceback.format_exc())
        
        return {
            "status": "error",
            "message": f"生成图片时出错: {str(e)[:100]}"
        }

# 测试
if __name__ == "__main__":
    prompt = "一只可爱的小猫"
    result = generate_image_url_tool(prompt)
    print(json.dumps(result, ensure_ascii=False, indent=2))