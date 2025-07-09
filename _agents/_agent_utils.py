import sys, os
import time
import logging
import traceback
import json
import re
from typing import Dict, Any, List, Optional, Callable, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
from functools import wraps

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'agents_errors.log'))  # 输出到文件
    ]
)
logger = logging.getLogger('agent_utils')

# 自定义异常类
class AgentExecutionError(Exception):
    """Agent执行错误基类"""
    pass

class JSONParseError(AgentExecutionError):
    """JSON解析错误"""
    pass

class ToolExecutionError(AgentExecutionError):
    """工具执行错误"""
    pass

class ModelTimeoutError(AgentExecutionError):
    """模型调用超时错误"""
    pass

# 性能计时装饰器
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} 执行耗时: {execution_time:.2f}秒")
        return result
    return wrapper

# 通用重试装饰器
def with_retry(max_retries=3, initial_delay=1, backoff_factor=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            last_exception = None
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, ModelTimeoutError) as e:
                    # 这些异常类型会触发重试
                    last_exception = e
                    logger.warning(f"网络或模型错误，进行重试 ({retries+1}/{max_retries}): {str(e)}")
                except (JSONParseError) as e:
                    # JSON解析错误也会触发重试
                    last_exception = e
                    logger.warning(f"JSON解析错误，进行重试 ({retries+1}/{max_retries}): {str(e)}")
                except Exception as e:
                    # 其他异常不会重试，直接抛出
                    logger.error(f"发生不可重试的异常: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
                
                retries += 1
                if retries < max_retries:
                    logger.info(f"等待 {delay} 秒后重试...")
                    time.sleep(delay)
                    delay *= backoff_factor
            
            # 如果所有重试都失败，记录错误并抛出最后的异常
            logger.error(f"达到最大重试次数 ({max_retries})，操作失败")
            raise last_exception or Exception("未知错误导致操作失败")
            
        return wrapper
    return decorator

# JSON修复工具类
class JSONFixer:
    @staticmethod
    def try_fix_json(json_str: str) -> Optional[str]:
        """尝试修复JSON字符串"""
        logger.info(f"尝试修复JSON: {json_str[:100]}...")
        
        if not json_str:
            return None
            
        # 检查是否是部分JSON字符串
        if not json_str.strip().startswith('{'):
            # 尝试从字符串中提取JSON部分
            match = re.search(r'(\{.*)', json_str)
            if match:
                json_str = match.group(1)
        
        # 检查是否缺少结束大括号
        if json_str.count('{') > json_str.count('}'):
            json_str = json_str + '}'
        
        # 尝试修复常见的URL格式问题
        json_str = json_str.replace('\\"', '"')
        
        # 尝试提取JSON部分
        match = re.search(r'(\{.*?\})', json_str, re.DOTALL)
        if match:
            json_str = match.group(1)
        
        # 处理可能的截断JSON
        if '"arguments": "{' in json_str:
            # 可能是嵌套JSON被截断
            try:
                # 尝试提取和修复嵌套的JSON
                nested_match = re.search(r'"arguments": "(\{.*?)"', json_str, re.DOTALL)
                if nested_match:
                    nested_json = nested_match.group(1)
                    # 确保嵌套JSON完整
                    if nested_json.count('{') > nested_json.count('}'):
                        nested_json = nested_json + '}'
                    # 替换回原始字符串
                    json_str = json_str.replace(nested_match.group(1), nested_json)
            except:
                pass
        
        # 尝试解析修复后的JSON
        try:
            logger.info(f"修复后的JSON: {json_str[:100]}...")
            parsed_json = json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析仍然失败: {e}")
            return None
            
    @staticmethod
    def extract_json_from_error(error_message: str) -> Optional[str]:
        """从错误消息中提取JSON字符串"""
        # 尝试不同的提取模式
        patterns = [
            r"```json\s*(.*?)\s*```",  # Markdown代码块格式
            r'(\{.*?\})',  # 简单的大括号匹配
            r'"arguments": "(\{.*?\})"'  # 嵌套在arguments字段中
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_message, re.DOTALL)
            if match:
                return match.group(1)
                
        return None

# Agent结果格式化器
class AgentResultFormatter:
    @staticmethod
    def format_success(output: str, steps: List = None) -> Dict[str, Any]:
        """格式化成功的Agent结果"""
        return {
            "success": True,
            "output": output,
            "intermediate_steps": steps or []
        }
        
    @staticmethod
    def format_error(error_message: str, error_type: str = "general") -> Dict[str, Any]:
        """格式化错误的Agent结果"""
        # 根据错误类型提供友好的错误消息
        friendly_messages = {
            "json": "处理过程中出现JSON解析错误。这通常是由于复杂数据格式或特殊字符导致的。\n\n"
                   "请尝试以下解决方案:\n"
                   "1. 使用更简单的描述方式\n"
                   "2. 避免使用特殊符号\n"
                   "3. 确保提供了必要的信息\n",
            "timeout": "处理请求超时。服务器可能正在经历高负载。\n\n"
                      "请尝试以下解决方案:\n"
                      "1. 稍后再试\n"
                      "2. 简化您的请求\n"
                      "3. 如果问题持续存在，请联系管理员\n",
            "network": "网络连接错误。无法连接到必要的服务。\n\n"
                      "请尝试以下解决方案:\n"
                      "1. 检查您的网络连接\n"
                      "2. 稍后再试\n"
                      "3. 如果问题持续存在，请联系管理员\n",
            "permission": "权限错误。缺少访问所需资源的权限。\n\n"
                         "请联系管理员解决此问题。\n",
            "general": "处理过程中出现错误。\n\n"
                      "请尝试简化您的请求或稍后再试。如果问题持续存在，请联系管理员。\n"
        }
        
        base_message = friendly_messages.get(error_type, friendly_messages["general"])
        detailed_message = f"{base_message}\n错误详情: {error_message}"
        
        return {
            "success": False,
            "output": detailed_message,
            "intermediate_steps": [],
            "error_type": error_type
        }

# 安全的工具执行器
class SafeToolExecutor:
    @staticmethod
    @with_retry(max_retries=2)
    def execute_tool(tool_func: Callable, **kwargs) -> Any:
        """安全地执行工具函数"""
        try:
            start_time = time.time()
            result = tool_func(**kwargs)
            execution_time = time.time() - start_time
            logger.info(f"工具 {tool_func.__name__} 执行成功，耗时: {execution_time:.2f}秒")
            return result
        except Exception as e:
            logger.error(f"工具 {tool_func.__name__} 执行失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise ToolExecutionError(f"工具执行失败: {str(e)}")

# 预处理数据提取器
class PreprocessorExtractor:
    @staticmethod
    def extract_preprocessed_data(query: str, marker: str) -> tuple:
        """从查询中提取预处理数据"""
        preprocessed_data = None
        if marker in query:
            # 提取预处理数据
            match = re.search(f"\\[{marker}：(.*?)\\]", query)
            if match:
                preprocessed_data = match.group(1)
                # 从查询中移除预处理数据标记，保留用户的实际问题
                clean_query = re.sub(f"\\[{marker}：.*?\\]\\s*", "", query)
                logger.info(f"检测到预处理数据，提取用户问题: {clean_query}")
                return preprocessed_data, clean_query
                
        return None, query 