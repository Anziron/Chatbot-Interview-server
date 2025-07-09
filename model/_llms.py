from langchain_openai import ChatOpenAI
import os
import dotenv
import time
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

dotenv.load_dotenv()

# 自定义异常类，用于区分不同类型的错误
class ModelTimeoutError(Exception):
    """当模型调用超时时抛出"""
    pass

class ModelConnectionError(Exception):
    """当连接到模型服务器失败时抛出"""
    pass

class ModelContentError(Exception):
    """当内容被拒绝或不适当时抛出"""
    pass

# 重试装饰器，针对不同类型的异常采用不同的策略
@retry(
    stop=stop_after_attempt(3),  # 最多尝试3次
    wait=wait_exponential(multiplier=1, min=2, max=10),  # 指数级回退等待
    retry=retry_if_exception_type((requests.exceptions.ConnectionError, 
                                  requests.exceptions.Timeout, 
                                  ModelTimeoutError,
                                  ModelConnectionError)),
    reraise=True
)
def create_model_with_retry(model_name: str, base_url: str, api_key: str, timeout: float) -> ChatOpenAI:
    """创建模型实例，带有重试机制"""
    try:
        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            request_timeout=timeout,
        )
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        print(f"连接模型服务器失败: {str(e)}")
        raise ModelConnectionError(f"无法连接到模型服务器: {str(e)}")
    except Exception as e:
        print(f"创建模型实例时发生未知错误: {str(e)}")
        raise

# 创建模型实例
try:
    model = create_model_with_retry(
        model_name="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        timeout=60.0,  # 增加超时时间到60秒
    )
    
    # 添加错误处理属性
    model.max_retries = 2  # 模型内部重试次数
    
except Exception as e:
    print(f"初始化模型失败: {str(e)}")
    # 创建一个后备模型实例，避免程序崩溃
    # 这里可以使用一个更可靠但可能性能较低的模型作为备用
    try:
        model = ChatOpenAI(
            model="qwen-max",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            request_timeout=30.0,
        )
    except Exception as backup_error:
        print(f"备用模型也初始化失败: {str(backup_error)}")
        # 在这种情况下，我们仍然需要一个model变量，但会在使用时检查其有效性
        model = None
