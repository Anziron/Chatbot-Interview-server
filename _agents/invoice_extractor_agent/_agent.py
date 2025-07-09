import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model._llms import model, ModelTimeoutError, ModelConnectionError
from langchain.agents import AgentExecutor, create_openai_functions_agent
import json
import re
import logging
import time
import traceback

from _tools._identify.invoice import identify_invoice
from _agents.invoice_extractor_agent._functions_prompt import functions_prompt
from _tools._excel.excel_tool import simple_excel_export, table_to_excel
from _agents._agent_utils import (
    logger, timing_decorator, with_retry, JSONFixer, 
    AgentResultFormatter, SafeToolExecutor, JSONParseError,
    PreprocessorExtractor
)

# 配置日志
logger = logging.getLogger('invoice_extractor_agent')

# Agent 初始化
@timing_decorator
def agent_init():
    try:
        tools = [
            identify_invoice, 
            simple_excel_export, 
            table_to_excel,
        ]
        
        # 创建OpenAI Functions Agent
        agent = create_openai_functions_agent(
            llm=model,
            tools=tools,
            prompt=functions_prompt
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            return_intermediate_steps=True,
            verbose=True,
            max_iterations=10,  # 减少最大迭代次数，避免过长时间运行
            handle_parsing_errors=True,
            max_execution_time=None
        )
    except Exception as e:
        logger.error(f"初始化invoice_extractor_agent失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# 安全执行Agent
@timing_decorator
@with_retry(max_retries=2, initial_delay=1, backoff_factor=2)
def safe_execute_agent(agent, query):
    """安全地执行Agent，带有重试机制"""
    try:
        start_time = time.time()
        result = agent.invoke({"input": query})
        execution_time = time.time() - start_time
        logger.info(f"Agent执行成功，耗时: {execution_time:.2f}秒")
        return result
    except Exception as e:
        if "timeout" in str(e).lower():
            logger.error(f"Agent执行超时: {str(e)}")
            raise ModelTimeoutError(f"模型响应超时: {str(e)}")
        elif "network" in str(e).lower() or "connection" in str(e).lower():
            logger.error(f"Agent网络连接错误: {str(e)}")
            raise ModelConnectionError(f"网络连接错误: {str(e)}")
        else:
            logger.error(f"Agent执行出错: {str(e)}")
            logger.error(traceback.format_exc())
            raise

# 直接执行工具
def direct_tool_execution(error_message, json_str, tool_name, query):
    """当Agent执行失败时，尝试直接执行工具"""
    try:
        if "identify_invoice" in tool_name.lower():
            # 提取图片路径
            img_path_match = re.search(r'(([A-Za-z]:)?[/\\].*?\.(png|jpg|jpeg|gif))', query)
            if img_path_match:
                img_path = img_path_match.group(1)
                logger.info(f"尝试直接调用发票识别工具: {img_path}")
                result = SafeToolExecutor.execute_tool(identify_invoice, file_path=img_path)
                return AgentResultFormatter.format_success(
                    f"已自动修复问题并成功识别发票。\n\n识别结果: {result}"
                )
        elif "simple_excel_export" in tool_name.lower():
            params = json.loads(json_str)
            logger.info(f"尝试直接调用Excel导出工具: {params}")
            result = SafeToolExecutor.execute_tool(simple_excel_export, **params)
            return AgentResultFormatter.format_success(
                f"已自动修复问题并成功导出到Excel: {result}"
            )
        elif "table_to_excel" in tool_name.lower():
            params = json.loads(json_str)
            logger.info(f"尝试直接调用表格导出工具: {params}")
            result = SafeToolExecutor.execute_tool(table_to_excel, **params)
            return AgentResultFormatter.format_success(
                f"已自动修复问题并成功保存表格到Excel: {result}"
            )
    except Exception as e:
        logger.error(f"直接工具执行失败: {str(e)}")
        logger.error(traceback.format_exc())
    
    return None

@timing_decorator
def run_agent(query: str):
    request_id = f"invoice_{int(time.time())}"
    logger.info(f"开始处理请求 {request_id}: {query}")
    
    try:
        # 检查是否有预处理的发票数据
        preprocessed_data, clean_query = PreprocessorExtractor.extract_preprocessed_data(
            query, "已识别发票数据"
        )
        
        # 初始化Agent
        agent = agent_init()
        
        # 如果有预处理数据，直接在查询中使用
        if preprocessed_data:
            logger.info(f"使用预处理的发票数据，长度: {len(preprocessed_data)}")
            input_query = f"发票数据: {preprocessed_data}\n用户问题: {clean_query}"
        else:
            input_query = query
        
        # 安全执行Agent
        result = safe_execute_agent(agent, input_query)
        logger.info(f"请求 {request_id} 处理成功")
        return result
    except JSONParseError as e:
        logger.error(f"JSON解析错误 - 请求 {request_id}: {str(e)}")
        return AgentResultFormatter.format_error(str(e), "json")
    except ModelTimeoutError as e:
        logger.error(f"模型超时 - 请求 {request_id}: {str(e)}")
        return AgentResultFormatter.format_error(str(e), "timeout")
    except ModelConnectionError as e:
        logger.error(f"网络连接错误 - 请求 {request_id}: {str(e)}")
        return AgentResultFormatter.format_error(str(e), "network")
    except Exception as e:
        error_message = str(e)
        logger.error(f"处理请求 {request_id} 出错: {error_message}")
        logger.error(traceback.format_exc())
        
        # 处理JSON解析错误
        if "not valid JSON" in error_message:
            logger.info("检测到JSON解析错误，尝试修复...")
            
            # 从错误消息中提取JSON字符串
            json_str = JSONFixer.extract_json_from_error(error_message)
            if json_str:
                # 尝试修复JSON
                fixed_json = JSONFixer.try_fix_json(json_str)
                if fixed_json:
                    # 根据错误消息判断是哪个工具出了问题
                    if "identify_invoice" in error_message:
                        tool_name = "identify_invoice"
                    elif "simple_excel_export" in error_message:
                        tool_name = "simple_excel_export"
                    else:
                        tool_name = "table_to_excel"
                    
                    # 尝试直接执行相应的工具
                    direct_result = direct_tool_execution(error_message, fixed_json, tool_name, query)
                    if direct_result:
                        return direct_result
            
            # 如果直接执行失败，返回友好的错误信息
            return AgentResultFormatter.format_error(
                "JSON解析错误。请使用更简单的描述或避免特殊字符。", 
                "json"
            )
        
        # 处理图片处理错误
        if "图片无法识别" in error_message or "image" in error_message.lower() and "error" in error_message.lower():
            return AgentResultFormatter.format_error(
                "无法处理图片。请确保图片清晰可见且包含有效的发票信息。",
                "permission"
            )
        
        # 处理其他类型的错误
        return AgentResultFormatter.format_error(error_message)

# 测试入口
if __name__ == "__main__":
    res = run_agent("读取发票图片，提取信息并保存到Excel")
    print(res)
