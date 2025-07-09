import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model._llms import model, ModelTimeoutError, ModelConnectionError, ModelContentError
from langchain.agents import initialize_agent, AgentType, AgentExecutor
try:
    # 尝试导入新版本
    from langchain_core.agents import AgentExecutor as CoreAgentExecutor
    use_langgraph = True
except ImportError:
    use_langgraph = False
    
from _tools._rag._rag_all import search_vector_store
from _tools._search.web_search import web_search
from _agents.basic_agent._functions_prompt import prompt  # 自定义推理提示词
import time
import logging
import traceback
import warnings

# 抑制LangChain弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'agent_errors.log'))  # 输出到文件
    ]
)
logger = logging.getLogger('basic_agent')

# 工具初始化
def tools_init(web_open: bool):
    try:
        tools = [search_vector_store, web_search] if web_open else [search_vector_store]
        return tools
    except Exception as e:
        logger.error(f"工具初始化失败: {str(e)}")
        # 如果工具初始化失败，返回最小可用工具集
        return [search_vector_store]


# Agent 初始化
def agent_init(web_state: bool):
    # 检查模型是否可用
    if model is None:
        logger.error("模型未初始化，无法创建Agent")
        raise ValueError("模型未初始化，请检查模型配置")
        
    try:
        tools = tools_init(web_state)
        
        # 使用with语句临时抑制警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            
            agent = initialize_agent(
                tools=tools,
                llm=model,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                agent_kwargs={"prompt": prompt},
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3,
                memory=None,
                streaming=False
            )
            
            # 包装成 AgentExecutor，开启中间步骤
            return AgentExecutor.from_agent_and_tools(
                agent=agent.agent,
                tools=tools,
                return_intermediate_steps=True,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5,  # 增加最大迭代次数
                early_stopping_method="force",  # 强制在达到最大迭代次数时停止
            )
    except Exception as e:
        logger.error(f"Agent初始化失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# 获取推理与回答
# 设置一个方法，给一个推理参数的判断，若是判断成功则将推理和内容都输出，若是判断失败，则只将内容进行输出
# illation_state: 推理状态
def get_answer_and_illation(query: str, web_state: bool, illation_state: bool):
    max_retries = 3
    retry_count = 0
    retry_delay = 2  # 初始延迟2秒
    
    while retry_count < max_retries:
        try:
            # 检查查询是否为空
            if not query or not query.strip():
                return "您的问题为空，请重新输入。", None
                
            # 检查模型状态
            if model is None:
                return "系统模型服务暂时不可用，请稍后再试。", None
                
            agent = agent_init(web_state)
            
            # 如果不需要推理过程，可以直接调用agent而不返回中间步骤
            if not illation_state:
                # 使用不返回中间步骤的调用方式
                result = agent.invoke({"input": query}, config={"return_intermediate_steps": False})
                return result['output'], None
            
            # 需要推理过程时的处理逻辑
            start_time = time.time()
            result = agent.invoke({"input": query}, config={"return_intermediate_steps": True})
            execution_time = time.time() - start_time
            logger.info(f"Agent执行成功，耗时: {execution_time:.2f}秒")

            # 提取当前问题的原始查询（移除历史上下文）
            current_question = query.split("Current question:")[-1].strip() if "Current question:" in query else query
            
            steps = result.get("intermediate_steps", [])
            illation_lines = [f"问题: {current_question}\n"]

            for action, observation in steps:
                # 清理思考日志中可能包含的历史记录引用
                thought_log = action.log
                if "Previous conversation:" in thought_log:
                    thought_parts = thought_log.split("Current question:", 1)
                    if len(thought_parts) > 1:
                        thought_log = "思考当前问题: " + thought_parts[1].strip()
                    
                illation_lines.append(f"思考: {thought_log}")
                illation_lines.append(f"工具: {action.tool}")
                
                # 清理工具输入中可能包含的历史记录
                tool_input = action.tool_input
                if isinstance(tool_input, str) and "Previous conversation:" in tool_input:
                    tool_input_parts = tool_input.split("Current question:", 1)
                    if len(tool_input_parts) > 1:
                        tool_input = tool_input_parts[1].strip()
                    
                illation_lines.append(f"工具输入: {tool_input}")
                formatted_observation = format_observation(observation)
                illation_lines.append(f"观察结果: {formatted_observation}\n")

            final_answer = result['output']
            illation_text = "\n".join(illation_lines)

            # 检查是否有实际答案内容
            if not final_answer or not final_answer.strip():
                return "抱歉，我无法为您的问题生成有效答案。请尝试重新表述您的问题。", illation_text if illation_state else None

            return final_answer, illation_text
            
        except ModelTimeoutError as e:
            logger.warning(f"模型调用超时 (尝试 {retry_count+1}/{max_retries}): {str(e)}")
            retry_count += 1
            if retry_count >= max_retries:
                return f"抱歉，模型响应超时。请稍后再试。错误详情: {str(e)[:100]}...", None
            time.sleep(retry_delay)
            retry_delay *= 2  # 指数级回退
            
        except ModelConnectionError as e:
            logger.warning(f"模型连接错误 (尝试 {retry_count+1}/{max_retries}): {str(e)}")
            retry_count += 1
            if retry_count >= max_retries:
                return f"抱歉，无法连接到模型服务。请检查网络连接并稍后再试。错误详情: {str(e)[:100]}...", None
            time.sleep(retry_delay)
            retry_delay *= 2
            
        except ModelContentError as e:
            # 内容错误不重试
            logger.error(f"内容错误: {str(e)}")
            return f"抱歉，您的请求包含不适当的内容，无法处理。请修改您的问题后再试。", None
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"调用Agent时发生错误: {error_msg}")
            logger.error(traceback.format_exc())
            
            # 提取当前问题（如果可能）
            current_question = query.split("Current question:")[-1].strip() if "Current question:" in query else query
            
            # 处理内容审核错误
            if "data_inspection_failed" in error_msg or "Output data may contain inappropriate content" in error_msg:
                return "抱歉，内容审核系统阻止了此请求。请修改您的问题后再试。", f"问题: {current_question}\n\n内容审核系统阻止了此请求。"
                
            # 判断是否需要重试
            if "timeout" in error_msg.lower() or "connection" in error_msg.lower() or "network" in error_msg.lower():
                retry_count += 1
                if retry_count >= max_retries:
                    return f"抱歉，系统暂时遇到技术问题。请稍后再试。错误详情: {error_msg[:100]}...", None
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                # 其他错误不重试
                return f"抱歉，处理您的请求时出错。请稍后再试。", f"问题: {current_question}\n\n错误详情: {error_msg[:200]}..."

    # 如果所有重试都失败
    return "抱歉，系统暂时不可用。请稍后再试。", None

# 格式化观察结果
def format_observation(observation):
    try:
        # 若 observation 是单个字符串则直接返回
        if isinstance(observation, str):
            return observation

        # 若 observation 是搜索结果列表
        if isinstance(observation, list) and len(observation) > 0 and isinstance(observation[0], dict):
            formatted = []
            for i, item in enumerate(observation, 1):
                title = item.get("title", "无标题")
                url = item.get("url", "无链接")
                content = item.get("content", "无摘要")
                formatted.append(f"{i}. 【{title}】\n链接: {url}\n摘要: {content}\n")
            return "\n".join(formatted)

        return str(observation)
    except Exception as e:
        logger.error(f"格式化观察结果时出错: {str(e)}")
        return f"[无法格式化结果: {str(e)}]"

# 测试入口
if __name__ == "__main__":
    answer, illation = get_answer_and_illation("你知道有关邓紫棋的信息吗", web_state=True, illation_state=True)
    print("推理过程：\n", illation)
    print("最终答案：\n", answer)
