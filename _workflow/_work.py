import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from _workflow._database import checkpointer
from _agents.basic_agent._agent import get_answer_and_illation
from _cache._cache_handle import get_content_from_cache, cache_content
from _token._price import cache_tokens_price, agent_tokens_price
import logging
import time
import traceback
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'workflow_errors.log'))  # 输出到文件
    ]
)
logger = logging.getLogger('workflow')

# 增加重试机制的装饰器
def with_retry(max_retries=3, initial_delay=1, backoff_factor=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            last_exception = None
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"操作失败，进行重试 ({retries+1}/{max_retries}): {str(e)}")
                    retries += 1
                    if retries < max_retries:
                        time.sleep(delay)
                        delay *= backoff_factor
            
            # 如果所有重试都失败，记录错误并返回错误信息
            logger.error(f"所有重试都失败: {str(last_exception)}")
            logger.error(traceback.format_exc())
            raise last_exception
            
        return wrapper
    return decorator

# 节点：调用 agent
def call_agent(state: MessagesState, config: dict):
    start_time = time.time()
    try:
        # 构建带上下文的提示
        messages = state["messages"]
        history = "\n".join([
            f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}"
            for msg in messages[:-1]
        ])
        
        # 将历史记录和当前问题组合
        current_query = messages[-1].content
        query_with_context = f"Previous conversation:\n{history}\n\nCurrent question: {current_query}" if history else current_query

        # 获取联网状态
        web_state = config["configurable"].get("web_state", False) if config else False
        illation_state = config["configurable"].get("illation_state", False) if config else False  
        logger.info(f"联网状态：{web_state}")
        logger.info(f"推理状态：{illation_state}")

        # 使用agent处理（不使用其内部记忆）
        answer, illation = get_answer_and_illation(query_with_context, web_state, illation_state)
        
        # 记录执行时间
        execution_time = time.time() - start_time
        logger.info(f"Agent执行完成，耗时: {execution_time:.2f}秒")
        
        if illation:
            logger.info(f"推理完成，答案长度: {len(answer) if answer else 0}")
            
        # 检查响应是否为空
        if not answer or not answer.strip():
            logger.warning("Agent返回了空响应")
            answer = "抱歉，我无法为您的问题生成有效答案。请尝试重新表述您的问题或稍后再试。"

        # 创建包含状态信息的元数据
        metadata = {
            "status": "completed",
            "execution_time": execution_time,
            "has_illation": illation is not None,
            "timestamp": time.time()
        }

        # 添加 AI 回复到消息流，并附加元数据
        result_message = AIMessage(content=answer, illation=illation, additional_kwargs={"metadata": metadata})
        return {"messages": state["messages"] + [result_message]}
        
    except Exception as e:
        # 记录错误执行时间
        execution_time = time.time() - start_time
        logger.error(f"调用Agent节点发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 创建错误状态元数据
        error_metadata = {
            "status": "error",
            "execution_time": execution_time,
            "error_type": type(e).__name__,
            "timestamp": time.time()
        }
        
        # 返回错误消息，避免工作流崩溃
        error_message = f"抱歉，系统处理您的请求时遇到了问题。错误详情: {str(e)[:100]}..."
        return {"messages": state["messages"] + [AIMessage(content=error_message, additional_kwargs={"metadata": error_metadata})]}

# 构建 LangGraph
try:
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("agent", call_agent)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    # 持久化 + 编译图
    app = workflow.compile(checkpointer=checkpointer)
    logger.info("工作流初始化成功")
except Exception as e:
    logger.critical(f"工作流初始化失败: {str(e)}")
    logger.critical(traceback.format_exc())
    # 创建一个简单的后备工作流
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("fallback", lambda state: {"messages": state["messages"] + [AIMessage(content="系统初始化失败，请联系管理员。")]})
    workflow.add_edge(START, "fallback")
    workflow.add_edge("fallback", END)
    app = workflow.compile()

# 聊天
@with_retry(max_retries=2, initial_delay=1, backoff_factor=2)
def chat(query: str, enable_web: bool, enable_illation: bool, thread_id: str = "abc123"):
    """
    说明：
    1. 如果enable_web为True，则启用联网
    2. 如果enable_illation为True，则启用推理
    3. 如果enable_web和enable_illation都为False，则不启用联网和推理
    """
    start_time = time.time()
    try:
        # 输入验证
        if not query or not query.strip():
            logger.warning("收到空查询")
            return "请输入您的问题", {"price": 0, "tokens": 0, "status": "completed"}, None
        
        # 查询缓存
        cached_answer, cache_illation = get_content_from_cache(query)
        if cached_answer and not enable_web:
            logger.info("[缓存命中]")
            cache_price = cache_tokens_price(query, cached_answer)
            cache_price["status"] = "completed"
            cache_price["source"] = "cache"
            cache_price["time"] = time.time() - start_time
            
            # 只有在启用推理的情况下才返回推理过程
            if enable_illation:
                return cached_answer, cache_price, cache_illation
            else:
                return cached_answer, cache_price, None

        # 没有命中则调用 agent
        logger.info("缓存未命中，调用 agent")
        logger.info(f"使用线程ID: {thread_id}")

        # 配置
        config = {"configurable": {"thread_id": thread_id, "web_state": enable_web,"illation_state": enable_illation}}

        # 调用 agent
        try:
            res = app.invoke({"messages": [HumanMessage(content=query)]}, config)
            # 确保获取到最后一条消息
            if res and "messages" in res and len(res["messages"]) > 0:
                last_message = res["messages"][-1]
                
                if isinstance(last_message, AIMessage):
                    agent_answer = last_message.content
                    agent_price = agent_tokens_price(query, agent_answer)
                    agent_price["status"] = "completed"
                    agent_price["source"] = "agent"
                    agent_price["time"] = time.time() - start_time
                    
                    # 获取元数据和推理过程
                    metadata = getattr(last_message, 'additional_kwargs', {}).get('metadata', {})
                    agent_illation = getattr(last_message, 'illation', None)
                    
                    # 检查是否有实际答案内容
                    if agent_answer and agent_answer.strip():
                        # 将答案和推理过程（如果有）写入缓存
                        try:
                            cache_content(query, agent_answer, agent_price["price"], agent_price["tokens"], agent_illation)
                        except Exception as cache_err:
                            logger.error(f"缓存写入失败: {str(cache_err)}")
                        
                        # 只有在启用推理的情况下才返回推理过程
                        if enable_illation:
                            return agent_answer, agent_price, agent_illation
                        else:
                            return agent_answer, agent_price, None
                else:
                    logger.warning(f"最后一条消息不是AI消息: {type(last_message)}")
            else:
                logger.warning("返回结果没有消息列表或消息列表为空")
            
            # 如果没有获取到有效回答
            logger.warning("未能从agent获取有效回答")
            return "抱歉，我无法为您的问题生成有效答案。请尝试重新表述您的问题。", {"price": 0, "tokens": 0, "status": "error", "time": time.time() - start_time}, None
            
        except Exception as agent_error:
            logger.error(f"调用agent时出错: {str(agent_error)}")
            logger.error(traceback.format_exc())
            return f"抱歉，处理您的请求时出错: {str(agent_error)[:100]}...", {"price": 0, "tokens": 0, "status": "error", "error": str(agent_error), "time": time.time() - start_time}, None
            
    except Exception as e:
        logger.error(f"聊天函数发生未处理的异常: {str(e)}")
        logger.error(traceback.format_exc())
        return f"系统发生错误，请稍后再试。错误信息: {str(e)[:100]}...", {"price": 0, "tokens": 0, "status": "error", "error": str(e), "time": time.time() - start_time}, None

# 测试
if __name__ == "__main__":
    # res1 = chat("你好我叫bob")
    # print(res1)
    
    # res2, price = chat("人的嗅觉在午后什么时间段不灵敏，与什么有关？")
    # print(f"内容：{res2},{price}")
    # clear_cache()
    res3, price, illation = chat("日本有哪些城市", enable_web=True, enable_illation=True)
    print(f"内容：{res3}")
    print(f"推理：{illation}")
    print(f"价格：{price}")
    pass
