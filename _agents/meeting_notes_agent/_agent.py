import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model._llms import model, ModelTimeoutError, ModelConnectionError
# 移除原有的AgentExecutor导入
from langchain.agents import create_openai_functions_agent
import json
import re
import dotenv
import logging
import time
import traceback

dotenv.load_dotenv()
from _tools._identify.meeting_minutes import identify_meeting_minutes
from _agents.meeting_notes_agent._functions_prompt import functions_prompt
from _tools._feishu.feishu_tool import save_to_feishu_doc
from _tools._email.send_email import send_email, send_plain_email
from _agents._agent_utils import (
    logger, timing_decorator, with_retry, JSONFixer, 
    AgentResultFormatter, SafeToolExecutor, JSONParseError,
    PreprocessorExtractor
)

# 配置日志
logger = logging.getLogger('meeting_notes_agent')

# 添加一个简单的自定义AgentExecutor实现
class SimpleAgentExecutor:
    """一个简单的AgentExecutor替代实现，避免LangChain版本兼容性问题"""
    
    def __init__(self, agent, tools, verbose=False, max_iterations=10, return_intermediate_steps=False):
        self.agent = agent
        self.tools = {tool.name: tool for tool in tools}
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.return_intermediate_steps = return_intermediate_steps
        
    def fix_json_format(self, json_str):
        """修复不完整或格式错误的JSON字符串"""
        if not json_str:
            return json_str
            
        # 检查是否缺少结束的引号和大括号
        if not json_str.endswith('"}') and '"' in json_str and '{' in json_str:
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
            
            # 尝试修复常见错误
            # 1. 处理转义字符问题
            json_str = json_str.replace('\\"', '"').replace('\\\\', '\\')
            
            # 2. 处理缺少引号的键
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)
            
            # 3. 处理值中的特殊字符
            json_str = re.sub(r':\s*([^",\s\{\}\[\]]+)', r': "\1"', json_str)
            
            # 4. 处理多余的逗号
            json_str = re.sub(r',\s*}', '}', json_str)
            
            try:
                # 再次尝试解析
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                # 如果仍然失败，尝试提取关键参数
                # 例如，从send_email工具调用中提取recipient, subject, body
                if "send_email" in json_str:
                    recipient_match = re.search(r'"(recipient|to)":\s*"([^"]*)"', json_str, re.IGNORECASE)
                    subject_match = re.search(r'"subject":\s*"([^"]*)"', json_str)
                    body_match = re.search(r'"(body|content)":\s*"([^"]*)"', json_str, re.IGNORECASE)
                    
                    if recipient_match and subject_match and body_match:
                        recipient = recipient_match.group(2)
                        subject = subject_match.group(1)
                        body = body_match.group(2)
                        
                        # 构建正确格式的JSON
                        fixed_json = f'{{"recipient": "{recipient}", "subject": "{subject}", "body": "{body}"}}'
                        return fixed_json
                
                # 如果无法修复，返回原始字符串
                return json_str
        
    def invoke(self, inputs):
        """执行agent推理和工具调用循环"""
        intermediate_steps = []
        iterations = 0
        
        while iterations < self.max_iterations:
            if self.verbose:
                logger.info(f"迭代 {iterations+1}/{self.max_iterations}")
            
            # 调用agent获取下一个动作
            try:
                agent_output = self.agent.invoke({
                    "input": inputs.get("input", ""),
                    "intermediate_steps": intermediate_steps
                })
            except Exception as e:
                logger.error(f"Agent调用出错: {str(e)}")
                # 如果是最后一次迭代，直接返回错误
                if iterations == self.max_iterations - 1:
                    return {
                        "output": f"处理过程中出现错误: {str(e)}",
                        "intermediate_steps": intermediate_steps if self.return_intermediate_steps else []
                    }
                # 否则尝试继续
                iterations += 1
                continue
            
            # 检查agent_output的类型并相应处理
            if hasattr(agent_output, "tool"):
                # 这是一个AgentActionMessageLog对象
                tool_name = agent_output.tool
                tool_args = agent_output.tool_input
                is_final_answer = False
            elif hasattr(agent_output, "return_values"):
                # 这是一个最终答案
                is_final_answer = True
                result = {
                    "output": agent_output.return_values.get("output", ""),
                    "intermediate_steps": intermediate_steps if self.return_intermediate_steps else []
                }
                return result
            elif isinstance(agent_output, dict):
                # 标准字典格式
                tool_name = agent_output.get("tool")
                tool_args = agent_output.get("tool_input", {})
                is_final_answer = "tool" not in agent_output
            else:
                # 未知格式，尝试作为最终答案处理
                logger.warning(f"未知的agent_output格式: {type(agent_output)}")
                is_final_answer = True
                result = {
                    "output": str(agent_output),
                    "intermediate_steps": intermediate_steps if self.return_intermediate_steps else []
                }
                return result
            
            # 如果是最终答案，返回结果
            if is_final_answer:
                if self.verbose:
                    logger.info("Agent没有指定工具，完成执行")
                
                # 尝试从不同格式中提取输出
                output = ""
                if hasattr(agent_output, "return_values"):
                    output = agent_output.return_values.get("output", "")
                elif isinstance(agent_output, dict):
                    output = agent_output.get("output", "")
                else:
                    output = str(agent_output)
                
                result = {
                    "output": output,
                    "intermediate_steps": intermediate_steps if self.return_intermediate_steps else []
                }
                return result
            
            # 处理工具调用
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                
                # 执行工具
                if self.verbose:
                    logger.info(f"执行工具: {tool_name} 参数: {tool_args}")
                
                try:
                    # 检查工具输入类型并转换为字典(如果是字符串)
                    if isinstance(tool_args, str):
                        try:
                            # 尝试修复JSON格式
                            fixed_json_str = self.fix_json_format(tool_args)
                            tool_args = json.loads(fixed_json_str)
                        except json.JSONDecodeError:
                            # 如果仍然无法解析，使用简单的输入
                            tool_args = {"input": tool_args}
                    
                    # 特殊处理send_email工具
                    if tool_name == "send_email":
                        # 确保参数格式正确
                        if isinstance(tool_args, dict):
                            # 检查必要参数
                            recipient = tool_args.get("recipient") or tool_args.get("to")
                            subject = tool_args.get("subject")
                            body = tool_args.get("body") or tool_args.get("content")
                            
                            if recipient and subject and body:
                                # 尝试使用send_plain_email而不是send_email
                                logger.info(f"使用send_plain_email发送邮件到: {recipient}")
                                tool_output = send_plain_email(recipient, subject, body)
                            else:
                                logger.error(f"邮件参数不完整: {tool_args}")
                                tool_output = "邮件参数不完整，无法发送"
                        else:
                            logger.error(f"邮件参数格式错误: {tool_args}")
                            tool_output = "邮件参数格式错误，无法发送"
                    # 特殊处理save_to_feishu_doc工具
                    elif tool_name == "save_to_feishu_doc":
                        # 检查参数
                        if isinstance(tool_args, dict) and "content" in tool_args:
                            content = tool_args["content"]
                            logger.info(f"调用save_to_feishu_doc工具，内容长度: {len(str(content))}")
                            tool_output = save_to_feishu_doc(content)
                        else:
                            logger.error(f"飞书文档保存参数格式错误: {tool_args}")
                            tool_output = "飞书文档保存参数格式错误，无法保存"
                    else:
                        # 调用其他工具函数
                        tool_output = tool(**tool_args)
                    
                    # 记录工具调用步骤
                    intermediate_steps.append((agent_output, tool_output))
                    
                    if self.verbose:
                        logger.info(f"工具输出: {tool_output}")
                except Exception as e:
                    logger.error(f"工具执行错误: {str(e)}")
                    logger.error(traceback.format_exc())
                    intermediate_steps.append((agent_output, f"Error: {str(e)}"))
            else:
                # 工具不存在
                logger.warning(f"未找到工具: {tool_name}")
                intermediate_steps.append((agent_output, f"Error: Tool '{tool_name}' not found"))
            
            iterations += 1
        
        # 达到最大迭代次数
        logger.warning(f"达到最大迭代次数: {self.max_iterations}")
        result = {
            "output": "达到最大迭代次数限制，未能完成任务。",
            "intermediate_steps": intermediate_steps if self.return_intermediate_steps else []
        }
        return result

# Agent 初始化
@timing_decorator
def agent_init():
    try:
        tools = [
            identify_meeting_minutes, 
            save_to_feishu_doc,
            send_email,
        ]
        
        # 创建OpenAI Functions Agent
        agent = create_openai_functions_agent(
            llm=model,
            tools=tools,
            prompt=functions_prompt
        )
        
        # 使用自定义的SimpleAgentExecutor替代原有的AgentExecutor
        return SimpleAgentExecutor(
            agent=agent,
            tools=tools,
            return_intermediate_steps=True,
            verbose=True,
            max_iterations=10
        )
    except Exception as e:
        logger.error(f"初始化meeting_notes_agent失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# 安全执行Agent
@timing_decorator
@with_retry(max_retries=2, initial_delay=1, backoff_factor=2)
def safe_execute_agent(agent, query):
    """安全地执行Agent，带有重试机制"""
    try:
        start_time = time.time()
        # 确保使用SimpleAgentExecutor的invoke方法
        if isinstance(agent, SimpleAgentExecutor):
            result = agent.invoke({"input": query})
        else:
            logger.warning("Agent不是SimpleAgentExecutor实例，尝试使用标准调用")
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
        if "identify_meeting_minutes" in tool_name.lower():
            # 提取图片路径
            img_path_match = re.search(r'(([A-Za-z]:)?[/\\].*?\.(png|jpg|jpeg|gif))', query)
            if img_path_match:
                img_path = img_path_match.group(1)
                logger.info(f"尝试直接调用会议纪要识别工具: {img_path}")
                result = SafeToolExecutor.execute_tool(identify_meeting_minutes, file_path=img_path)
                return AgentResultFormatter.format_success(
                    f"已自动修复问题并成功识别会议纪要。\n\n识别结果: {result}"
                )
        elif "save_to_feishu_doc" in tool_name.lower():
            params = json.loads(json_str)
            logger.info(f"尝试直接调用飞书文档保存工具: {params}")
            result = SafeToolExecutor.execute_tool(save_to_feishu_doc, **params)
            return AgentResultFormatter.format_success(
                f"已自动修复问题并成功保存到飞书文档: {result}"
            )
        elif "send_email" in tool_name.lower():
            params = json.loads(json_str)
            if "recipient" in params and "subject" in params and "content" in params:
                logger.info(f"尝试直接调用邮件发送工具: {params}")
                result = SafeToolExecutor.execute_tool(
                    send_email,
                    recipient=params["recipient"],
                    subject=params["subject"],
                    content=params["content"],
                    image_url=params.get("image_url", "")
                )
                return AgentResultFormatter.format_success(
                    f"已自动修复问题并成功发送邮件: {result}"
                )
    except Exception as e:
        logger.error(f"直接工具执行失败: {str(e)}")
        logger.error(traceback.format_exc())
    
    return None

# 检查环境变量配置
def check_environment_vars():
    """检查飞书API所需的环境变量是否配置"""
    required_vars = ["FEISHU_APP_ID", "FEISHU_APP_SECRET"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"缺少飞书API所需的环境变量: {', '.join(missing_vars)}")
        return False
    return True

@timing_decorator
def run_agent(query: str):
    request_id = f"meeting_{int(time.time())}"
    logger.info(f"开始处理请求 {request_id}: {query}")
    
    try:
        # 检查是否有预处理的会议纪要数据
        preprocessed_data, clean_query = PreprocessorExtractor.extract_preprocessed_data(
            query, "已识别会议纪要"
        )
        
        # 初始化Agent
        agent = agent_init()
        
        # 提前检查环境变量
        feishu_configured = check_environment_vars()
        if not feishu_configured and "飞书" in query:
            logger.warning("飞书API环境变量未配置，但用户请求涉及飞书")
        
        # 如果有预处理数据，直接在查询中使用
        if preprocessed_data:
            logger.info(f"使用预处理的会议纪要数据，长度: {len(preprocessed_data)}")
            input_query = f"会议纪要数据: {preprocessed_data}\n用户问题: {clean_query}"
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
                    if "identify_meeting_minutes" in error_message:
                        tool_name = "identify_meeting_minutes"
                    elif "save_to_feishu_doc" in error_message:
                        tool_name = "save_to_feishu_doc"
                    else:
                        tool_name = "send_email"
                    
                    # 尝试直接执行相应的工具
                    direct_result = direct_tool_execution(error_message, fixed_json, tool_name, query)
                    if direct_result:
                        return direct_result
            
            # 如果直接执行失败，返回友好的错误信息
            return AgentResultFormatter.format_error(
                "JSON解析错误。请使用更简单的描述或避免特殊字符。", 
                "json"
            )
        
        # 处理环境变量配置错误
        if "APP_ID" in error_message and "APP_SECRET" in error_message:
            return AgentResultFormatter.format_error(
                "保存到飞书文档失败: 服务器未配置飞书应用的APP_ID和APP_SECRET环境变量。\n"
                "已识别会议纪要内容，但无法保存到飞书文档。请联系管理员配置相关环境变量。",
                "permission"
            )
        
        # 处理图片处理错误
        if "图片无法识别" in error_message or "image" in error_message.lower() and "error" in error_message.lower():
            return AgentResultFormatter.format_error(
                "无法处理图片。请确保图片清晰可见且包含有效的会议纪要信息。",
                "permission"
            )
        
        # 处理其他类型的错误
        return AgentResultFormatter.format_error(error_message)

# 测试入口
if __name__ == "__main__":
    res = run_agent("读取会议纪要图片，保存到飞书文档，并且发送到example@example.com邮箱")
    print(res)
