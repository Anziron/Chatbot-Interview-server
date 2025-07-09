from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

# 系统提示词
system_message = """你是一位聪明的 AI 助手，善于按照特定顺序逐步调用工具来完成复杂任务。

特别注意：
1. 对于复杂任务，严格按照以下步骤顺序执行：
   - 首先分析整个任务流程，确定需要调用的工具及其顺序
   - 然后一步一步执行，每次只调用一个工具
   - 将前一个工具的输出作为下一个工具的输入
   - 不要试图在一个步骤中完成多个操作

2. 对于"读取发票并保存到Excel"这类任务：
   - 第一步：使用identify_invoice工具识别图片内容
   - 第二步：使用simple_excel_export或table_to_excel工具将识别结果保存到Excel

3. 确保每个工具的输入格式正确，特别是JSON格式的数据
"""

# 创建ChatPromptTemplate
functions_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_message),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 使用示例
# if __name__ == "__main__":
#     print("OpenAI Functions Agent提示词已创建，可以在agent.py中使用。")
#     print("示例用法:")
#     print('''
# from _agents._functions_prompt import functions_prompt
# from langchain.agents import create_openai_functions_agent

# agent = create_openai_functions_agent(
#     llm=model,
#     tools=tools,
#     prompt=functions_prompt
# )

# agent_executor = AgentExecutor.from_agent_and_tools(
#     agent=agent,
#     tools=tools,
#     return_intermediate_steps=True,
#     verbose=True
# )
#     ''') 