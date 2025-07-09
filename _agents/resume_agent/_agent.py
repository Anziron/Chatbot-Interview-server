import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model._llms import model
from langchain.schema import SystemMessage, HumanMessage

from _tools._pdf.read import read_pdf
from _agents.resume_agent._functions_prompt import RESUME_SYSTEM_PROMPT

def parse_qa_from_result(result: str):
    questions = []
    answers = []

    in_questions, in_answers = False, False
    current_question = ""  # 用于收集完整的问题内容
    
    for line in result.strip().splitlines():
        line = line.strip()
        # 判断是否进入问题列表区域
        if "问题列表" in line:
            in_questions = True
            in_answers = False
            continue
        # 判断是否进入答案列表区域
        if "标准答案列表" in line:
            in_questions = False
            in_answers = True
            continue
        # 跳过无关符号行
        if line.startswith(("-", "—", "·")) or line == "---":
            continue
        
        # 从问题中提取完整内容
        if in_questions and line:
            # 检查是否是新的问题（以数字开头）
            if len(line) > 1 and line[0].isdigit() and any(c in line[1:3] for c in ['.', '、', ':', '：']):
                # 如果已经有收集的问题，则保存它
                if current_question:
                    questions.append(current_question.strip())
                    current_question = ""
                
                # 去除问题序号，保留完整问题内容
                idx_end = 1
                while idx_end < len(line) and line[idx_end].isdigit():
                    idx_end += 1
                
                if idx_end < len(line) and line[idx_end] in ['.', '、', ':', '：']:
                    content = line[idx_end+1:].strip()
                    current_question = content
                else:
                    current_question = line
            else:
                # 将这行作为上一个问题的延续
                current_question += " " + line
        
        # 处理答案部分
        if in_answers and line:
            # 处理数字序号开头的答案
            if len(line) > 1 and line[0].isdigit():
                idx_end = 1
                while idx_end < len(line) and line[idx_end].isdigit():
                    idx_end += 1
                if idx_end < len(line) and line[idx_end] in ['.', '、', ':', '：']:
                    content = line[idx_end+1:].strip()
                    # 去除类型标签（如 "**背景确认问题参考答案：**"）
                    if content.startswith("**") and "**" in content[2:]:
                        type_end = content.find("**", 2)
                        if type_end != -1 and type_end + 2 < len(content):
                            answers.append(content[type_end+2:].strip())
                        else:
                            answers.append(content)
                    else:
                        answers.append(content)
    
    # 添加最后一个问题（如果有的话）
    if current_question:
        questions.append(current_question.strip())
    
    # 调试信息
    print(f"解析到 {len(questions)} 个问题和 {len(answers)} 个答案")
    for i, q in enumerate(questions):
        print(f"问题 {i+1}: {q}")
    
    return questions, answers


def analyze_resume(resume_path):
    """直接使用LLM读取简历并生成结构化问题与标准答案"""
    print(f"开始分析简历: {resume_path}")
    
    # 先直接读取PDF内容
    try:
        resume_text = read_pdf({"file_path": resume_path})
        if "错误" in resume_text:
            print(f"读取PDF出错: {resume_text}")
            resume_text = "无法读取简历内容"
    except Exception as e:
        print(f"读取PDF异常: {str(e)}")
        resume_text = "无法读取简历内容"
    
    # 使用LLM直接生成问题和答案，而不是通过agent
    messages = [
        SystemMessage(content=RESUME_SYSTEM_PROMPT),
        HumanMessage(content=f"请根据以下简历内容，生成结构化面试问题和标准答案：\n\n{resume_text}")
    ]
    
    result = model(messages).content
    print("LLM 输出：", result)

    questions, answers = parse_qa_from_result(result)
    return resume_text, questions, answers  # 返回文本、问题和标准答案列表


# 测试
if __name__ == "__main__":
    pass