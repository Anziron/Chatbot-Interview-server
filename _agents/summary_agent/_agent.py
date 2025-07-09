import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain.schema import SystemMessage, HumanMessage
from _agents.summary_agent._functions_prompt import SUMMARY_EVAL_PROMPT
from model._llms import model
import json
import re

def generate_summary_and_score(questions, user_answers, resume_text, standard_answers):
    try:
        # 确保三个列表长度一致
        if len(questions) != len(user_answers) or len(questions) != len(standard_answers):
            print(f"警告: 问题数量({len(questions)})、回答数量({len(user_answers)})和标准答案数量({len(standard_answers)})不一致")
            # 取最小长度
            min_len = min(len(questions), len(user_answers), len(standard_answers))
            questions = questions[:min_len]
            user_answers = user_answers[:min_len]
            standard_answers = standard_answers[:min_len]
        
        # 构建格式化的问题、回答和标准答案
        qa_pairs = []
        for i, (q, a, sa) in enumerate(zip(questions, user_answers, standard_answers)):
            # 去除问题中可能的markdown格式，但保留完整问题内容
            clean_q = q
            if "**" in q:
                # 提取问题全文，包括类型和具体内容
                clean_q = q.replace("**", "")
            
            qa_pairs.append(f"问题{i+1}: {clean_q}\n用户回答{i+1}: {a}\n标准答案{i+1}: {sa}\n")
            
        input_text = "【简历内容】\n" + resume_text + "\n\n【面试问答记录】\n" + "\n".join(qa_pairs)

        # 构建更明确的提示
        prompt = f"""
        你是一个专业的面试评分与总结助手。请根据以下内容进行评分和总结：

        {input_text}

        请按照以下JSON格式返回结果，不要包含任何其他内容：
        {{
        "summary": "对候选人整体表现的总结",
        "scores": [
            {{
            "question": "问题1",
            "score": 分数,
            "standard_answer": "标准答案1",
            "user_answer": "用户回答1",
            "comment": "评语1"
            }},
            ... 其他问题的评分 ...
        ]
        }}
        """

        messages = [
            SystemMessage(content=SUMMARY_EVAL_PROMPT),
            HumanMessage(content=prompt)
        ]

        print("正在生成总结和评分...")
        response = model(messages)
        
        # 保存原始响应内容
        raw_content = response.content
        print(f"原始响应: {raw_content}")
        
        # 尝试多种方式解析JSON
        result = None
        try:
            # 方法1：直接解析
            result = json.loads(raw_content)
        except Exception as e1:
            print(f"直接解析JSON失败: {str(e1)}")
            try:
                # 方法2：提取JSON部分
                json_match = re.search(r'```json\s*(.*?)\s*```', raw_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    result = json.loads(json_str)
                else:
                    # 方法3：寻找花括号包围的JSON
                    json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        result = json.loads(json_str)
            except Exception as e2:
                print(f"提取JSON失败: {str(e2)}")
        
        # 如果成功解析JSON
        if result:
            summary = result.get("summary", "无法提取总结内容")
            scores = result.get("scores", [])
            
            # 确保scores包含所有必要字段并与原始问题保持一致
            formatted_scores = []
            for i, q in enumerate(questions):
                # 查找对应的评分项 - 修改匹配逻辑，提高匹配率
                # 1. 先尝试完全匹配
                score_item = next((s for s in scores if s.get("question", "").strip() == q.strip()), None)
                
                # 2. 如果没找到，尝试部分内容匹配
                if not score_item:
                    # 提取问题的核心内容（去掉问题类型标签）
                    core_q = q.split("**")[-1].strip() if "**" in q else q.strip()
                    # 尝试模糊匹配
                    score_item = next((s for s in scores if core_q in s.get("question", "").strip() 
                                     or s.get("question", "").strip() in core_q
                                     or any(keyword in s.get("question", "").strip() for keyword in core_q.split()[:5] if len(keyword) > 3)), None)
                
                # 3. 如果仍然没找到，检查第几个问题
                if not score_item and i < len(scores):
                    score_item = scores[i]  # 假设顺序相同
                
                if not score_item:
                    # 如果找不到匹配项，则创建默认项
                    score_item = {
                        "question": q,
                        "score": 0 if user_answers[i] == "未回答" else 5,
                        "standard_answer": standard_answers[i],
                        "user_answer": user_answers[i],
                        "comment": "未回答此问题" if user_answers[i] == "未回答" else "未找到此问题的评分，使用默认评分"
                    }
                else:
                    # 确保所有必要字段存在，并使用完整的问题文本
                    score_item = score_item.copy()  # 创建副本以避免修改原始数据
                    score_item["question"] = q
                    score_item["standard_answer"] = score_item.get("standard_answer", standard_answers[i])
                    score_item["user_answer"] = score_item.get("user_answer", user_answers[i])
                    score_item["score"] = score_item.get("score", 0 if user_answers[i] == "未回答" else 5)
                    # 保留原始评语，如果没有则使用默认评语
                    if "comment" not in score_item or not score_item["comment"]:
                        score_item["comment"] = "未回答此问题" if user_answers[i] == "未回答" else ""
                
                formatted_scores.append(score_item)
            
            return summary, formatted_scores
        
        # 如果所有解析方法都失败，创建默认结果
        print("所有JSON解析方法都失败，使用默认评分")
        summary = "系统无法解析模型响应，但根据面试过程，候选人的回答显示了一定的技术能力和经验。"
        scores = []
        for i, (q, a, sa) in enumerate(zip(questions, user_answers, standard_answers)):
            scores.append({
                "question": q,
                "score": 5,  # 默认中等分数
                "standard_answer": sa,
                "user_answer": a,
                "comment": "系统无法自动评分，建议人工复核。"
            })
        
        return summary, scores
    except Exception as e:
        print(f"生成总结和评分时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 出错时返回默认值
        summary = "系统在处理面试结果时遇到错误，无法生成详细评估。"
        scores = []
        for i, (q, a, sa) in enumerate(zip(questions, user_answers, standard_answers)):
            scores.append({
                "question": q[:100] + "..." if len(q) > 100 else q,  # 截断过长的问题
                "score": 0,
                "standard_answer": sa[:100] + "..." if len(sa) > 100 else sa,
                "user_answer": a[:100] + "..." if len(a) > 100 else a,
                "comment": "系统错误，无法评分。"
            })
        
        return summary, scores
