import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import shutil

# 导入项目模块
from model._llms import model
from _agents.resume_agent._agent import analyze_resume
from _agents.summary_agent._agent import generate_summary_and_score
from _tools._pdf.generate import generate_pdf_report
from langchain.schema import HumanMessage

# 创建路由器
router = APIRouter(prefix="/interview", tags=["interview"])

# 创建必要的目录
os.makedirs("uploads", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# 全局变量存储面试会话数据
interview_sessions = {}

# 模型：面试会话
class InterviewSession(BaseModel):
    resume_path: str
    resume_text: str
    questions: List[str]
    standard_answers: List[str]
    user_answers: List[str] = []
    current_question_index: int = 0
    is_completed: bool = False
    summary: Optional[str] = None
    scores: Optional[List[Dict[str, Any]]] = None
    report_path: Optional[str] = None
    total_score: Optional[int] = None
    total_possible_score: Optional[int] = None
    score_percentage: Optional[float] = None
    hire_recommendation: Optional[str] = None

# 模型：问题请求
class QuestionRequest(BaseModel):
    session_id: str

# 模型：回答请求
class AnswerRequest(BaseModel):
    session_id: str
    answer: str

# 模型：会话状态响应
class SessionResponse(BaseModel):
    session_id: str
    current_question_index: int
    total_questions: int
    current_question: Optional[str] = None
    is_completed: bool
    report_path: Optional[str] = None

# 模型：AI回答请求
class AiAnswerRequest(BaseModel):
    session_id: str
    question_index: int

# 判断是否处理简历的函数
def should_process_resume(resume_text: str) -> bool:
    """
    根据简历内容判断是否应该进入面试流程
    返回: True表示应该处理, False表示拒绝处理
    """
    # 使用LLM判断简历是否符合要求
    prompt = f"""
    请判断以下简历是否适合进入面试流程。
    
    要求：
    1. 简历应包含基本的个人信息和联系方式
    2. 简历应有明确的教育背景和工作经验描述
    3. 简历不应该是空白或无意义的内容
    
    请只回答"是"或"否"，不需要解释。
    
    简历内容：
    {resume_text}
    """
    
    response = model([HumanMessage(content=prompt)]).content.strip().lower()
    
    # 解析回答，查找表示同意的关键词
    return "是" in response or "yes" in response or "true" in response

@router.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)):
    """上传简历文件并开始面试流程"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只接受PDF格式的文件")
    
    # 生成唯一的会话ID
    session_id = str(uuid.uuid4())
    
    # 保存上传的文件
    resume_path = f"uploads/{session_id}_{file.filename}"
    with open(resume_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 分析简历
    try:
        resume_text, questions, standard_answers = analyze_resume(resume_path)
        
        # 判断是否处理简历
        if not should_process_resume(resume_text):
            return JSONResponse(
                status_code=400,
                content={"message": "简历内容不符合要求，无法进入面试流程"}
            )
        
        # 存储会话数据
        interview_sessions[session_id] = InterviewSession(
            resume_path=resume_path,
            resume_text=resume_text,
            questions=questions,
            standard_answers=standard_answers
        )
        
        return JSONResponse(
            content={
                "session_id": session_id,
                "message": "简历上传成功，已准备好面试问题",
                "total_questions": len(questions)
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"处理简历时出错: {str(e)}"}
        )

@router.get("/question/")
async def get_question(session_id: str):
    """获取当前面试问题"""
    if session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="面试会话不存在")
    
    session = interview_sessions[session_id]
    
    # 检查面试是否已完成
    if session.is_completed:
        return JSONResponse(
            content={
                "session_id": session_id,
                "is_completed": True,
                "message": "面试已结束",
                "report_path": session.report_path
            }
        )
    
    # 获取当前问题
    if session.current_question_index < len(session.questions):
        current_question = session.questions[session.current_question_index]
        return JSONResponse(
            content={
                "session_id": session_id,
                "current_question_index": session.current_question_index,
                "total_questions": len(session.questions),
                "current_question": current_question,
                "is_completed": False
            }
        )
    else:
        # 所有问题都已回答，生成报告
        return await complete_interview(session_id)

@router.post("/answer/")
async def submit_answer(request: AnswerRequest):
    """提交问题的回答"""
    if request.session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="面试会话不存在")
    
    session = interview_sessions[request.session_id]
    
    # 检查面试是否已结束
    if session.is_completed:
        return JSONResponse(
            content={
                "session_id": request.session_id,
                "is_completed": True,
                "message": "面试已结束",
                "report_path": session.report_path
            }
        )
    
    # 记录回答
    session.user_answers.append(request.answer)
    session.current_question_index += 1
    
    # 检查是否还有下一个问题
    if session.current_question_index < len(session.questions):
        next_question = session.questions[session.current_question_index]
        return JSONResponse(
            content={
                "session_id": request.session_id,
                "current_question_index": session.current_question_index,
                "total_questions": len(session.questions),
                "current_question": next_question,
                "is_completed": False
            }
        )
    else:
        # 所有问题都已回答，生成报告
        return await complete_interview(request.session_id)

@router.post("/ai-answer/")
async def get_ai_answer(request: AiAnswerRequest):
    """获取AI生成的回答建议"""
    if request.session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="面试会话不存在")
    
    session = interview_sessions[request.session_id]
    
    # 验证问题索引
    if request.question_index < 0 or request.question_index >= len(session.questions):
        raise HTTPException(status_code=400, detail="问题索引无效")
    
    # 获取当前问题和标准答案
    current_question = session.questions[request.question_index]
    standard_answer = session.standard_answers[request.question_index]
    
    # 使用LLM生成类似但不完全相同的回答
    prompt = f"""
    请根据以下问题和标准答案，生成一个合理的回答。
    不要完全照抄标准答案，而是用自己的语言表达类似的内容。
    
    问题: {current_question}
    标准答案: {standard_answer}
    """
    
    ai_answer = model([HumanMessage(content=prompt)]).content
    
    return JSONResponse(
        content={
            "ai_answer": ai_answer
        }
    )

@router.post("/skip-remaining/")
async def skip_remaining_questions(session_id: str):
    """跳过剩余问题并完成面试"""
    if session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="面试会话不存在")
    
    session = interview_sessions[session_id]
    
    # 如果面试已完成，直接返回
    if session.is_completed:
        return JSONResponse(
            content={
                "session_id": session_id,
                "is_completed": True,
                "message": "面试已结束",
                "report_path": session.report_path
            }
        )
    
    # 为未回答的问题填充"未回答"
    remaining = len(session.questions) - len(session.user_answers)
    for _ in range(remaining):
        session.user_answers.append("未回答")
    
    # 更新会话状态
    session.current_question_index = len(session.questions)
    
    # 生成报告
    return await complete_interview(session_id)

async def complete_interview(session_id: str):
    """生成面试总结和报告"""
    session = interview_sessions[session_id]
    
    try:
        # 生成总结和评分
        summary, scores = generate_summary_and_score(
            session.questions, 
            session.user_answers, 
            session.resume_text, 
            session.standard_answers
        )
        
        # 计算总分和录用建议
        total_possible_score = len(scores) * 10
        total_actual_score = sum(item.get("score", 0) for item in scores)
        score_percentage = (total_actual_score / total_possible_score) * 100 if total_possible_score > 0 else 0
        hire_recommendation = "建议录用" if score_percentage >= 60 else "建议不录用"
        
        # 生成PDF报告
        report_path = f"reports/{session_id}_interview_report.pdf"
        generate_pdf_report(summary, scores, report_path)
        
        # 更新会话状态
        session.is_completed = True
        session.summary = summary
        session.scores = scores
        session.report_path = report_path
        session.total_score = total_actual_score
        session.total_possible_score = total_possible_score
        session.score_percentage = score_percentage
        session.hire_recommendation = hire_recommendation
        
        return JSONResponse(
            content={
                "session_id": session_id,
                "is_completed": True,
                "message": "面试已完成，报告已生成",
                "report_path": f"/reports/{os.path.basename(report_path)}",
                "summary": summary,
                "total_score": total_actual_score,
                "total_possible_score": total_possible_score,
                "score_percentage": round(score_percentage, 1),
                "hire_recommendation": hire_recommendation
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"生成面试总结时出错: {str(e)}"}
        )

@router.get("/report/{session_id}")
async def get_report(session_id: str):
    """获取面试报告PDF"""
    if session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="面试会话不存在")
    
    session = interview_sessions[session_id]
    
    if not session.is_completed or not session.report_path:
        raise HTTPException(status_code=400, detail="面试尚未完成或报告未生成")
    
    if not os.path.exists(session.report_path):
        raise HTTPException(status_code=404, detail="报告文件不存在")
    
    return FileResponse(
        path=session.report_path, 
        filename=f"{session_id}_interview_report.pdf",
        media_type="application/pdf"
    )
