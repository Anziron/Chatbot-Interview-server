from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
import os

# 导入中文字体支持
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import platform

# 注册中文字体
def register_chinese_font():
    try:
        # 尝试注册不同平台上常见的中文字体
        if platform.system() == "Windows":
            # Windows系统字体路径
            font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
            fallback_path = "C:/Windows/Fonts/simkai.ttf"  # 楷体
            
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont('SimHei', font_path))
                return 'SimHei'
            elif os.path.exists(fallback_path):
                pdfmetrics.registerFont(TTFont('SimKai', fallback_path))
                return 'SimKai'
                
        elif platform.system() == "Darwin":  # macOS
            font_path = "/System/Library/Fonts/PingFang.ttc"
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont('PingFang', font_path))
                return 'PingFang'
                
        elif platform.system() == "Linux":
            # 常见Linux中文字体路径
            font_paths = [
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/truetype/arphic/uming.ttc"
            ]
            
            for path in font_paths:
                if os.path.exists(path):
                    font_name = os.path.basename(path).split('.')[0]
                    pdfmetrics.registerFont(TTFont(font_name, path))
                    return font_name
                    
        # 默认返回None，表示没有找到合适的中文字体
        print("警告: 未找到支持中文的字体，PDF可能无法正确显示中文")
        return None
        
    except Exception as e:
        print(f"注册中文字体时出错: {str(e)}")
        return None

def generate_pdf_report(summary: str, scores: list, output_path: str = "interview_report.pdf") -> str:
    """
    生成面试总结报告PDF

    参数:
        summary (str): 总体总结文本
        scores (list): 每题评分信息，结构为 [{"question":..., "score":..., "standard_answer":..., "user_answer":..., "comment":...}, ...]
        output_path (str): 输出的 PDF 文件路径

    返回:
        str: 生成的 PDF 路径
    """
    # 注册中文字体
    chinese_font = register_chinese_font()
    
    # 设置文档
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    # 样式设置
    styles = getSampleStyleSheet()
    
    # 创建支持中文的样式
    body_style = ParagraphStyle(
        name='Body',
        fontSize=11,
        leading=16,
        alignment=TA_LEFT,
    )
    
    # 创建强调样式
    emphasis_style = ParagraphStyle(
        name='Emphasis',
        fontSize=12,
        leading=18,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold',
    )
    
    # 如果找到了中文字体，则使用它
    if chinese_font:
        body_style.fontName = chinese_font
        styles["Title"].fontName = chinese_font
        styles["Heading2"].fontName = chinese_font
        emphasis_style.fontName = chinese_font
    
    elements = []

    try:
        # 标题
        elements.append(Paragraph("<b>面试总结报告</b>", styles["Title"]))
        elements.append(Spacer(1, 0.5 * cm))

        # 总体总结部分
        elements.append(Paragraph("<b>总结：</b>", styles["Heading2"]))
        elements.append(Paragraph(summary, body_style))
        elements.append(Spacer(1, 0.5 * cm))
        
        # 计算总分和录用建议
        total_possible_score = len(scores) * 10
        total_actual_score = sum(item.get("score", 0) for item in scores)
        score_percentage = (total_actual_score / total_possible_score) * 100 if total_possible_score > 0 else 0
        hire_recommendation = "建议录用" if score_percentage >= 60 else "建议不录用"
        
        # 添加总分和录用建议部分
        elements.append(Paragraph("<b>总评：</b>", styles["Heading2"]))
        elements.append(Paragraph(f"总分：{total_actual_score} / {total_possible_score}（{score_percentage:.1f}%）", body_style))
        elements.append(Spacer(1, 0.3 * cm))
        elements.append(Paragraph(f"<b>录用建议：</b> {hire_recommendation}", emphasis_style))
        elements.append(Spacer(1, 0.5 * cm))

        # 逐题评分部分
        elements.append(Paragraph("<b>评分明细：</b>", styles["Heading2"]))
        for idx, item in enumerate(scores, 1):
            question = item.get("question", "")
            score = item.get("score", "")
            comment = item.get("comment", "")
            standard_answer = item.get("standard_answer", "")
            user_answer = item.get("user_answer", "")
            
            # 确保问题内容和标准答案是完整的
            if "**" in question and not question.strip().endswith("**"):
                # 问题可能包含类型标签，保留完整内容
                q_text = f"<b>问题{idx}:</b> {question}"
            else:
                q_text = f"<b>问题{idx}:</b> {question}"
                
            s_text = f"<b>评分：</b>{score} / 10"
            c_text = f"<b>评语：</b>{comment}"
            
            # 格式化标准答案和用户回答
            sa_text = f"<b>标准答案：</b> {standard_answer}"
            ua_text = f"<b>候选人回答：</b> {user_answer}"

            elements.append(Paragraph(q_text, body_style))
            elements.append(Spacer(1, 0.2 * cm))
            elements.append(Paragraph(s_text, body_style))
            elements.append(Spacer(1, 0.2 * cm))
            elements.append(Paragraph(sa_text, body_style))
            elements.append(Spacer(1, 0.2 * cm))
            elements.append(Paragraph(ua_text, body_style))
            elements.append(Spacer(1, 0.2 * cm))
            elements.append(Paragraph(c_text, body_style))
            elements.append(Spacer(1, 0.5 * cm))

        # 生成文档
        doc.build(elements)
        print(f"PDF报告已生成：{os.path.abspath(output_path)}")
        return os.path.abspath(output_path)
    except Exception as e:
        print(f"生成PDF报告时出错: {str(e)}")
        try:
            # 计算总分和录用建议
            total_possible_score = len(scores) * 10
            total_actual_score = sum(item.get("score", 0) for item in scores)
            score_percentage = (total_actual_score / total_possible_score) * 100 if total_possible_score > 0 else 0
            hire_recommendation = "建议录用" if score_percentage >= 60 else "建议不录用"
            
            # 尝试生成一个简单的PDF报告
            c = canvas.Canvas(output_path, pagesize=A4)
            c.setFont("Helvetica", 12)
            c.drawString(100, 750, "面试报告 (生成时出错)")
            c.drawString(100, 730, "错误: " + str(e))
            c.drawString(100, 710, "总结: " + (summary[:100] + "..." if len(summary) > 100 else summary))
            c.drawString(100, 690, f"总分: {total_actual_score}/{total_possible_score} ({score_percentage:.1f}%)")
            c.drawString(100, 670, f"录用建议: {hire_recommendation}")
            c.save()
            print(f"生成了简单的错误报告PDF: {os.path.abspath(output_path)}")
            return os.path.abspath(output_path)
        except:
            print("无法创建任何PDF文件")
            return "PDF生成失败"
