import os
from langchain.tools import tool
import pdfplumber


@tool
def read_pdf(file_path: str) -> str:
    """
    读取PDF文件内容
    
    Args:
        file_path: PDF文件路径
        
    Returns:
        提取的文本内容
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return f"错误：文件不存在 - {file_path}"
            
        # 检查文件是否为PDF
        if not file_path.lower().endswith('.pdf'):
            return f"错误：文件不是PDF格式 - {file_path}"
        
        # 使用pdfplumber读取PDF内容
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
                text += "\n\n"  # 添加页面分隔符
                
        # 如果没有提取到文本，可能是扫描件
        if not text.strip():
            return "错误：无法提取文本，可能是扫描PDF或图片PDF"
            
        return text.strip()
    
    except Exception as e:
        return f"读取PDF时出错: {str(e)}" 
    
    

