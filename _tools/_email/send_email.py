import os
import smtplib
import time
import re
import html
import imghdr
import urllib.request
import urllib.parse
import tempfile
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.header import Header
from email.utils import formataddr
import traceback
import logging
import base64
from typing import Union, Optional, Dict, Any, List
from langchain_core.tools import tool
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('email_tools')

# 获取环境变量或使用默认值
SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.qq.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', 465))
SMTP_USERNAME = os.environ.get('EMAIL_SENDER', '1312235991@qq.com')
SMTP_PASSWORD = os.environ.get('EMAIL_AUTH_CODE', 'anrdvyqqnuevhgad')
SMTP_SENDER_NAME = os.environ.get('SMTP_SENDER_NAME', 'AI助手')
SMTP_TIMEOUT = int(os.environ.get('SMTP_TIMEOUT', 30))  # 默认30秒超时

def validate_email(email: str) -> bool:
    """验证邮箱格式是否正确"""
    if not email:
        return False
    
    # 简单的邮箱格式验证
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    return bool(email_pattern.match(email))

def download_image_from_url(url: str) -> Optional[bytes]:
    """从URL下载图片，返回图片数据"""
    try:
        # 设置超时
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.read()
    except Exception as e:
        logger.error(f"无法下载图片: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def detect_image_format(image_data: bytes) -> Optional[str]:
    """检测图片格式"""
    if not image_data:
        return None
    
    # 使用imghdr检测图片格式
    image_format = imghdr.what(None, image_data)
    return image_format

def create_secure_smtp_connection():
    """创建安全的SMTP连接"""
    try:
        # 使用SSL连接
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=SMTP_TIMEOUT)
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        return server
    except Exception as e:
        logger.error(f"SMTP连接失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@tool
def send_email(to: str, subject: str, body: str, attachments: List[Dict[str, str]] = None) -> str:
    """
    发送电子邮件
    
    参数:
    - to: 收件人邮箱
    - subject: 邮件主题
    - body: 邮件正文
    - attachments: 附件列表，每个附件为一个字典，包含文件名和文件路径
    
    返回:
    - 成功/错误信息
    """
    if not validate_email(to):
        return f"无效的收件人邮箱地址: {to}"
    
    try:
        # 创建MIMEMultipart对象
        msg = MIMEMultipart()
        msg['From'] = formataddr((Header(SMTP_SENDER_NAME, 'utf-8').encode(), SMTP_USERNAME))
        msg['To'] = to
        msg['Subject'] = Header(subject, 'utf-8').encode()
        
        # 添加正文
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # 添加附件
        if attachments:
            for attachment in attachments:
                try:
                    with open(attachment['path'], 'rb') as f:
                        part = MIMEText(f.read(), 'base64', 'utf-8')
                        part['Content-Type'] = 'application/octet-stream'
                        part['Content-Disposition'] = f'attachment; filename="{attachment["name"]}"'
                        msg.attach(part)
                except Exception as e:
                    logger.error(f"添加附件失败: {str(e)}")
                    # 继续处理其他附件
        
        # 发送邮件
        start_time = time.time()
        server = create_secure_smtp_connection()
        server.sendmail(SMTP_USERNAME, to, msg.as_string())
        server.quit()
        
        end_time = time.time()
        logger.info(f"邮件发送成功，耗时: {end_time - start_time:.2f}秒")
        return f"邮件已成功发送到 {to}"
        
    except Exception as e:
        logger.error(f"发送邮件失败: {str(e)}")
        logger.error(traceback.format_exc())
        return f"发送邮件失败: {str(e)}"

def send_html_email(to: str, subject: str, html_content: str, image_url: str = None) -> str:
    """
    发送HTML格式的电子邮件，可选择嵌入图片
    
    参数:
    - to: 收件人邮箱
    - subject: 邮件主题
    - html_content: HTML格式的邮件内容
    - image_url: 可选，要嵌入的图片URL
    
    返回:
    - 成功/错误信息
    """
    if not validate_email(to):
        return f"无效的收件人邮箱地址: {to}"
    
    try:
        # 创建MIMEMultipart对象
        msg = MIMEMultipart('related')
        msg['From'] = formataddr((Header(SMTP_SENDER_NAME, 'utf-8').encode(), SMTP_USERNAME))
        msg['To'] = to
        msg['Subject'] = Header(subject, 'utf-8').encode()
        
        # 添加HTML内容
        html_part = MIMEText(html_content, 'html', 'utf-8')
        msg.attach(html_part)
        
        # 如果提供了图片URL，尝试嵌入图片
        if image_url:
            try:
                # 安全地处理图片URL - 转义特殊字符
                safe_url = image_url.replace('&', '&amp;')
                
                # 尝试下载图片
                logger.info(f"尝试从URL下载图片: {safe_url[:50]}...")
                image_data = download_image_from_url(image_url)
                
                if image_data:
                    # 检测图片格式
                    image_format = detect_image_format(image_data) or 'jpeg'
                    
                    # 创建图片附件
                    image_cid = f'image_{int(time.time())}'
                    img = MIMEImage(image_data, _subtype=image_format)
                    img.add_header('Content-ID', f'<{image_cid}>')
                    img.add_header('Content-Disposition', 'inline')
                    msg.attach(img)
                    
                    # 更新HTML内容中的图片引用
                    html_content_with_cid = html_content.replace(image_url, f'cid:{image_cid}')
                    msg.get_payload()[0].set_payload(html_content_with_cid)
                    
                    logger.info("图片成功嵌入到邮件中")
                else:
                    logger.warning("无法下载图片，将发送不带图片的邮件")
            except Exception as img_error:
                logger.error(f"处理图片时出错: {str(img_error)}")
                logger.error(traceback.format_exc())
                # 继续发送不带图片的邮件
        
        # 发送邮件
        start_time = time.time()
        server = create_secure_smtp_connection()
        server.sendmail(SMTP_USERNAME, to, msg.as_string())
        server.quit()
        
        end_time = time.time()
        logger.info(f"HTML邮件发送成功，耗时: {end_time - start_time:.2f}秒")
        return f"HTML邮件已成功发送到 {to}"
        
    except Exception as e:
        logger.error(f"发送HTML邮件失败: {str(e)}")
        logger.error(traceback.format_exc())
        return f"发送HTML邮件失败: {str(e)}"

def simple_send_email(to: str, subject: str, body: str) -> str:
    """
    简化版的邮件发送函数，专为图片URL问题设计
    不尝试解析或构建复杂的HTML，而是直接发送纯文本邮件
    
    参数:
    - to: 收件人邮箱
    - subject: 邮件主题
    - body: 邮件正文(纯文本)
    
    返回:
    - 成功/错误信息
    """
    if not validate_email(to):
        return f"无效的收件人邮箱地址: {to}"
    
    try:
        # 创建简单的MIMEText对象
        msg = MIMEText(body, 'plain', 'utf-8')
        msg['From'] = formataddr((Header(SMTP_SENDER_NAME, 'utf-8').encode(), SMTP_USERNAME))
        msg['To'] = to
        msg['Subject'] = Header(subject, 'utf-8').encode()
        
        # 发送邮件
        start_time = time.time()
        server = create_secure_smtp_connection()
        server.sendmail(SMTP_USERNAME, to, msg.as_string())
        server.quit()
        
        end_time = time.time()
        logger.info(f"简单邮件发送成功，耗时: {end_time - start_time:.2f}秒")
        return f"邮件已成功发送到 {to}"
        
    except Exception as e:
        logger.error(f"发送简单邮件失败: {str(e)}")
        logger.error(traceback.format_exc())
        return f"发送邮件失败: {str(e)}"

def send_plain_email(to: str, subject: str, body: str) -> str:
    """
    最简单的纯文本邮件发送函数，专为解决JSON解析和URL特殊字符问题设计
    
    参数:
    - to: 收件人邮箱
    - subject: 邮件主题
    - body: 邮件正文(纯文本)
    
    返回:
    - 成功/错误信息
    """
    if not validate_email(to):
        return f"无效的收件人邮箱地址: {to}"
    
    try:
        # 清理可能导致问题的特殊字符
        clean_body = body
        # 移除URL中可能导致问题的特殊字符
        if "http" in clean_body:
            # 将URL中的&替换为文本形式
            clean_body = re.sub(r'&(?=[^;]{4})', '[和]', clean_body)
            # 将URL中的%替换为文本形式
            clean_body = re.sub(r'%(?=[0-9A-Fa-f]{2})', '[百分号]', clean_body)
        
        # 创建最简单的MIMEText对象
        msg = MIMEText(clean_body, 'plain', 'utf-8')
        msg['From'] = formataddr((Header(SMTP_SENDER_NAME, 'utf-8').encode(), SMTP_USERNAME))
        msg['To'] = to
        msg['Subject'] = Header(subject, 'utf-8').encode()
        
        # 发送邮件
        start_time = time.time()
        try:
            # 使用最简单的方式创建连接
            server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=SMTP_TIMEOUT)
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, to, msg.as_string())
            server.quit()
            
            end_time = time.time()
            logger.info(f"纯文本邮件发送成功，耗时: {end_time - start_time:.2f}秒")
            return f"邮件已成功发送到 {to}"
        except Exception as smtp_error:
            logger.error(f"SMTP发送失败，尝试备用方法: {str(smtp_error)}")
            
            # 备用方法：使用最基础的SMTP连接
            backup_server = smtplib.SMTP(SMTP_SERVER, 25, timeout=SMTP_TIMEOUT)
            backup_server.starttls()  # 使用TLS
            backup_server.login(SMTP_USERNAME, SMTP_PASSWORD)
            backup_server.sendmail(SMTP_USERNAME, to, msg.as_string())
            backup_server.quit()
            
            end_time = time.time()
            logger.info(f"使用备用方法发送纯文本邮件成功，耗时: {end_time - start_time:.2f}秒")
            return f"邮件已成功发送到 {to} (使用备用方法)"
            
    except Exception as e:
        logger.error(f"发送纯文本邮件失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 最终备用方法：尝试使用最简单的内容
        try:
            # 创建极简内容
            minimal_body = "您请求的图片已生成。由于技术原因，无法在邮件中直接显示图片URL，请登录系统查看。"
            minimal_msg = MIMEText(minimal_body, 'plain', 'utf-8')
            minimal_msg['From'] = SMTP_USERNAME
            minimal_msg['To'] = to
            minimal_msg['Subject'] = "您请求的图片已生成"
            
            server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=SMTP_TIMEOUT)
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, to, minimal_msg.as_string())
            server.quit()
            
            logger.info("使用极简内容发送邮件成功")
            return f"邮件已成功发送到 {to} (使用极简内容)"
        except Exception as final_error:
            logger.error(f"所有邮件发送方法均失败: {str(final_error)}")
            return f"发送邮件失败: {str(e)} (所有方法均已尝试)"

# 测试
if __name__ == "__main__":
    result = send_email("example@example.com", "测试邮件", "这是一封测试邮件")
    print(result)
