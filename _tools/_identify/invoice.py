import os
from openai import OpenAI
from langchain_core.tools import tool
import base64

model = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
def base64_encode(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

@tool
def identify_invoice(image_url: str):
    """
    识别发票图片中的关键信息，提取发票代码、发票号码、开票日期、购买方、销售方、金额、税额等信息，返回JSON格式
    
    参数:
    image_url (str): 发票图片的本地路径
    
    返回:
    str: 包含发票关键信息的JSON字符串
    """
    completion = model.chat.completions.create(
    model="qwen-vl-ocr-latest",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpg;base64,{base64_encode(image_url)}"},
                    # 输入图像的最小像素阈值，小于该值图像会按原比例放大，直到总像素大于min_pixels
                    "min_pixels": 28 * 28 * 4,
                    # 输入图像的最大像素阈值，超过该值图像会按原比例缩小，直到总像素低于max_pixels
                    "max_pixels": 28 * 28 * 8192
                },
                {"type": "text",
                 "text": "请提取图像中的发票关键信息。根据发票类型（增值税专用发票、普通发票、电子发票等），提取以下信息：发票代码、发票号码、开票日期、购买方名称、购买方纳税人识别号、销售方名称、销售方纳税人识别号、商品或服务名称、金额、税率、税额、价税合计等。请准确无误地提取上述关键信息，不要遗漏和捏造虚假信息。模糊或难以辨认的内容可以用英文问号?代替。返回数据格式以JSON方式输出，格式为：{'发票类型':'xxx', '发票代码':'xxx', '发票号码':'xxx', '开票日期':'xxx', '购买方名称':'xxx', '购买方纳税人识别号':'xxx', '销售方名称':'xxx', '销售方纳税人识别号':'xxx', '商品或服务名称':'xxx', '金额':'xxx', '税率':'xxx', '税额':'xxx', '价税合计':'xxx'}"},
            ]
        }
    ])

    return completion.choices[0].message.content

if __name__ == "__main__":
    print(identify_invoice("C:/Users/29787/Desktop/pic.jpg"))