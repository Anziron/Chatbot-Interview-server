import os
import base64
from openai import OpenAI
from langchain_core.tools import tool

# 创建 OpenAI 客户端（阿里云 DashScope 兼容接口）
model = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 图像转 base64
def base64_encode(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 工具函数：识别会议纪要
@tool
def identify_meeting_minutes(image_url: str):
    """
    识别会议纪要图片中的关键信息，返回JSON格式。

    参数:
    image_url (str): 会议纪要图片的本地路径

    返回:
    str: 包含关键信息的JSON字符串
    """
    completion = model.chat.completions.create(
        model="qwen-vl-ocr-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{base64_encode(image_url)}"
                        },
                        "min_pixels": 28 * 28 * 4,
                        "max_pixels": 28 * 28 * 8192
                    },
                    {
                        "type": "text",
                        "text": (
                            "请从会议纪要图像中准确提取以下关键信息：会议时间、会议地点、主持人、参会人员、记录人、会议议题、会议内容。"
                            "不要遗漏信息，也不要捏造。若有模糊或遮挡的字可用英文问号?代替。"
                            "请以标准JSON格式输出，格式为："
                            "{'会议时间':'xxx', '会议地点':'xxx', '主持人':'xxx', '参会人员':'xxx', '记录人':'xxx', '会议议题':'xxx', '会议内容':'xxx'}"
                        )
                    }
                ]
            }
        ]
    )
    return completion.choices[0].message.content

# 测试调用
if __name__ == "__main__":
    result = identify_meeting_minutes("E:/chatbot/_tools/_identify/1.png")
    print(result)
