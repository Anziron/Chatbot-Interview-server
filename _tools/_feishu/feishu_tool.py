import os
import json
import requests
from langchain_core.tools import tool


@tool
def save_to_feishu_doc(content, folder_id=None) -> str:
    """
    将内容保存到飞书云文档

    Args:
        content: 要保存的内容，可以是JSON字符串或字典
        folder_id: 文件夹token，默认使用环境变量中的配置

    Returns:
        str: 保存结果信息
    """
    try:
        # 获取配置
        app_id = os.getenv("FEISHU_APP_ID")
        app_secret = os.getenv("FEISHU_APP_SECRET")
        folder_id = folder_id or os.getenv("FEISHU_folder_id")

        if not app_id or not app_secret:
            return "缺少 APP_ID 或 APP_SECRET 配置"

        # 获取 tenant_access_token
        token_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/"
        token_payload = {
            "app_id": app_id,
            "app_secret": app_secret
        }
        token_resp = requests.post(token_url, json=token_payload)
        token_data = token_resp.json()

        if token_resp.status_code != 200 or token_data.get("code") != 0:
            return f"获取 token 失败: {token_data}"

        tenant_access_token = token_data["tenant_access_token"]

        # 处理 content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                content = {"content": content}

        content_str = "\n".join(f"{k}: {v}" for k, v in content.items())

        # 创建文档
        create_doc_url = "https://open.feishu.cn/open-apis/docx/v1/documents"
        headers = {
            "Authorization": f"Bearer {tenant_access_token}",
            "Content-Type": "application/json"
        }
        doc_payload = {
            "folder_id": folder_id,
            "title": "自动生成文档"
        }
        doc_resp = requests.post(create_doc_url, headers=headers, json=doc_payload)
        doc_data = doc_resp.json()

        if doc_resp.status_code != 200 or doc_data.get("code") != 0:
            return f"创建文档失败: {doc_data}"

        document_id = doc_data["data"]["document"]["document_id"]
        doc_url = f"https://your-domain.feishu.cn/docx/{document_id}"  # 替换为您的实际域名

        # 使用新版块API添加内容
        create_block_url = f"https://open.feishu.cn/open-apis/docx/v1/documents/{document_id}/blocks/{document_id}/children"
        block_payload = {
                            "children": [{
                                "block_type": 2,
                                "text": {
                                    "elements": [{
                                        "text_run": {
                                            "content": content_str
                                        }
                                    }]
                                }
                            }]
                        }
        
        block_resp = requests.post(create_block_url, headers=headers, json=block_payload)
        block_data = block_resp.json()

        if block_resp.status_code != 200 or block_data.get("code") != 0:
            return f"添加内容失败: {block_data}"

        return f"文档已保存: {doc_url}"

    except Exception as e:
        return f"系统异常: {str(e)}"

if __name__ == "__main__":
    # 示例内容
    example_content = {
        "title": "示例文档",
        "description": "这是一个自动生成的文档示例。",
        "items": ["项目1", "项目2", "项目3"]
    }
    
    # 调用函数保存到飞书文档
    result = save_to_feishu_doc(example_content)
    print(result)