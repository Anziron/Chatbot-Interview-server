import pandas as pd
import os
import json
from langchain_core.tools import tool

# 基础URL，根据实际部署环境可能需要修改
BASE_URL = "https://anziron.cyou"

@tool
def simple_excel_export(data: str, filename: str = "data_export") -> str:
    """
    将文本数据保存为简单的Excel文件
    
    Args:
        data: 要保存的文本数据
        filename: 文件名(不需要包含扩展名)
    
    Returns:
        str: 保存结果信息
    """
    try:
        # 确保文件名有正确的扩展名
        if not filename.endswith(".xlsx"):
            filename += ".xlsx"
        
        # 确保导出目录存在 - 使用绝对路径
        save_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "exports"))
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            
        # 构建完整的文件路径
        full_path = os.path.join(save_path, filename)
        
        # 创建简单的DataFrame
        df = pd.DataFrame({"内容": [data]})
        
        # 导出到Excel
        df.to_excel(full_path, index=False)
        
        # 生成完整的HTTP下载链接
        download_link = f"{BASE_URL}/exports/{filename}"
        
        return f"数据已成功保存为Excel文件。下载链接: {download_link}"
    except Exception as e:
        return f"错误: 保存Excel文件时遇到问题 - {str(e)}"

@tool
def table_to_excel(table_data: str, filename: str = "table_export") -> str:
    """
    将表格数据保存为Excel文件
    
    Args:
        table_data: 表格数据，格式为JSON字符串，例如：
                   '[{"姓名":"张三","年龄":25},{"姓名":"李四","年龄":30}]'
                   或者单个对象：'{"发票号码":"123456","金额":100}'
        filename: 文件名(不需要包含扩展名)
    
    Returns:
        str: 保存结果信息
    """
    try:
        # 确保文件名有正确的扩展名
        if not filename.endswith(".xlsx"):
            filename += ".xlsx"
        
        # 确保导出目录存在 - 使用绝对路径
        save_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "exports"))
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            
        # 构建完整的文件路径
        full_path = os.path.join(save_path, filename)
        
        # 解析JSON数据
        try:
            json_data = json.loads(table_data)
            
            # 如果是字典，转换为列表
            if isinstance(json_data, dict):
                data_list = [json_data]
            elif isinstance(json_data, list):
                data_list = json_data
            else:
                return f"错误: 数据必须是JSON对象或对象列表"
                
            # 创建DataFrame
            df = pd.DataFrame(data_list)
            
            # 导出到Excel
            df.to_excel(full_path, index=False)
            
            # 生成完整的HTTP下载链接
            download_link = f"{BASE_URL}/exports/{filename}"
            
            return f"表格数据已成功保存为Excel文件。下载链接: {download_link}"
        except json.JSONDecodeError:
            return f"错误: 提供的数据不是有效的JSON格式"
    except Exception as e:
        return f"错误: 保存Excel文件时遇到问题 - {str(e)}"

if __name__ == "__main__":
    # 测试简单Excel导出
    test_data = "这是一段测试文本，将被保存到Excel文件中。"
    print(simple_excel_export(test_data, "simple_test"))
    
    # 测试表格数据导出
    test_table = '[{"姓名":"张三","年龄":25},{"姓名":"李四","年龄":30}]'
    print(table_to_excel(test_table, "table_test")) 