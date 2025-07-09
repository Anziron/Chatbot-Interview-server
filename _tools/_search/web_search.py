import os
import warnings
from langchain_core.tools import tool

# 尝试导入最新的 TavilySearch 工具
try:
    from langchain_tavily import TavilySearch
    tavily_imported = True
except ImportError:
    tavily_imported = False
    warnings.warn(
        "langchain-tavily 包未安装，将使用弃用的 TavilySearchResults。\n"
        "建议安装新版本: pip install langchain-tavily",
        DeprecationWarning
    )
    # 如果新包不可用，回退到旧的实现
    from langchain_community.tools import TavilySearchResults


@tool
def web_search(query: str):
    """
    搜索互联网上的信息，返回多个相关结果
    """
    # 根据可用的包选择适当的搜索工具
    if tavily_imported:
        search = TavilySearch(max_results=3)
    else:
        # 兼容模式：使用旧的 TavilySearchResults
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            search = TavilySearchResults(max_results=3)
    
    # 检查是否需要使用API密钥
    if "TAVILY_API_KEY" not in os.environ:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return [{"title": "搜索功能暂时不可用", "content": "未配置Tavily API密钥", "url": ""}]
    
    try:
        return search.invoke(query)
    except Exception as e:
        # 捕获搜索过程中的异常，返回友好的错误信息
        error_message = str(e)
        return [{"title": "搜索过程中出错", "content": f"错误信息: {error_message[:200]}", "url": ""}]

if __name__ == "__main__":
    res = web_search("广东今天天气怎么样")
    print(res)
