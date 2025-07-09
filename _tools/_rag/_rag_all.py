from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import hashlib
import shutil
import time
from functools import lru_cache

save_file_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"files"))
embedding = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))

# 全局变量初始化
_combined_store = None
_last_modified_time = 0

# 将上传的文件保存到本地
def save_file(file, description=None):
    file_location = os.path.join(save_file_path, file.filename)
    with open(file_location, "wb") as f:
        f.write(file.file.read())  
    
    # 如果提供了描述，可以将其保存到某处
    if description:
        # 这里可以实现保存描述的逻辑
        pass
        
    return file_location

# 读取文件各种类型的文件
def read_file(file_path):
    encodings = ['utf-8', 'gbk', 'latin1', 'iso-8859-1']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
                return content
        except UnicodeDecodeError:
            print(f"文件: {file_path} 无法使用 {enc} 编码读取")
    raise ValueError("文件无法读取")

def text_splitter(text, chunk_size=100, chunk_overlap=10):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "。", "!", "！", "?", "？"]
    )
    texts = text_splitter.split_text(text)
    return texts

path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"vector_store"))
# 使用hash值保存向量文件
def save_vector_store(text):
    # 将内容进行wordEmbedding向量化
    _vector = FAISS.from_texts(
        text, 
        embedding=embedding,
        # 使用更高效的索引方式
        index_kwargs={"nlist": 5, "nprobe": 2}  # IVF索引参数
    )
                               
    text = "".join(text)
    hash_value = hashlib.md5(text.encode()).hexdigest()
    print(hash_value)
    _vector.save_local(f"{path}/{hash_value}")

# 使用HNSW索引保存向量文件 - 比标准FAISS更快的检索速度
def save_vector_store_hnsw(text):
    # 将内容进行wordEmbedding向量化
    _vector = FAISS.from_texts(
        text, 
        embedding=embedding,
        # 使用HNSW索引以加快检索
        index_kwargs={"M": 16, "efConstruction": 200, "efSearch": 128, "hnsw": True}
    )
                               
    text = "".join(text)
    hash_value = hashlib.md5(text.encode()).hexdigest()
    print(f"使用HNSW索引保存: {hash_value}")
    _vector.save_local(f"{path}/{hash_value}_hnsw")

# 使用LRU缓存加速加载过程
@lru_cache(maxsize=1)  # 最多缓存1个结果
def load_vector_store():
    # 缓存失效检查 - 如果向量存储目录有变化，则清除缓存
    global _last_modified_time, _combined_store
    current_modified_time = 0
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for name in files:
                file_path = os.path.join(root, name)
                current_modified_time = max(current_modified_time, os.path.getmtime(file_path))
    
    # 检查是否需要重新加载
    try:
        if '_last_modified_time' in globals() and _last_modified_time == current_modified_time:
            print("使用缓存的向量存储")
            return _combined_store
    except:
        pass
    
    # 更新最后修改时间
    _last_modified_time = current_modified_time
    
    vector_list = []
    # 确保向量存储目录存在
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建向量存储目录: {path}")
        return None
        
    # 获取向量文件列表
    try:
        start_time = time.time()
        files = os.listdir(path)
        if not files:
            print("向量存储目录为空")
            return None
            
        for file in files:
            text_load = FAISS.load_local(f"{path}/{file}",
                                    embeddings=embedding,
                                    allow_dangerous_deserialization=True)
            vector_list.append(text_load)
            
        # 合并向量
        if vector_list:
            combined_store = vector_list[0]
            for store in vector_list[1:]:
                combined_store.merge_from(store)
            
            # 存储到全局变量中供缓存使用
            _combined_store = combined_store
            
            end_time = time.time()
            print(f"加载向量存储完成，耗时: {end_time - start_time:.2f}秒")
            return combined_store
        else:
            return None
    except Exception as e:
        print(f"加载向量存储时出错: {e}")
        return None

# 删除文件和向量
def delete_file_and_vector(file_name):
    # 读取文件内容（自动处理编码）
    file_path = os.path.join(save_file_path, file_name)
    text = read_file(file_path)

    # 分割文本，与保存时保持一致
    texts = text_splitter(text)

    # 拼接后生成 hash
    text_joined = "".join(texts)
    hash_value = hashlib.md5(text_joined.encode()).hexdigest()
    vector_dir = f"{path}/{hash_value}"

    # 删除向量文件夹
    if os.path.exists(vector_dir):
        shutil.rmtree(vector_dir)
        print(f"已删除向量文件夹：{vector_dir}")
    else:
        print(f"未找到向量数据：{vector_dir}")

    # 也删除可能存在的HNSW索引版本
    hnsw_vector_dir = f"{path}/{hash_value}_hnsw"
    if os.path.exists(hnsw_vector_dir):
        shutil.rmtree(hnsw_vector_dir)
        print(f"已删除HNSW向量文件夹：{hnsw_vector_dir}")

    # 删除原始文件
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"已删除文件：{file_path}")
    
    # 清除缓存
    load_vector_store.cache_clear()

@tool
def search_vector_store(input_text: str) -> str:
    """
    搜索知识库中的文本，并判断是否相关，若不相关则返回提示。
    """
    try:
        start_time = time.time()
        combined_store = load_vector_store()
        if combined_store is None:
            return "知识库为空，请先上传文件。"
            
        res = combined_store.similarity_search_with_score(input_text, k=5)
        
        top_doc, score = res[0]
        
        end_time = time.time()
        print(f"搜索耗时: {end_time - start_time:.4f}秒")

        if score < 0.7:
            return top_doc.page_content

        return "知识库中未找到相关信息，建议尝试联网搜索。"
    except Exception as e:
        print(f"搜索知识库时出错: {e}")
        return "搜索知识库时出错，请尝试联网搜索或稍后再试。"

# 批量查询接口，可以一次查询多个问题，减少加载开销
def batch_search_vector_store(input_texts: list) -> list:
    """
    批量搜索知识库中的文本
    """
    try:
        start_time = time.time()
        combined_store = load_vector_store()
        if combined_store is None:
            return ["知识库为空，请先上传文件。"] * len(input_texts)
        
        results = []
        for input_text in input_texts:
            res = combined_store.similarity_search_with_score(input_text, k=5)
            top_doc, score = res[0]
            if score < 0.7:
                results.append(top_doc.page_content)
            else:
                results.append("知识库中未找到相关信息，建议尝试联网搜索。")
                
        end_time = time.time()
        print(f"批量搜索{len(input_texts)}个问题，总耗时: {end_time - start_time:.4f}秒")
        
        return results
    except Exception as e:
        print(f"批量搜索知识库时出错: {e}")
        return ["搜索知识库时出错，请尝试联网搜索或稍后再试。"] * len(input_texts)

# 测试各部分方法
if __name__ == "__main__":
    # text = read_file("D:/chatbot/_tools/_rag/2.txt")
    # texts = text_splitter(text)
    # save_vector_store(texts)

    res = search_vector_store("地球上有些水母能在脱水状态下维持DNA完整达30年以上")
    print(res)
    # delete_file_and_vector("files/x.txt")


