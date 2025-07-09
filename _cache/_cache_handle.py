import os
import hashlib
import shutil
import time
from functools import lru_cache
from scipy.spatial.distance import cosine
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings

cache_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"cache_database"))
embedding = DashScopeEmbeddings(
    model="text-embedding-v2", 
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 全局变量用于存储合并后的向量存储
_cached_combined_store = None
_last_modified_time = 0

def cache_content(question: str, answer: str, price: float, tokens: int, illation: str) -> str:
    # 将问题与答案保存到csv文件
    cache_csv(question, answer, price, tokens, illation)
    # 计算问题hash值
    hash_value = hashlib.md5(question.encode()).hexdigest()
    # 创建FAISS向量存储，使用HNSW索引提高查询效率
    vector_store = FAISS.from_texts(
        texts=[question],
        embedding=embedding,
        metadatas=[{"answer": answer, "illation": illation}],
        index_kwargs={"M": 16, "efConstruction": 200, "efSearch": 128, "hnsw": True}
    )
    # 保存FAISS向量存储
    vector_store.save_local(f"{cache_path}/{hash_value}")
    
    # 清除缓存，确保下次查询时重新加载
    get_combined_store.cache_clear()
    
    return hash_value

# 使用LRU缓存加载合并后的向量存储
@lru_cache(maxsize=1)
def get_combined_store():
    global _cached_combined_store, _last_modified_time
    
    # 检查缓存目录中的文件是否有更新
    current_modified_time = 0
    if os.path.exists(cache_path):
        for root, dirs, files in os.walk(cache_path):
            for name in files:
                file_path = os.path.join(root, name)
                if os.path.isfile(file_path):
                    current_modified_time = max(current_modified_time, os.path.getmtime(file_path))
    
    # 如果没有更新且已经有缓存，则直接返回缓存
    if _cached_combined_store is not None and _last_modified_time == current_modified_time:
        print("使用内存中的缓存向量存储")
        return _cached_combined_store
    
    # 记录当前修改时间
    _last_modified_time = current_modified_time
    
    # 如果目录不存在或为空，则返回None
    if not os.path.exists(cache_path) or not os.listdir(cache_path):
        return None
    
    # 加载所有FAISS向量存储
    start_time = time.time()
    vector_stores = []
    
    try:
        for file in os.listdir(cache_path):
            file_path = os.path.join(cache_path, file)
            if os.path.isdir(file_path):
                vector_stores.append(
                    FAISS.load_local(
                        file_path, 
                        embedding, 
                        allow_dangerous_deserialization=True
                    )
                )
    
        # 如果没有任何FAISS向量存储，返回None
        if not vector_stores:
            return None
        
        # 合并所有FAISS向量存储
        combined_store = vector_stores[0]
        for store in vector_stores[1:]:
            combined_store.merge_from(store)
        
        # 更新全局缓存
        _cached_combined_store = combined_store
        
        end_time = time.time()
        print(f"重新加载向量存储，耗时: {end_time - start_time:.4f}秒")
        
        return combined_store
    
    except Exception as e:
        print(f"加载向量存储时出错: {e}")
        return None

def get_content_from_cache(question: str, similarity_threshold: float = 0.85):
    # 获取合并后的FAISS向量存储
    start_time = time.time()
    combined_store = get_combined_store()
    
    # 如果没有任何FAISS向量存储，返回None
    if not combined_store:
        return None, None
    
    # 获取最相似的文档和分数
    try:
        results = combined_store.similarity_search_with_score(question, k=1)
        if not results:
            return None, None
            
        doc, score = results[0]
        similarity = 1 - score  # FAISS返回的是距离，转换为相似度
        
        end_time = time.time()
        print(f"缓存查询耗时: {end_time - start_time:.4f}秒，相似度: {similarity:.3f}")
        
        if similarity >= similarity_threshold:
            metadata = doc.metadata
            answer = metadata.get("answer")
            illation = metadata.get("illation")
            return answer, illation
            
        return None, None
        
    except Exception as e:
        print(f"查询缓存时出错: {e}")
        return None, None

def cache_csv(question: str, answer: str, price: float, tokens: int, illation: str):
    # 文件路径
    file_path = f"_cache/cache_text.csv"
    file_exists = os.path.exists(file_path)
    # 是否写入表头
    write_header = True
    # 如果文件存在，则不写入表头
    if file_exists and os.path.getsize(file_path) > 0:
        write_header = False
    # 打开文件
    with open(file_path, "a", encoding="utf-8") as f:
        # 如果需要写入表头，则写入表头
        if write_header:
            f.write("question,answer,price,tokens,illation\n")
        # 写入数据
        f.write(f"{question},{answer},{price},{tokens},{illation}\n")

def clear_cache():
    # 删除所有FAISS向量存储
    if os.path.exists(cache_path):
        for file in os.listdir(cache_path):
            file_path = os.path.join(cache_path, file)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
    
    # 清除CSV文件
    csv_path = "_cache/cache_text.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)
        
    # 清除内存缓存
    global _cached_combined_store
    _cached_combined_store = None
    get_combined_store.cache_clear()
    
    print("缓存已清空")

if __name__ == "__main__":
    # cache_content("你好", "你好")
    # result = get_content_from_cache("你好")
    # print(f"匹配结果: {result}")
    # clear_cache()
    pass
