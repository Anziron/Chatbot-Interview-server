import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import UploadFile, File, Form
from _tools._rag._rag_all import read_file, text_splitter, save_vector_store, save_vector_store_hnsw, search_vector_store, batch_search_vector_store, save_file, delete_file_and_vector
from pydantic import BaseModel
from fastapi import APIRouter
from typing import Optional, List

router = APIRouter()
# 获取保存文件的路径
save_file_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_tools", "_rag", "files"))

@router.post("/upload")
def upload_file(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Form(100),
    chunk_overlap: Optional[int] = Form(10),
    use_hnsw: Optional[bool] = Form(True)  # 默认使用HNSW索引
):
    file_location = save_file(file)
    if file_location:
        text = read_file(file_location)
        texts = text_splitter(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # 根据参数选择使用普通索引还是HNSW索引
        if use_hnsw:
            save_vector_store_hnsw(texts)
        else:
            save_vector_store(texts)
        return {"message": "文件上传成功"}
    return {"message": "文件上传失败"}

class Search(BaseModel):
    query: str

@router.post("/search")
def search(search: Search):
    res = search_vector_store(search.query)
    return {"result": res}

class BatchSearch(BaseModel):
    queries: List[str]

@router.post("/batch_search")
def batch_search(search: BatchSearch):
    results = batch_search_vector_store(search.queries)
    return {"results": results}

class DeleteRequest(BaseModel):
    file_name: str

@router.post("/delete")
def delete_file(req: DeleteRequest):
    delete_file_and_vector(req.file_name)
    return {"message": "文件删除成功"}

@router.get("/files")
def get_files():
    files = []
    if os.path.exists(save_file_path):
        for file in os.listdir(save_file_path):
            files.append({"name": file})
    return files

# def open_browser():
#     webbrowser.open("http://127.0.0.2:8000/docs")

# if __name__ == "__main__":
#     threading.Timer(2.0, open_browser).start()
#     uvicorn.run(app, host="0.0.0.0", port=8000)
