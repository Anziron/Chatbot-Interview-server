import os
import dotenv
dotenv.load_dotenv()
import uvicorn
import webbrowser
import threading
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from api.work_api import router as work_router
from api.rag_api import router as rag_router
from api.agent_api import router as agent_router, UPLOAD_DIR
from api.interview_api import router as interview_router

app = FastAPI()

# 创建缓存目录
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"_cache","cache_database"))
os.makedirs(CACHE_DIR, exist_ok=True)

# 创建导出目录
EXPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"exports"))
os.makedirs(EXPORTS_DIR, exist_ok=True)

STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"))
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# 配置模板
templates = Jinja2Templates(directory="templates")

# 添加API路由
app.include_router(work_router)
app.include_router(rag_router)
app.include_router(agent_router)
app.include_router(interview_router)

# 添加静态文件服务，用于访问上传的图片
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# 添加静态文件服务，用于下载导出的文件
app.mount("/exports", StaticFiles(directory=EXPORTS_DIR), name="exports")

# 首页路由
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 知识库页面路由
@app.get("/rag", response_class=HTMLResponse)
async def rag_page(request: Request):
    return templates.TemplateResponse("knowledge.html", {"request": request})

# 面试页面路由
@app.get("/interview", response_class=HTMLResponse)
async def interview_page(request: Request):
    return templates.TemplateResponse("interview.html", {"request": request})

@app.get("/favicon.ico")
async def favicon():
    return FileResponse(os.path.join(STATIC_DIR, "favicon.ico"))
    
def open_browser():
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    # 确保图片上传目录存在
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    # 确保导出目录存在
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    threading.Timer(2.0, open_browser).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)



