# main.py (수정된 전체 코드)

import shutil
import os
import tempfile
from typing import Dict, Optional
from contextlib import asynccontextmanager # ✅ lifespan을 위해 import 추가

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks
from user_answers import save_user_answers, get_user_answers
from llm import store_document_vectors
from rag_chatbot import get_chatbot_response
from auth import get_current_user, get_current_user_optional
from pydantic import BaseModel
from utils import get_predefined_questions
from fastapi.middleware.cors import CORSMiddleware
# ✅ chatbot_manager에서 초기화 함수를 추가로 import 합니다.
from chatbot_manager import router as chatbot_router, init_pinecone_client

# --- Pydantic 모델 정의 (기존과 동일) ---
class AnswersRequest(BaseModel):
    answers: Dict[str, str]

class ChatRequest(BaseModel):
    query: str
    userId: Optional[str] = None

# --- Lifespan 이벤트 핸들러 추가 ---
@asynccontextmanager # ✅ FastAPI의 lifespan 이벤트 핸들러 데코레이터
async def lifespan(app: FastAPI): # ✅ 애플리케이션의 시작과 종료 시점을 관리하는 함수 추가
    # 애플리케이션 시작 시 실행될 코드
    print("### 애플리케이션 시작 ###")
    init_pinecone_client() # ✅ 여기서 Pinecone 클라이언트를 초기화합니다.
    yield
    # 애플리케이션 종료 시 실행될 코드
    print("### 애플리케이션 종료 ###")

# --- FastAPI 앱 생성 시 lifespan 연결 ---
app = FastAPI(lifespan=lifespan) # ✅ FastAPI 앱에 lifespan 핸들러를 연결합니다.

# --- CORS 설정 (기존과 동일) ---
origins = [
    "https://staging.d1dbfs3o76ym6j.amplifyapp.com",
    "https://staging.d1dbbls3b75wb6.amplifyapp.com",
    "https://www.my-fortpoilo-fopofo.com"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 라우터 포함 (기존과 동일) ---
app.include_router(chatbot_router)

# --- 백그라운드 작업 및 API 엔드포인트들 (이하 변경 없음) ---
def store_and_cleanup_task(file_path: str, user_id: str):
    try:
        store_document_vectors(file_path, user_id)
    except Exception as e:
        print(f"!!!!!!!! [BACKGROUND ERROR] 파일 처리 중 오류 발생: {e} !!!!!!!!!!")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"--- [백그라운드] 임시 파일 '{file_path}' 삭제 완료 ---")

@app.get("/")
async def read_root():
    return {"message": "안녕하세요! 챗봇 API 서버입니다."}

@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    temp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            temp_file_path = tmp.name
            shutil.copyfileobj(file.file, tmp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임시 파일 생성 중 오류 발생: {e}")
    finally:
        await file.close()

    background_tasks.add_task(store_and_cleanup_task, temp_file_path, user_id)
    return {"message": "파일 업로드가 시작되었습니다. 처리가 완료되면 사용할 수 있습니다."}

@app.post("/save-answers")
async def save_answers_api(request: AnswersRequest, user_id: str = Depends(get_current_user)):
    received_data = request.answers
    predefined_questions = get_predefined_questions()
    answers_list_to_save = []
    for i, q_map in enumerate(predefined_questions):
        question_key = f"question_{i + 1}"
        answer_key = f"answer_{i + 1}"
        if question_key in received_data and answer_key in received_data:
            answers_list_to_save.append({
                "question": q_map["short_text"], 
                "answer": received_data[answer_key]
            })
    save_user_answers(user_id, answers_list_to_save)
    return {"message": "질문 답변 저장 완료"}

@app.get("/get-answers/{user_id}")
async def get_answers_api(user_id: str):
    answers = get_user_answers(user_id)
    return {"user_id": user_id, "answers": answers}

@app.post("/chat")
async def chat(request: ChatRequest, logged_in_user_id: Optional[str] = Depends(get_current_user_optional)):
    target_user_id = None
    if request.userId:
        target_user_id = request.userId
    elif logged_in_user_id:
        target_user_id = logged_in_user_id
    
    if not target_user_id:
        raise HTTPException(status_code=401, detail="답변 대상 사용자를 특정할 수 없습니다.")
    
    response = get_chatbot_response(request.query, target_user_id)
    return {"response": response}