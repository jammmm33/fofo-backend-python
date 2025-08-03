# main.py

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from user_answers import save_user_answers, get_user_answers
from llm import store_document_vectors
from rag_chatbot import get_chatbot_response
from auth import get_current_user, get_current_user_optional # ✅ get_current_user_optional을 함께 import
from typing import Dict, Optional
from pydantic import BaseModel
from utils import get_predefined_questions
from fastapi.middleware.cors import CORSMiddleware
from chatbot_manager import router as chatbot_router
import os

# --- Pydantic 모델 정의 ---
class AnswersRequest(BaseModel):
    answers: Dict[str, str]

class ChatRequest(BaseModel):
    query: str
    userId: Optional[str] = None

app = FastAPI()

# ✅ 1. CORS 설정을 환경 변수를 사용하도록 수정합니다.
FRONTEND_URL = os.getenv("FRONTEND_URL")
origins = []
if FRONTEND_URL:
    origins.append(FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chatbot_router)

@app.get("/")
async def read_root():
    return {"message": "안녕하세요! 챗봇 API 서버입니다."}


@app.post("/upload")
async def upload(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    # 'user_files' 폴더가 없으면 생성
    os.makedirs('user_files', exist_ok=True)
    
    file_path = f"user_files/{user_id}_{file.filename}"
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        store_document_vectors(file_path, user_id)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"'{file.filename}' 처리 중 오류 발생: {str(e)}")
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return {"message": "문서 업로드 및 벡터 저장 완료"}


@app.post("/save-answers")
async def save_answers_api(request: AnswersRequest, user_id: str = Depends(get_current_user)):
    # ... (이 함수는 기존 코드와 동일)
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


# ✅ --- '/chat' API를 이 올바른 버전 하나로 교체합니다 --- ✅
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