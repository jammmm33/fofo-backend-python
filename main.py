import shutil
import os
import tempfile
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks
from user_answers import save_user_answers, get_user_answers
from llm import store_document_vectors
from rag_chatbot import get_chatbot_response
from auth import get_current_user, get_current_user_optional
from pydantic import BaseModel
from utils import get_predefined_questions
from fastapi.middleware.cors import CORSMiddleware
from chatbot_manager import router as chatbot_router

# --- Pydantic 모델 정의 ---
class AnswersRequest(BaseModel):
    answers: Dict[str, str]

class ChatRequest(BaseModel):
    query: str
    userId: Optional[str] = None

app = FastAPI()

# --- CORS 설정 ---
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

# --- 라우터 포함 ---
app.include_router(chatbot_router)

# --- 백그라운드 작업을 위한 헬퍼 함수 ---
def store_and_cleanup_task(file_path: str, user_id: str):
    """
    백그라운드에서 문서 벡터화 및 임시 파일 삭제를 수행
    """
    try:
        store_document_vectors(file_path, user_id)
    except Exception as e:
        print(f"!!!!!!!! [BACKGROUND ERROR] 파일 처리 중 오류 발생: {e} !!!!!!!!!!")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"--- [백그라운드] 임시 파일 '{file_path}' 삭제 완료 ---")

# --- 기존 API 엔드포인트들 (변경 없음) ---
@app.get("/")
async def read_root():
    return {"message": "안녕하세요! 챗봇 API 서버입니다."}

# --- ✅ 수정된 /upload 엔드포인트 ---
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

    # 시간이 오래 걸리는 작업을 백그라운드로 넘기고 즉시 응답
    background_tasks.add_task(store_and_cleanup_task, temp_file_path, user_id)

    return {"message": "파일 업로드가 시작되었습니다. 처리가 완료되면 사용할 수 있습니다."}


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