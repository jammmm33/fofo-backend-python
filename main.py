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
import tempfile
import os

# --- Pydantic 모델 정의 ---
class AnswersRequest(BaseModel):
    answers: Dict[str, str]

class ChatRequest(BaseModel):
    query: str
    userId: Optional[str] = None

print("--- FastAPI 앱 초기화 시작 ---") # 이 줄 추가
app = FastAPI()
print("--- FastAPI 인스턴스 생성 완료 ---") # 이 줄 추가

# ✅ 1. CORS 설정을 환경 변수를 사용하도록 수정합니다.
origins = [
    "https://staging.d1dbfs3o76ym6j.amplifyapp.com",
    "https://distragng_dthfjhjfdhfjgsn_gadfljtyago.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # 모든 메소드 허용
    allow_headers=["*"], # 모든 헤더 허용
)

print("--- CORS 미들웨어 추가 완료 ---") # 이 줄 추가

app.include_router(chatbot_router)
print("--- 챗봇 라우터 포함 완료 ---") # 이 줄 추가

@app.get("/")
async def read_root():
    return {"message": "안녕하세요! 챗봇 API 서버입니다."}


@app.post("/upload")
async def upload(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    print(f"--- /upload 엔드포인트 호출됨, user_id: {user_id}, filename: {file.filename} ---")

    # 파일 크기 로깅 (추가)
    try:
        # 파일 내용 읽기 (스트리밍 방식이므로 용량이 크면 지연될 수 있음)
        content = await file.read() # 이 부분에서 문제가 발생할 가능성이 가장 높음
        print(f"--- 파일 내용 읽기 완료. 크기: {len(content)} 바이트 ---")

        # 임시 파일 생성 및 쓰기
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            tmp.write(content)
            temp_file_path = tmp.name
        print(f"--- 파일 내용 임시 파일 '{temp_file_path}'에 쓰기 완료 ---")

        # store_document_vectors 호출
        print(f"--- store_document_vectors 호출 시작 ---")
        store_document_vectors(temp_file_path, user_id)
        print(f"--- store_document_vectors 호출 완료 ---")

    except Exception as e:
        print(f"!!!!!!!! ERROR: /upload 처리 중 치명적인 오류 발생: {e} !!!!!!!!!!")
        # 오류 발생 시 임시 파일이 남아있을 수 있으므로 cleanup 추가
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"파일 업로드 처리 중 오류 발생: {e}")
    finally:
        # 이 finally 블록은 tmp.name이 정의되지 않았을 때 오류를 일으킬 수 있으므로
        # try-except-finally 구조를 더 명확히 합니다.
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"--- 임시 파일 '{temp_file_path}' 삭제 완료 ---")

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