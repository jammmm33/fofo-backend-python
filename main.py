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
    print(f"--- /upload 엔드포인트 호출됨, user_id: {user_id}, filename: {file.filename} ---") # 추가

    # 안전하게 임시 파일을 생성합니다. suffix는 파일 확장자를 유지하는 데 도움이 됩니다.
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        print(f"--- 임시 파일 생성됨: {tmp.name} ---") # 추가
        # 업로드된 파일의 내용을 임시 파일에 씁니다.
        try:
            content = await file.read()
            print(f"--- 파일 내용 읽기 완료. 크기: {len(content)} 바이트 ---") # 추가
            tmp.write(content)
            temp_file_path = tmp.name
            print(f"--- 파일 내용 임시 파일에 쓰기 완료 ---") # 추가
        except Exception as e:
            print(f"!!!!!!!! ERROR: 파일 읽기/쓰기 중 오류 발생: {e} !!!!!!!!!!") # 추가
            raise HTTPException(status_code=500, detail=f"파일 처리 중 오류 발생: {e}")

    try:
        # 임시 파일의 경로를 다음 함수로 전달합니다.
        print(f"--- store_document_vectors 호출 시작 ---") # 추가
        store_document_vectors(temp_file_path, user_id)
        print(f"--- store_document_vectors 호출 완료 ---") # 추가
    except Exception as e:
        print(f"!!!!!!!! ERROR: store_document_vectors 함수 호출 중 오류 발생: {e} !!!!!!!!!!") # 추가
        raise HTTPException(status_code=500, detail=f"문서 벡터 저장 중 오류 발생: {e}")
    finally:
        # 함수 실행이 성공하든 실패하든, 임시 파일은 항상 삭제합니다.
        os.remove(temp_file_path)
        print(f"--- 임시 파일 삭제 완료 ---") # 추가

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