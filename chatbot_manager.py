# chatbot_manager.py (수정된 전체 코드)

import os
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from pinecone import Pinecone
from pymongo import MongoClient
from dotenv import load_dotenv
from auth import get_current_user
from user_answers import delete_user_answers

load_dotenv()

# --- MongoDB 클라이언트 설정 ---
client = MongoClient(os.getenv("MONGO_URI"))
db = client["chatbot_db"]
chatbots_collection = db["chatbot_metadata"]

# --- Pinecone 클라이언트 변수 선언 ---
pc: Optional[Pinecone] = None # ✅ Pinecone 클라이언트를 초기화하지 않고 변수만 선언합니다.
INDEX_NAME = os.getenv("PINECONE_INDEX", "chatbot-index")

# --- Pinecone 클라이언트를 초기화하는 함수를 새로 만듭니다 ---
def init_pinecone_client(): # ✅ Pinecone 초기화 로직을 별도의 함수로 분리했습니다.
    """
    이 함수는 main.py의 lifespan에서 호출됩니다.
    """
    global pc # 전역 변수 pc를 사용하겠다고 선언
    print("--- Pinecone 클라이언트 연결 시도 ---")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("--- Pinecone 클라이언트 설정 완료 ---")


# --- FastAPI 라우터 설정 ---
router = APIRouter()

class ChatbotMetadata(BaseModel):
    chatbot_id: str
    user_id: str
    created_at: Optional[str] = None

# --- API 엔드포인트들 ---

# (register_chatbot, get_my_chatbot, get_answers_api 함수는 변경 없음)
### 챗봇 등록
@router.post("/register-chatbot")
def register_chatbot(metadata: ChatbotMetadata, user_id: str = Depends(get_current_user)):
    chatbots_collection.replace_one(
        {'user_id': user_id}, 
        metadata.dict(), 
        upsert=True
    )
    return {"message": "챗봇이 성공적으로 등록되었습니다.", "chatbot": metadata}

### 챗봇 조회
@router.get("/my-chatbot")
def get_my_chatbot(user_id: str = Depends(get_current_user)):
    chatbot_data = chatbots_collection.find_one({"user_id": user_id})
    if not chatbot_data:
        raise HTTPException(status_code=404, detail="등록된 챗봇이 존재하지 않습니다.")
    chatbot_data["_id"] = str(chatbot_data["_id"])
    return chatbot_data

@router.get("/get-answers")
async def get_answers_api(user_id: str = Depends(get_current_user)):
    questions = get_user_answers(user_id)
    if not questions:
        return []
    return questions


### 챗봇/개발자가 질문한 사용자 답변 삭제
@router.delete("/delete-chatbot")
def delete_chatbot(user_id: str = Depends(get_current_user)):
    if pc is None: # ✅ Pinecone 클라이언트가 초기화되었는지 확인하는 로직 추가
        raise HTTPException(status_code=503, detail="Pinecone 클라이언트가 아직 준비되지 않았습니다.")
    
    delete_result = chatbots_collection.delete_one({"user_id": user_id})
    if delete_result.deleted_count == 0:
         raise HTTPException(status_code=404, detail="삭제할 챗봇이 없습니다.")

    try:
        if INDEX_NAME in pc.list_indexes().names():
            index = pc.Index(INDEX_NAME)
            index.delete(delete_all=True, namespace=user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone 데이터 삭제 중 오류 발생: {e}")

    delete_user_answers(user_id)
    return {"message": "챗봇이 성공적으로 삭제되었습니다."}

### 챗봇 삭제 
@router.delete("/pinecone-vectors")
def delete_pinecone_vectors(user_id: str = Depends(get_current_user)):
    if pc is None: # ✅ Pinecone 클라이언트가 초기화되었는지 확인하는 로직 추가
        raise HTTPException(status_code=503, detail="Pinecone 클라이언트가 아직 준비되지 않았습니다.")

    try:
        if INDEX_NAME in pc.list_indexes().names():
            index = pc.Index(INDEX_NAME)
            index.delete(delete_all=True, namespace=user_id)
            print(f"--- Pinecone 네임스페이스 '{user_id}'의 벡터 삭제 완료 ---")
            return {"message": f"사용자 '{user_id}'의 Pinecone 문서 벡터가 성공적으로 삭제되었습니다."}
        else:
            print(f"--- Pinecone 인덱스 '{INDEX_NAME}'가 존재하지 않아 삭제를 건너뜁니다. ---")
            return {"message": "삭제할 Pinecone 인덱스가 없습니다."}
    except Exception as e:
        print(f"!!!!!!!! ERROR: Pinecone 데이터 삭제 중 오류 발생 !!!!!!!!!!")
        print(e)
        raise HTTPException(status_code=500, detail=f"Pinecone 데이터 삭제 중 오류 발생: {e}")