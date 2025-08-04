import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from user_answers import delete_user_answers, save_user_answers, get_user_answers
from pydantic import BaseModel
from pinecone import Pinecone

# --- MongoDB 연결을 위한 라이브러리 추가 ---
from pymongo import MongoClient
from dotenv import load_dotenv

# --- 로컬 파일에서 함수 임포트 ---
from auth import get_current_user
from user_answers import delete_user_answers

# --- .env 파일 로드 ---
load_dotenv()

print("--- 환경 변수 로드 완료 ---")

# --- MongoDB 클라이언트 설정 ---
print("--- MongoDB 클라이언트 연결 시도 ---")

client = MongoClient(os.getenv("MONGO_URI"))
db = client["chatbot_db"]
chatbots_collection = db["chatbot_metadata"]

print("--- MongoDB 연결 및 컬렉션 설정 완료 ---")

# --- Pinecone 클라이언트 설정 ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "chatbot-index")

print("--- Pinecone 클라이언트 연결 시도 ---") # 이 줄 추가

pc = Pinecone(api_key=PINECONE_API_KEY)

print("--- Pinecone 클라이언트 설정 완료 ---")


# --- FastAPI 라우터 설정 ---
router = APIRouter()


class ChatbotMetadata(BaseModel):
    chatbot_id: str
    user_id: str
    created_at: Optional[str] = None


# --- API 엔드포인트 (MongoDB를 사용하도록 수정) ---

### 챗봇 등록(등록해야지 챗봇 수정/삭제 가능)
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
    """
    현재 로그인된 사용자의 저장된 질문 목록을 반환하는 API
    """
    questions = get_user_answers(user_id)
    if not questions:
        # 질문이 없는 경우 빈 리스트를 반환하거나, 에러를 발생시킬 수 있습니다.
        return []
    return questions


### 챗봇/개발자가 질문한 사용자 답변 삭제
@router.delete("/delete-chatbot")
def delete_chatbot(user_id: str = Depends(get_current_user)):
    """현재 로그인된 사용자의 모든 관련 데이터(Pinecone, MongoDB)를 삭제합니다."""
    # 1. MongoDB에서 챗봇 메타데이터 존재 여부 확인
    delete_result = chatbots_collection.delete_one({"user_id": user_id})
    
    if delete_result.deleted_count == 0:
         raise HTTPException(status_code=404, detail="삭제할 챗봇이 없습니다.")

    # 2. Pinecone에서 벡터 데이터 삭제
    try:
        if INDEX_NAME in pc.list_indexes().names():
            index = pc.Index(INDEX_NAME)
            index.delete(delete_all=True, namespace=user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone 데이터 삭제 중 오류 발생: {e}")

    # 3. MongoDB에서 질문 답변 데이터 삭제
    delete_user_answers(user_id)

    return {"message": "챗봇이 성공적으로 삭제되었습니다."}


### 챗봇 삭제 
@router.delete("/pinecone-vectors")
def delete_pinecone_vectors(user_id: str = Depends(get_current_user)):
    """현재 로그인된 사용자의 Pinecone 벡터 데이터만 삭제합니다."""
    try:
        if INDEX_NAME in pc.list_indexes().names():
            index = pc.Index(INDEX_NAME)
            # delete_all=True와 namespace를 함께 사용하면 해당 네임스페이스의 모든 벡터가 삭제됩니다.
            index.delete(delete_all=True, namespace=user_id)
            print(f"--- Pinecone 네임스페이스 '{user_id}'의 벡터 삭제 완료 ---")
            return {"message": f"사용자 '{user_id}'의 Pinecone 문서 벡터가 성공적으로 삭제되었습니다."}
        else:
            # 인덱스 자체가 없는 경우, 삭제할 것도 없으므로 성공으로 간주합니다.
            print(f"--- Pinecone 인덱스 '{INDEX_NAME}'가 존재하지 않아 삭제를 건너뜁니다. ---")
            return {"message": "삭제할 Pinecone 인덱스가 없습니다."}
    except Exception as e:
        print(f"!!!!!!!! ERROR: Pinecone 데이터 삭제 중 오류 발생 !!!!!!!!!!")
        print(e)
        raise HTTPException(status_code=500, detail=f"Pinecone 데이터 삭제 중 오류 발생: {e}")