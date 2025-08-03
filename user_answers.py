# user_answers.py

from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
# ✅ chatbot_db 데이터베이스를 사용하도록 수정 (다른 파일과 통일)
db = client["chatbot_db"] 
# ✅ 이 변수 이름을 아래 함수들에서 사용합니다.
answers_collection = db["user_answers"]

def save_user_answers(user_id: str, answers: list): # ✅ answers 타입을 list로 명시
    answers_collection.update_one(
        {"user_id": user_id},
        {"$set": {"answers": answers, "updated_at": datetime.utcnow()}},
        upsert=True
    )

def get_user_qa_pairs(user_id: str) -> list:
    """
    특정 사용자의 질문-답변 데이터를 {'question': ..., 'answer': ...} 형태의 리스트로 반환합니다.
    """
    doc = answers_collection.find_one({"user_id": user_id})
    if doc and "answers" in doc:
        return doc['answers']
    return []

def get_user_answers(user_id: str) -> list: # ✅ 반환 타입을 list로 명시
    # ✅ 변수 이름을 answers_collection으로 수정
    doc = answers_collection.find_one({"user_id": user_id})
    if doc and "answers" in doc:
        # ✅ 질문 텍스트만 리스트로 반환
        return [item['question'] for item in doc['answers']] 
    return []

def delete_user_answers(user_id: str):
    # ✅ 변수 이름을 answers_collection으로 수정
    answers_collection.delete_one({"user_id": user_id})

# ✅ 중복 정의되었던 하단의 get_user_answers 함수는 삭제합니다.