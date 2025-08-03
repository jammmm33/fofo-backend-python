from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"

security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """
    기존처럼 토큰을 필수로 요구하는 함수입니다. 
    토큰이 없으면 401 에러를 발생시킵니다.
    """
    if credentials is None:
        raise HTTPException(status_code=401, detail="인증 자격 증명이 제공되지 않았습니다.")
    
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("userId")
        if user_id is None:
            raise HTTPException(status_code=401, detail="토큰에 userId 정보가 없습니다.")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")

# ✅ 토큰이 없어도 괜찮은 옵셔널 버전을 새로 만듭니다.
async def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """
    토큰이 있으면 사용자 ID를 반환하고, 없으면 None을 반환하는 함수입니다.
    공개 API에서 사용됩니다.
    """
    if credentials is None:
        return None # 토큰이 없으면 그냥 None을 반환
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("userId")
    except JWTError:
        return None # 토큰이 유효하지 않아도 오류 대신 None을 반환