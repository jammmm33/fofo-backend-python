# 🤖 FOPOFO - AI Backend (Python)

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/) [![LangChain](https://img.shields.io/badge/LangChain-0086D1?style=for-the-badge)](https://www.langchain.com/) [![Pinecone](https://img.shields.io/badge/Pinecone-0C59E8?style=for-the-badge&logo=pinecone&logoColor=white)](https://www.pinecone.io/)

> FOPOFO 프로젝트의 AI 기능을 전담하는 Python 백엔드 서버입니다. LangChain과 RAG(검색 증강 생성) 기술을 사용하여, 사용자가 업로드한 문서를 기반으로 답변하는 AI 챗봇 API를 제공합니다.

-  **🔗 웹 배포 링크:** https://staging.d1dbfs3o76ym6j.amplifyapp.com/

<br>

## **📜 주요 역할**

이 서버는 FOPOFO 서비스의 핵심인 AI 챗봇 기능을 독립적으로 처리하는 마이크로서비스입니다. 주 역할은 다음과 같습니다.

-   **문서 임베딩:** 사용자가 업로드한 문서를 벡터로 변환하여 Pinecone 벡터 데이터베이스에 저장합니다.
-   **문서 검색:** 사용자의 질문과 의미적으로 가장 유사한 문서 조각을 Pinecone에서 검색합니다.
-   **답변 생성:** 검색된 문서 내용을 참고 자료(Context)로 삼아, OpenAI의 LLM이 최종 답변을 생성하도록 요청하고 결과를 반환합니다.

## **🛠️ 기술 스택**

-   **`Python 3.11`**
-   **`FastAPI`**: 고성능 비동기 웹 프레임워크
-   **`LangChain`**: LLM 애플리케이션 개발을 위한 핵심 프레임워크
-   **`Pinecone`**: 벡터 검색을 위한 데이터베이스
-   **`OpenAI`**: 임베딩 및 LLM 모델 API
-   **`Gunicorn` / `Uvicorn`**: 비동기 웹 서버 게이트웨이 인터페이스(ASGI) 서버
-   **배포:** `AWS Elastic Beanstalk`

## **⚙️ 주요 API 엔드포인트**

-   **`POST /upload`**: 사용자가 업로드한 문서를 받아 비동기적으로 임베딩 및 Pinecone 저장을 처리합니다.
-   **`POST /chat`**: 사용자의 질문을 받아 RAG 파이프라인을 통해 답변을 생성하고 반환합니다.
-   **`DELETE /pinecone-vectors`**: 사용자의 문서 데이터를 Pinecone에서 삭제합니다.

<br>
