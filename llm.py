# llm.py

import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def store_document_vectors(file_path: str, user_id: str):
    # ✅ --- 여기에 print 문을 추가하여 디버깅합니다 ---
    try:
        print(f"--- 1. 파일 처리 시작: {file_path} ---")

        if file_path.endswith(".pdf"):
            documents = PyPDFLoader(file_path).load()
        elif file_path.endswith(".docx"):
            documents = Docx2txtLoader(file_path).load()
        elif file_path.endswith(".txt"):
            documents = TextLoader(file_path).load()
        else:
            raise ValueError("지원하지 않는 파일 형식입니다.")
        
        print(f"--- 2. 파일 로드 완료, 텍스트 분할 시작 ---")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(documents)

        print(f"--- 3. 텍스트 분할 완료. 총 {len(texts)}개의 조각 ---")

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX", "chatbot-index")

        print(f"--- 4. Pinecone 인덱스 확인 시작 ---")

        if index_name not in pc.list_indexes().names():
            print(f"--- 4a. 인덱스가 없으므로 새로 생성합니다: {index_name} ---")
            pc.create_index(
                name=index_name,
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        print(f"--- 5. Pinecone 인덱스 준비 완료. OpenAI 임베딩 및 Pinecone 업로드 시작... (이 단계는 시간이 걸릴 수 있습니다) ---")
        
        PineconeVectorStore.from_documents(
            documents=texts,
            embedding=embedding,
            index_name=index_name,
            namespace=user_id
        )

        print(f"--- 6. Pinecone 업로드 완료! ---")
    
    except Exception as e:
        print(f"!!!!!!!! ERROR: store_document_vectors 함수에서 오류 발생 !!!!!!!!!!")
        print(e)
        raise e # 오류를 다시 발생시켜 서버에 알림
    # ✅ --- 여기까지 디버깅 코드 추가 ---