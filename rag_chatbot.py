# rag_chatbot.py (최종 수정 완료본)

import os
from dotenv import load_dotenv
import unicodedata
import re
from utils import get_predefined_questions
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone
# ✅ RetrievalQA를 import 합니다.
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from user_answers import get_user_qa_pairs

load_dotenv()

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

def clean_string(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized_text = unicodedata.normalize('NFC', text)
    return re.sub(r'\s+', '', normalized_text)

def get_chatbot_response(query: str, user_id: str) -> str:
    # --- 1. MongoDB에서 저장된 Q&A 답변 우선 검색 ---
    predefined_questions = get_predefined_questions()
    query_to_find = query
    for q_map in predefined_questions:
        if q_map["full_text"] == query:
            query_to_find = q_map["short_text"]
            break
    
    qa_pairs = get_user_qa_pairs(user_id)
    if qa_pairs:
        for pair in qa_pairs:
            if pair.get("question") == query_to_find:
                print(f"--- MongoDB 답변 사용 ('{query_to_find}' 와 정확히 일치) ---")
                return pair.get("answer", "저장된 답변을 찾았으나 내용이 없습니다.")

    # --- 2. MongoDB에 답변이 없으면 Pinecone 문서 검색 및 답변 생성 ---
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX")

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding_model,
        namespace=user_id
    )
    
    # ✅ 프롬프트 템플릿은 강력한 규칙을 그대로 유지합니다.
    prompt_template = """
    당신은 '컨텍스트'로 주어진 문서 내용만을 기반으로 답변해야 하는 AI 챗봇입니다.
    당신의 역할은 사용자가 업로드한 이력서, 자기소개서 등의 내용을 바탕으로 면접관의 질문에 답변하는 것입니다.

    ***규칙***
    1. 답변은 반드시 아래에 제공되는 '컨텍스트' 안의 정보만을 사용해야 합니다.
    2. '컨텍스트'에 질문에 대한 답변을 찾을 수 없는 경우, 절대로 외부 지식을 사용하거나 내용을 추측해서 답변하지 마십시오.
    3. 정보가 없는 경우에는 반드시 "제출된 자료를 통해서는 해당 질문에 충분히 답변드리기 어렵습니다." 라고만 답변해야 합니다.
    4. 당신의 역할은 구직자(지원자)이며, 친절하고 진정성 있는 전문가의 말투를 사용해주세요.

    컨텍스트:
    {context}

    질문:
    {question}

    답변:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    # ✅ --- 이 부분이 수정의 핵심입니다 --- ✅
    # RetrievalQA 체인을 사용하면, retriever가 문서를 찾아 프롬프트의 {context}에
    # 자동으로 넣어주는 과정을 한 번에 처리해 매우 안정적입니다.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 찾은 문서들을 하나로 묶어 컨텍스트로 사용
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True # 디버깅을 위해 소스 문서도 함께 반환
    )
    
    # 체인을 실행하여 결과를 받습니다.
    result = qa_chain.invoke({"query": query})
    
    # 검색된 소스 문서가 있는지 확인하여 로그에 남깁니다.
    if result.get("source_documents"):
        print(f"--- 검색된 컨텍스트: {result['source_documents'][0].page_content[:500]}... ---")
    else:
        print("--- 검색된 컨텍스트가 없습니다. ---")

    # 최종 답변을 반환합니다.
    return result.get("result", "답변을 생성하지 못했습니다.")