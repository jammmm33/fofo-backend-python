# rag_chatbot.py

import os
from dotenv import load_dotenv
import unicodedata
import re
from utils import get_predefined_questions
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from user_answers import get_user_qa_pairs # ✅ MongoDB 답변 조회를 위해 import


load_dotenv()

# ✅ OpenAI 임베딩 모델은 한 번만 초기화하여 재사용합니다.
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# ✅ LangChain 모델도 한 번만 초기화하여 재사용합니다.
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

def clean_string(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized_text = unicodedata.normalize('NFC', text)
    return re.sub(r'\s+', '', normalized_text)

def get_chatbot_response(query: str, user_id: str) -> str:
    # --- 1. MongoDB에서 답변을 찾기 전, 받은 질문(query)을 변환합니다. ---
    
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

    # --- 2. MongoDB에서 정확히 일치하는 답변을 찾지 못하면, Pinecone에서 문서를 검색합니다. ---
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX")

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding_model,
        namespace=user_id
    )
    
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
    
    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs=chain_type_kwargs
    )
    
    result = qa_chain.invoke({"query": query})
    return result.get("result", "답변을 생성하지 못했습니다.")