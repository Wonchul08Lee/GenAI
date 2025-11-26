import streamlit as st
import pandas as pd

from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from typing import List, Literal, Optional, TypedDict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
import uuid

import re
import docx2txt
import tempfile

emdeding_id = "jhgan/ko-sbert-sts"
qa_model_id = "monologg/koelectra-base-v3-finetuned-korquad"
gen_model_id = "beomi/gemma-ko-2b"

tokenizer_opt = {
    "max_length": 512, 
    "truncation": True,
    "do_sample": True,
    "device": -1
}


filepath = "./NewBie_개발환경가이드_tmp"
persist_directory = "./chromaDB_Chatbot"

st.set_page_config(page_title="GenAI Intensive Project", layout="wide")
st.sidebar.header("안녕하세요!")
mode = st.sidebar.radio("모드를 선택하세요:", ("데이터 저장", "Chatbot"))

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=emdeding_id)

@st.cache_resource
def load_QA_model(tokenizer_opt):
    qa_pipeline = pipeline(
        task="question-answering",
        model=qa_model_id,
        tokenizer_kwargs=tokenizer_opt,
        device=-1 
    )
    return qa_pipeline

@st.cache_resource
def load_Gen_model():
    tokenizer = AutoTokenizer.from_pretrained(gen_model_id)
    model = AutoModelForCausalLM.from_pretrained(gen_model_id)
    return tokenizer, model


# Word 문서 로드 함수
@st.cache_data
def load_word_documents(file: str):
    
    loader = UnstructuredWordDocumentLoader(file)
    docs = loader.load()

    sentences = []
    progress_bar = st.progress(0)
    total_docs = len(docs)

    for idx, doc in enumerate(docs):
        all_text = doc.page_content
        page_sentences = re.split(r'\.|\n', all_text)
        page_sentences = [s.strip() for s in page_sentences if s.strip()]
        sentences.extend(page_sentences)

        progress = int((idx + 1) / total_docs * 100)
        progress_bar.progress(progress)
        st.markdown(f"진행률: {progress}%")
        
    st.write(f"문장 개수: {len(sentences)}")
    return sentences


def load_word_documents_docx2txt(uploaded_file):
    """
    uploaded_file: Streamlit UploadedFile 객체 또는 파일 경로
    """
    # Streamlit 업로드 파일 처리
    if hasattr(uploaded_file, "read"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
    else:
        tmp_path = uploaded_file

    st.markdown("Word 문서 로드 중...")
    
    # docx2txt로 텍스트 추출
    text = docx2txt.process(tmp_path)
    
    # 문장 단위 분리
    lines = text.split("\n")
    total_lines = len(lines)
    sentences = []

    progress_bar = st.progress(0)

    for idx, line in enumerate(lines):
        line = line.strip()
        if line:
            for s in re.split(r'\.|\n', line):
                s = s.strip()
                if s:
                    sentences.append(s)

        # 진행률 업데이트
        progress = int((idx + 1) / total_lines * 100)
        progress_bar.progress(progress)

    st.write(f"문장 개수: {len(sentences)}")
    return sentences

# VectorDB 저장 함수
def save_vectorDB(docs, persist_directory):
    embedding_model = load_embedding_model()

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_directory
)
    return vector_db

def save_vectorDB_with_progress(docs, persist_directory):
    embedding_model = load_embedding_model()  # HuggingFaceEmbeddings

    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

    progress_bar = st.progress(0)
    total_docs = len(docs)

    for idx, doc in enumerate(docs):
        vector_db.add_documents([doc])
        progress = int((idx + 1) / total_docs * 100)
        progress_bar.progress(progress)

    return vector_db


# VectorDB에서 검색(retrieve) 함수
def load_vectorDB(persist_directory):
    embedding_model = load_embedding_model()
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
)
    return vector_db

def run_qa(question, vector_db, qa_model_id, tokenizer_opt):
    retriever = vector_db.as_retriever(search_kwargs={"k":5})
    relevant_docs = retriever.invoke(question)
    if len(relevant_docs) == 0: 
        st.warning("검색 결과가 없습니다.")
        return None, None
    # 여러 문서를 context로 합침 (길이 증가)
    best_context = "\n".join([doc.page_content for doc in relevant_docs[:5]])
    #st.markdown("### 검색된 컨텍스트")
    #st.markdown(best_context)

    qa_pipeline = load_QA_model(tokenizer_opt)
    result = qa_pipeline(question=question,context=best_context)

    st.markdown("### QA 답변")
    st.write(result['answer'])

    return result, best_context


def remove_duplicate_sentences(text):
    sentences = text.split('\n')
    seen = set()
    result = []
    for s in sentences:
        s_strip = s.strip()
        if s_strip and s_strip not in seen:
            result.append(s_strip)
            seen.add(s_strip)
    return '\n'.join(result)

if mode == "데이터 저장":
    st.header("Word 문서 업로드 및 ChromaDB 저장")
    
    uploaded_file = st.file_uploader("Word 파일 업로드", type=["docx"])
    if uploaded_file is None:
        st.warning("Word 파일을 업로드해주세요.")
    else:
       
        with st.spinner("문서 처리 중..."):
            sentences = load_word_documents_docx2txt(uploaded_file)
        st.success(f"{uploaded_file.name} 는 성공적으로 로드되었습니다.")
        
        docs = [Document(page_content=s) for s in sentences]  # 각 문장을 Document로 변환
        if st.button("ChromaDB에 저장"):
            
            with st.spinner("ChromaDB 저장 중..."):
                save_vectorDB_with_progress(docs, persist_directory)
            st.success("문서가 ChromaDB에 성공적으로 저장되었습니다.")

elif mode == "Chatbot":
    st.header("모든게 해결 될 지니...")
    question = st.text_input("질문을 입력하세요:")
    
    if st.button("답변 생성"):
        if not question:
            st.warning("질문을 입력해주세요.")
        else:
            vector_db = load_vectorDB(persist_directory)
            result, best_context = run_qa(question, vector_db, qa_model_id, tokenizer_opt)

            # 생성형 모델 준비
            gen_tokenizer, gen_model = load_Gen_model()
            gen_tokenizer.model_max_length = 1024
            gen_pipeline = pipeline(
                task="text-generation",
                model=gen_model,
                tokenizer=gen_tokenizer,
                max_new_tokens=200,
                truncation=True,
                do_sample=True,
                temperature=0.1,
                top_p=0.8,
                repetition_penalty=1.2,
                device=-1
            )
            gen_llm = HuggingFacePipeline(pipeline=gen_pipeline)
                    
            prompt = PromptTemplate.from_template("""                
                당신은 경험 많은 소프트웨어 개발자입니다. 
                아래 문서를 참고하여, 질문에 대한 최종 답변만 출력하세요. 
                프롬프트 내용, 문서 내용, 설명은 출력하지 마세요.

                [참고 문서 시작] 
                {context} 
                [참고 문서 끝] 

                [정답 시작] 
                {answer} 
                [정답 끝] 
                
                === 여기서부터 아래에는 최종 답변만 출력합니다 === 
                """)

            chain = prompt | gen_llm

            final_response = chain.invoke({
                "question": question,
                "context": best_context,
                "answer": result['answer']
            })
            final_response = remove_duplicate_sentences(final_response)
            final_response = ". ".join(list(dict.fromkeys(final_response.split(". "))))
            # === 이후 문장만 추출
            if "===" in final_response:
                final_response = final_response.split("===")[-1].strip()

            # [최종 답변 시작] 이후 문장만 추출
            if "[최종 답변 시작]" in final_response:
                final_response = final_response.split("[최종 답변 시작]")[-1].strip()

            # [최종 답변 끝] 앞까지만
            if "[최종 답변 끝]" in final_response:
                final_response = final_response.split("[최종 답변 끝]")[0].strip()

            st.write(final_response)