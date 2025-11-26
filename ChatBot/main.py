import streamlit as st
import pandas as pd

from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from typing import List, Literal, Optional, TypedDict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TrainingArguments, Trainer
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
import re
import docx2txt
import tempfile
from finetune_LoRA import lora_finetune_from_feedback
from feedback import display_messages, save_feedback

from load_model import (
    embedding_model_id,
    qa_model_id,
    gen_model_id,
    load_embedding_model,
    load_QA_model,
    load_Gen_model,
    tokenizer_opt,
)


def load_word_documents_docx2txt(uploaded_file):    
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
    vector_db = Chroma(persist_directory=persist_directory,embedding_function=embedding_model)
    return vector_db

def run_qa(question, vector_db, qa_model_id, tokenizer_opt):
    retriever = vector_db.as_retriever(search_kwargs={"k":5})
    relevant_docs = retriever.invoke(question)
    if len(relevant_docs) == 0: 
        st.warning("검색 결과가 없습니다.")
        return None, None

    # 여러 문서를 context로 합침
    best_context = "\n".join([doc.page_content for doc in relevant_docs[:5]])

    # QA pipeline 불러오기
    safe_opts = {"truncation": True, "padding": "max_length", "max_length": tokenizer_opt['max_length']}
    qa_pipeline = pipeline(
        task="question-answering",
        model=qa_model_id,
        tokenizer_kwargs=safe_opts,
        device=-1
    )

    # context 토큰화
    tokens = qa_pipeline.tokenizer(best_context, truncation=False)['input_ids']
    max_len = tokenizer_opt['max_length']

    # stride가 max_len보다 크면 max_len//2로 조정
    stride = min(tokenizer_opt.get('stride', 32), max_len - 1)

    # context가 너무 길면 뒤쪽 토큰만 사용
    if len(tokens) > max_len:
        tokens = tokens[-max_len:]
    best_context = qa_pipeline.tokenizer.decode(tokens)

    result = qa_pipeline(question=question, context=best_context, stride=stride, max_seq_len=max_len)

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

def prepare_gen_pipeline():
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
    return HuggingFacePipeline(pipeline=gen_pipeline)

def get_prompt_template():
    return PromptTemplate.from_template("""
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

def extract_final_answer(raw_text):
    """
    프롬프트 결과에서 최종 답변만 추출
    """
    text = raw_text
    # === 이후 문장만 추출
    if "===" in text:
        text = text.split("===")[-1].strip()
    # [최종 답변 시작] 이후 문장만 추출
    if "[최종 답변 시작]" in text:
        text = text.split("[최종 답변 시작]")[-1].strip()
    # [최종 답변 끝] 앞까지만
    if "[최종 답변 끝]" in text:
        text = text.split("[최종 답변 끝]")[0].strip()
    return text

def save_chat_history(question, answer):
    """
    세션에 질문/답변 저장
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({
        "question": question,
        "answer": answer
    })

# history를 문자열로 합침 (질문/답변 모두 포함)
def get_history_text():
    history = ""
    if "chat_history" in st.session_state:
        for chat in st.session_state.chat_history:
            history += f"Q: {chat['question']}\nA: {chat['answer']}\n"
    return history

def truncate_context(text, max_len=512):
    tokens = text.split()  # 간단히 공백 기준 토큰화
    if len(tokens) > max_len:
        tokens = tokens[-max_len:]  # 뒤쪽 최근 내용만 유지
    return " ".join(tokens)


# --- Streamlit UI ---
mode = st.sidebar.radio("모드를 선택하세요:", ("데이터 저장", "Chatbot"))
persist_directory = "./chromaDB_Chatbot"

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

    tab1, tab2 = st.tabs(["Chatbot", "Finetune"])
    with tab1:
        # 세션 상태에 질문/답변 리스트 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "last_contexts" not in st.session_state:
            st.session_state.last_contexts = []
        if "feedback_values" not in st.session_state:
            st.session_state.feedback_values = {}

        display_messages()

        user_input = st.chat_input("질문을 입력하세요:")
        
        if user_input:
            if not user_input:
                st.warning("질문을 입력해주세요.")
            else:
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                # history와 현재 질문을 합쳐서 질의 생성
                history_text = get_history_text().strip()
                full_query = f"{history_text} 현재 질문: {user_input}" if history_text else user_input
                full_query = " ".join(full_query.splitlines())
                if len(full_query) < 10:
                    full_query = user_input

                vector_db = load_vectorDB(persist_directory)
                result, best_context = run_qa(full_query, vector_db, qa_model_id, tokenizer_opt)

                # 생성형 모델 준비
                gen_llm = prepare_gen_pipeline()                    
                prompt = get_prompt_template()

                chain = prompt | gen_llm

                final_response = chain.invoke({
                    "question": user_input,
                    "context": best_context,
                    "answer": result['answer']
                })
                final_response = remove_duplicate_sentences(final_response)
                final_response = ". ".join(list(dict.fromkeys(final_response.split(". "))))

                final_response = extract_final_answer(final_response)
                save_chat_history(user_input, final_response)

                st.session_state.messages.append({"role": "assistant", "content": final_response})                
                st.rerun()
    with tab2:
        if st.button("피드백으로 LoRA 파인튜닝 실행"):
            feedbacks = st.session_state.get("feedbacks", [])
            st.write(f"수집된 피드백 개수: {len(feedbacks)}")
            from collections import Counter
            label_counts = Counter([f["feedback"] for f in feedbacks])
            if len(feedbacks) < 2:
                st.info("done")
            elif min(label_counts.values(), default=0) < 2:
                st.warning("각 라벨별로 최소 2개 이상의 피드백이 필요합니다.")
            else:
                lora_finetune_from_feedback(feedbacks)