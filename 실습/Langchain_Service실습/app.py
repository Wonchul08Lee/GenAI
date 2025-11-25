import streamlit as st
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

st.set_page_config(page_title="RAG QA System", layout="wide")
tab1, tab2 = st.tabs(["Data 저장", "QA"])

@st.cache_resource
def load_models():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    model_id = "monologg/koelectra-base-v3-finetuned-korquad"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)

    qa_pipeline = pipeline(
        "question-answering", 
        model=model, 
        tokenizer=tokenizer, 
        device = -1, 
        max_length=512,
        do_sample=False,
        temperature=0.1,
        trucnation=True
    )

    return embedding_model, tokenizer, qa_pipeline

embedding_model, qa_tokenizer, text_gen = load_models()


st.sidebar.header("이원철님, 안녕하세요!")
st.sidebar.markdown("Langchain 이용 서비스 실습입니다.")


with tab1:
    st.header("CSV 문서 업로드 및 ChromaDB 저장")

    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, encoding='utf-8')

        if 'text' not in df.columns:
            st.error("CSV 파일에 'text' 열이 없습니다. 올바른 파일을 업로드해주세요.")
        else:
            docs = [Document(page_content=text) for text in df['text'].tolist()]
            if st.button("ChromaDB에 저장"):
                Chroma.from_documents(
                    documents=docs,
                    embedding=embedding_model,
                    persist_directory="./chroma_db"
                )
                st.success("문서가 ChromaDB에 성공적으로 저장되었습니다.")

with tab2:
    st.header("질문 답변 시스템")
    question = st.text_input("질문을 입력하세요:")

    if st.button("답변 생성"):

        if not question:
            st.warning("질문을 입력해주세요.")
        else:
            retriever = Chroma(
                embedding_function = embedding_model,
                persist_directory="./chroma_db"
            ).as_retriever(search_kwargs={"k":3})    
        docs = retriever.invoke(question)             

        if not docs:
            st.error("유사 문서를 찾을 수 없습니다. 먼저 문서를 업로드하고 저장하세요.")
        else:
            best_context = docs[0].page_content
            result = text_gen(question=question, context=best_context)

            st.subheader("정답")
            st.write(result['answer'])

            st.subheader("선택된 문서[top 3]")
            for i, doc in enumerate(docs):
                st.markdown(f"**문서 {i+1}:** {doc.page_content}")