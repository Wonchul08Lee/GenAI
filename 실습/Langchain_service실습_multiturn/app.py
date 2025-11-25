import streamlit as st
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

st.set_page_config(page_title="Multiturn RAG QA System", layout="wide")
st.sidebar.header("이원철님, 안녕하세요!")
mode = st.sidebar.radio("모드를 선택하세요:", ("데이터 저장", "QA"))

@st.cache_resource
def load_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def load_qa_model():
    model_id = "monologg/koelectra-base-v3-finetuned-korquad"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)

    pipe = pipeline(
        "question-answering", 
        model=model, 
        tokenizer=tokenizer, 
        device = -1, 
        max_length=512,
        do_sample=False,
        temperature=0.1,
        truncation=True
    )

    return pipe
embedding_model = load_embedding_model()
text_gen = load_qa_model()

if mode == "데이터 저장":
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

elif mode == "QA":
    st.header("질문 답변 시스템")

    @st.cache_resource
    def get_retriever():
        res = Chroma(
            embedding_function=embedding_model,
            persist_directory="./chroma_db"
        ).as_retriever(search_kwargs={"k": 3})
        return res
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    user_input = st.chat_input("질문을 입력하세요:")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        if "참고문서" in user_input or "참고 문서" in user_input:
            if st.session_state.last_contexts:
                ref_response = "### 참고문서:\n" + "\n".join(
                    f"{i+1}. {ctx}" for i, ctx in enumerate(st.session_state.last_contexts)
                )
            else:
                ref_response = "참고문서가 없습니다."
        
            st.session_state.messages.append({"role": "assistant", "content": ref_response})
            with st.chat_message("assistant"):
                st.markdown(ref_response)
        else:
            retriever = get_retriever()
            docs = retriever.invoke(user_input)
            top_docs = docs[:3]
            best_contexts = [doc.page_content for doc in top_docs]
            st.session_state.last_contexts = best_contexts
            combined_context = "\n".join(best_contexts)

            result = text_gen(question=user_input, context=combined_context)    
            answer = result['answer']

            bot_response = f"### 답변:{answer}"
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            with st.chat_message("assistant"):
                st.markdown(bot_response)
