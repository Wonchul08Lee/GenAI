import streamlit as st

st.header("User input")
name = st.text_input("Enter your name:")
st.write(f"입력한 이름: {name}")

sop = st.text_area("자기소개를 입력하세요:")
st.write(f"자기소개 내용: {sop}") 

age = st.number_input("나이를 입력하세요:", min_value=0, max_value=120, step=1)
st.write(f"입력한 나이: {age}세")

birth = st.date_input("생일을 선택하세요:")
st.write(f"선택한 생일: {birth}")

my_color = st.selectbox("좋아하는 색상을 선택하세요:", ["빨강", "파랑", "초록", "노랑"])
st.write(f"선택한 색상: {my_color}")

hobby = st.multiselect("취미를 선택하세요:", ["독서", "운동", "여행", "요리"])
st.write(f"선택한 취미: {', '.join(hobby)}")

gender = st.radio("성별을 선택하세요:", ["남성", "여성", "기타"])
st.write(f"선택한 성별: {gender}")

agree = st.checkbox("약관에 동의합니다.")
if agree:
    st.write("약관에 동의하셨습니다.")

score = st.slider("점수를 선택하세요:", min_value=0, max_value=100, step=1)
st.write(f"선택한 점수: {score}점")

st.header("Button and events")
if st.button("클릭하세요"):
    st.write("버튼이 클릭되었습니다!")

toggle = st.toggle("토글 스위치")
st.write(f"toggle 상태: {toggle}")

import pandas as pd
import numpy as np

st.header("데이터 시각화")
df = pd.DataFrame(np.random.randn(5, 3), columns=['A', 'B', 'C'])
st.write(df)
st.dataframe(df)

col1, col2 = st.columns(2)
with col1:
    st.write(df)
with col2:
    st.write(df)

st.line_chart(df)
st.bar_chart(df)


st.sidebar.header("사이드바 menu")
page = st.sidebar.radio("이동할 페이지 선택", ["홈","ChatBot","설정"])
st.write(f"선택한 페이지: {page}")

st.header("파일 업로드 & 다운로드")
uploaded_file = st.file_uploader("파일을 업로드하세요.", type=["csv", "txt", "png", "jpg", "docx"])
if uploaded_file is not None:
    st.write(f"업로드한 파일명: {uploaded_file.name}")

st.download_button("샘플 파일 다운로드", data="This is a sample file.", file_name="sample.txt")


if "message" not in st.session_state:
    st.session_state.message = []

st.title("Chatbot demo")
st.write("간단한 챗봇 데모입니다. 메시지를 입력하고 전송하세요.")

for msg in st.session_state.message:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("메시지를 입력하세요.")
if user_input:
    st.session_state.message.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    bot_response = f"따라할래요: {user_input}"
    st.session_state.message.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)

