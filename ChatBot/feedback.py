import streamlit as st

def display_messages():
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                feedback_options = ["", "good", "bad"]
                feedback_default = st.session_state.feedback_values.get(idx, "")
                feedback = st.radio(
                    "이 답변은 만족스러웠나요?",
                    feedback_options,
                    key=f"feedback_{idx}",
                    index=feedback_options.index(feedback_default)
                )
                st.session_state.feedback_values[idx] = feedback

                # 피드백 저장
                if feedback in ("good", "bad"):
                    if "feedbacks" not in st.session_state:
                        st.session_state.feedbacks = []
                    # 중복 방지
                    exists = False
                    for f in st.session_state.feedbacks:
                        if f.get("index") == idx:
                            f["feedback"] = feedback
                            exists = True
                            break
                    if not exists:
                        st.session_state.feedbacks.append({
                            "index": idx,
                            "question": msg.get("question", ""),
                            "answer": msg["content"],
                            "feedback": feedback
                        })

def save_feedback(index, answer, selected):
    if "feedbacks" not in st.session_state:
        st.session_state.feedbacks = []
    st.session_state.feedbacks.append({
        "index": index,
        "answer": answer,
        "feedback": selected
    })
