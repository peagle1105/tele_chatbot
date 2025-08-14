# app.py
import streamlit as st
from rag_engine import load_llm_and_qa_chain
import torch

# --- Page Config ---
st.set_page_config(
    page_title="⚕️ Trợ lý Y tế AI",
    page_icon="⚕️",
    layout="centered"
)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# --- Title ---
st.title("💬 Trợ lý Y tế Thông Minh")
st.markdown("Dựa trên mô hình fine-tune + RAG từ cơ sở dữ liệu y khoa.")

# --- Load Model on Startup ---
@st.cache_resource
def load_model():
    with st.spinner("🧠 Đang tải mô hình và cơ sở dữ liệu..."):
        return load_llm_and_qa_chain()

if st.session_state.qa_chain is None:
    try:
        st.session_state.qa_chain = load_model()
        st.success("✅ Mô hình đã sẵn sàng!")
    except Exception as e:
        st.error(f"❌ Lỗi khi tải mô hình: {e}")
        st.stop()

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("Hãy hỏi về sức khỏe của bạn..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show "typing" indicator
    with st.chat_message("assistant"):
        with st.spinner("🔍 Đang tìm kiếm và suy luận..."):
            try:
                result = st.session_state.qa_chain.invoke({"query": prompt})
                response = result["result"]
                sources = result.get("source_documents", [])
                
                # Display answer
                st.markdown(response)
                
                # Optional: Show sources
                if sources:
                    with st.expander("📄 Xem tài liệu tham khảo"):
                        for idx, doc in enumerate(sources):
                            st.markdown(f"**Nguồn {idx+1}:**\n{doc.page_content[:300]}...")
            except Exception as e:
                response = "Xin lỗi, tôi không thể xử lý câu hỏi này ngay bây giờ."
                st.markdown(response)
                st.error(f"Lỗi: {e}")

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})