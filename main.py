from unsloth import FastLanguageModel
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
import torch
from transformers.pipelines import pipeline
import os

# --- Cấu hình ---
model_name = "./models/fine_tuned_model"
vector_db_path = "./agents/vector_db_store/rag_vector_db"
embedding_model_path = "/mnt/d/HOPT/Agent_build/Telemed_agent/models/all-MiniLM-L6-v2-f16.gguf"
max_seq_length = 2048
load_in_4bit = True

# --- Bước 1: Load LLM đã fine-tune từ Unsloth ---
def load_llm():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,
        device_map="auto",
    )

    # Áp dụng inference
    FastLanguageModel.for_inference(model)

    # Tạo pipeline với Hugging Face
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    # Wrap thành LLM cho LangChain
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm, tokenizer

# --- Bước 2: Tạo hoặc đọc Vector DB ---
from langchain_huggingface import HuggingFaceEmbeddings

def read_vector_db():
    # ✅ Dùng HuggingFace thay vì GPT4All
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    db = FAISS.load_local(
        vector_db_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return db

# --- Bước 3: Tạo prompt cho RAG ---
def create_prompt():
    template = """
Bạn là một trợ lý y tế thông minh, có nhiệm vụ trả lời câu hỏi dựa trên tài liệu được cung cấp.
Hãy làm theo các bước sau:
1. Đọc kỹ thông tin trong phần "Context".
2. Xác định các điểm chính liên quan đến câu hỏi.
3. Tổng hợp và suy luận để đưa ra câu trả lời đầy đủ, rõ ràng.
4. Nếu không chắc chắn, hãy nói "Tôi không biết dựa trên tài liệu đã cho." – Đừng bịa.

Context:
{context}

Câu hỏi:
{question}

Hãy suy nghĩ và trả lời:
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

# --- Bước 4: Tạo chuỗi RAG ---
def create_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain

# --- Bước 5: Chat tương tác ---
def chat():
    print("🎯 Đang tải mô hình và cơ sở dữ liệu...")
    llm, tokenizer = load_llm()
    db = read_vector_db()
    prompt = create_prompt()
    qa_chain = create_qa_chain(llm, prompt, db)

    print("✅ Sẵn sàng! Nhập 'exit' để thoát.\n")

    while True:
        question = input("Bạn hỏi: ").strip()
        if question.lower() == "exit":
            print("Tạm biệt!")
            break
        if not question:
            continue
        # Gọi RAG chain
        try:
            result = qa_chain.invoke({"query": question})
            print(f"Trả lời: {result['result']}\n")
        except Exception as e:
            print(f"Lỗi khi xử lý câu hỏi: {e}")

if __name__ == "__main__":
    chat()