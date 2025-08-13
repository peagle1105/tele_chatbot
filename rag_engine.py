# rag_engine.py

from unsloth import FastLanguageModel
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
from transformers.pipelines import pipeline

# --- Config ---
model_name = "./models/fine_tuned_model"
vector_db_path = "./agents/vector_db_store/rag_vector_db"
max_seq_length = 2048
load_in_4bit = True

def load_llm_and_qa_chain():
    """
    Loads the fine-tuned LLM, vector DB, and creates the RAG QA chain.
    Returns: qa_chain
    """
    # --- Load LLM ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,
        device_map="auto",
    )
    FastLanguageModel.for_inference(model)

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
    llm = HuggingFacePipeline(pipeline=pipe)

    # --- Load Embeddings & Vector DB ---
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    db = FAISS.load_local(
        vector_db_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # --- Prompt Template ---
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
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # --- Create QA Chain ---
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return qa_chain