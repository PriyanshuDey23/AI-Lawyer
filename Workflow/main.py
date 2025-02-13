import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from .rag_pipeline import custom_prompt_template

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY=os.getenv("GROQ_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
FAISS_DB_PATH="Vectorstore/db_faiss"


pdfs_directory = 'Data/'
llm_model=ChatGroq(model="deepseek-r1-distill-llama-70b")

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents


def create_chunks(documents): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


def get_embedding_model():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings


def create_vector_store(db_faiss_path, text_chunks):
    faiss_db=FAISS.from_documents(text_chunks, get_embedding_model())
    faiss_db.save_local(db_faiss_path)
    return faiss_db


def retrieve_docs(faiss_db, query):
    return faiss_db.similarity_search(query)


def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context


def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})


uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)


user_query = st.text_area("Enter your prompt: ", height=150 , placeholder= "Ask Anything!")

ask_question = st.button("Ask AI Lawyer")

if ask_question:

    if uploaded_file and user_query:
        upload_pdf(uploaded_file)
        documents = load_pdf(pdfs_directory + uploaded_file.name)
        text_chunks = create_chunks(documents)
        faiss_db = create_vector_store(FAISS_DB_PATH, text_chunks)

        retrieved_docs=retrieve_docs(faiss_db, user_query)
        response=answer_query(documents=retrieved_docs, model=llm_model, query=user_query)

        st.chat_message("user").write(user_query)
        st.chat_message("AI Lawyer").write(response)

    else:
        st.error("Kindly upload a valid PDF file and/or ask a valid Question!")
