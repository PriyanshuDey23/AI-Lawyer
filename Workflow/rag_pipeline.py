from langchain_groq import ChatGroq
# from vector_database import faiss_db 
from Workflow.vector_database import faiss_db # from vector store file
from langchain_core.prompts import ChatPromptTemplate
import os

# Uncomment the following if you're NOT using pipenv
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY=os.getenv("GROQ_API_KEY")

# Setup LLM (Use DeepSeek R1 with Groq)
llm_model=ChatGroq(model="deepseek-r1-distill-llama-70b",api_key=GROQ_API_KEY)

# Retrieve Docs
def retrieve_docs(query):
    return faiss_db.similarity_search(query) # Similarity search

# get only the content part
def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context


# Answer Question
custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context
Question: {question} 
Context: {context} 
Answer:
"""

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model # | -> Pipe function, pass the prompt in which model
    return chain.invoke({"question": query, "context": context})

# Testing
# question="If a government forbids the right to assemble peacefully which articles are violated and why?"
# retrieved_docs=retrieve_docs(question)
# print("AI Lawyer: ",answer_query(documents=retrieved_docs, model=llm_model, query=question))
