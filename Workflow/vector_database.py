from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Upload & Load raw PDF(s)

pdfs_directory = 'Data/' # Save the data

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

# Extract the Text
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents


file_path = 'Data/Human_rights_law_in_India.pdf' # For testing
documents = load_pdf(file_path)
#print("PDF pages: ",len(documents))

# Create Chunks
def create_chunks(documents): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

text_chunks = create_chunks(documents)
#print("Chunks count: ", len(text_chunks))


# Setup Embeddings Model 
def get_embedding_model():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings



# Index Documents **Store embeddings in FAISS (vector store)
FAISS_DB_PATH="Vectorstore/db_faiss"
faiss_db=FAISS.from_documents(text_chunks, get_embedding_model())
faiss_db.save_local(FAISS_DB_PATH)
