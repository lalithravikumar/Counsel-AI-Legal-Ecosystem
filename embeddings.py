import os
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def get_pdf_texts_from_folder(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_reader = PdfReader(os.path.join(folder_path, filename))
            for page in pdf_reader.pages:
                texts.append(page.extract_text())
    return texts

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=300,
        length_function=len
    )
    return text_splitter.split_text(text)

def save_embeddings(folder_path, output_file):
    texts = get_pdf_texts_from_folder(folder_path)
    all_chunks = []
    for text in texts:
        chunks = get_text_chunks(text)
        all_chunks.extend(chunks)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_texts(texts=all_chunks, embedding=embeddings)

    # Save the vectorstore to a file
    with open(output_file, 'wb') as f:
        pickle.dump(vectorstore, f)

folder_path = r"C:\Users\bjlal\Documents\KMS J\LLMData"
output_file = "LawEmbeddings.pkl"
save_embeddings(folder_path, output_file)
