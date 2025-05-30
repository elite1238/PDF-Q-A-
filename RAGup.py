import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from textblob import TextBlob
import uuid
import hashlib
import os
import re
from transformers import AutoTokenizer, AutoModel
import torch

@st.cache_resource
def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")#distilbert-base-uncased
    return tokenizer, model

@st.cache_resource
def load_llm(model_name):
    return OllamaLLM(model=model_name)

# Correct the spelling of the question using TextBlob

def correct_spelling(text):
    return str(TextBlob(text).correct())

# preproceesing the pdf file 
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

def preprocess_text(raw_text):
    return clean_text(raw_text)

def split_into_chunks(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

class LocalEmbeddings:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)
        return embeddings

    def embed_query(self, query):
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

class QdrantVectorStore:
    def __init__(self, collection_name, embeddings_model, texts):
        self.collection_name = collection_name
        self.embeddings_model = embeddings_model
        self.texts = texts

        self.client = QdrantClient(
            url="https://237e9761-7315-4019-ac83-acba4da8a1dd.europe-west3-0.gcp.cloud.qdrant.io",
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.n00Tana6DfjP4M331-7efFJLM6iBhRWXxto4Y-xJI88")

        self._prepare_collection()
        self._upload_texts()
    def _prepare_collection(self):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

    # Compute the embedding dimension dynamically
        sample_embedding = self.embeddings_model.embed_documents(["test"])[0]
        embedding_dim = len(sample_embedding)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE
            )
        )

    def _upload_texts(self):
        embeddings = self.embeddings_model.embed_documents(self.texts)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload={"text": self.texts[i]}
            ) for i in range(len(self.texts))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def similarity_search(self, query, k=4):
        query_embedding = self.embeddings_model.embed_query(query).tolist()
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k
        )
        return [result.payload["text"] for result in results]

def create_qa_chain(llm, docs, question):
    context = "\n\n".join(docs)
    prompt = f"""Based on the following context, answer the question as accurately and precise as possible without any additional information or explanations.
Context: {context}
Question: {question}
Answer:"""
    return llm.invoke(prompt)

def get_file_hash(file):
    file.seek(0)
    file_bytes = file.read()
    file.seek(0)
    return hashlib.md5(file_bytes).hexdigest()

def main():
    st.set_page_config(page_title="PDF AI ", page_icon="🤖")
    st.header("Ask your PDF 💬")

    available_models = ["llama3.2", "phi3.5", "gemma2:2b", "qwen2.5:7b"]
    selected_model = st.selectbox("Choose LLM Model:", available_models)
    llm = load_llm(selected_model)

    pdf = st.file_uploader("Upload your PDF file", type=["pdf"])

    if "pdf_hash" not in st.session_state:
        st.session_state.pdf_hash = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    if "knowledgebase" not in st.session_state:
        st.session_state.knowledgebase = None

    if pdf:
        current_hash = get_file_hash(pdf)
        if st.session_state.pdf_hash != current_hash:
            st.session_state.pdf_hash = current_hash
            st.session_state.chunks = None
            st.session_state.knowledgebase = None
            st.success("PDF changed. Reprocessing...")

        if st.session_state.chunks is None:
            with st.spinner("Extracting and preprocessing PDF..."):
                text = ""
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text

                preprocessed = preprocess_text(text)
                st.session_state.chunks = split_into_chunks(preprocessed)
                st.success("Preprocessed and split into chunks!")

        user_question = st.text_input("Ask a question about your PDF:")
        user_question = llm.invoke ("accurately correct if there is any mistakes in the question if there is no correction return as it is \nQuestion:" +user_question) if user_question else None

        if user_question:
            try:
                if st.session_state.knowledgebase is None:
                    with st.spinner("Building knowledge base..."):
                        tokenizer, model = load_embedding_model()
                        embeddings = LocalEmbeddings(tokenizer, model)
                        st.session_state.knowledgebase = QdrantVectorStore(pdf.name, embeddings, st.session_state.chunks)

                with st.spinner("Answering your question..."):
                    docs = st.session_state.knowledgebase.similarity_search(user_question)
                    response = create_qa_chain(llm, docs, user_question)
                    st.success("Answer:")
                    st.write(response)

            except Exception as e:
                st.error(f"Error: {e}")
                st.error("Ensure Qdrant and Ollama are correctly configured.")

main()
