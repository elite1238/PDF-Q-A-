import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class LocalEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_documents(self, texts):
        return self.model.encode(texts)
    
    def embed_query(self, query):
        return self.model.encode([query])[0]

class LocalVectorStore:
    def __init__(self, texts, embeddings_model):
        self.texts = texts
        self.embeddings_model = embeddings_model
        self.embeddings = self.embeddings_model.embed_documents(texts)
    
    def similarity_search(self, query, k=4):
        query_embedding = self.embeddings_model.embed_query(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.texts[i] for i in top_indices]

def create_qa_chain(llm, docs, question):
    context = "\n\n".join(docs)
    prompt = f"""Based on the following context, answer the question as accurately as possible.

Context:
{context}

Question: {question}

Answer:"""
    return llm.invoke(prompt)

def main():
    st.set_page_config(page_title="PDF AI (Local & Free)", page_icon="🤖")
    st.header("Ask your PDF 💬 (100% Free & Local)")
    
    available_models = ["llama3.2", "phi3.5", "gemma2:2b", "qwen2.5:7b"]
    selected_model = st.selectbox("Choose LLM Model:", available_models)
    
    # Use session state to manage cache clearing and data persistence
    if "pdf_hash" not in st.session_state:
        st.session_state.pdf_hash = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = None

    pdf = st.file_uploader("Upload your PDF file", type=["pdf"])

    import hashlib

    def get_file_hash(file):
        file.seek(0)
        file_bytes = file.read()
        file.seek(0)
        return hashlib.md5(file_bytes).hexdigest()

    # If new PDF is uploaded, clear cache and reload
    if pdf is not None:
        pdf_hash = get_file_hash(pdf)
        if st.session_state.pdf_hash != pdf_hash:
            # New PDF uploaded, clear cache and process
            st.session_state.pdf_hash = pdf_hash
            st.session_state.embeddings = None
            st.session_state.knowledge_base = None
            st.session_state.chunks = None

        if st.session_state.knowledge_base is None or st.session_state.chunks is None:
            with st.spinner("Processing PDF..."):
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
            
                text_splitter = CharacterTextSplitter(
                    separator="\n", 
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)
                st.session_state.chunks = chunks

                embeddings = LocalEmbeddings()
                knowledge_base = LocalVectorStore(chunks, embeddings)
                st.session_state.embeddings = embeddings
                st.session_state.knowledge_base = knowledge_base

            st.success(f"PDF processed! Split into {len(st.session_state.chunks)} chunks.")

        user_question = st.text_input("Ask a question about your PDF:")

        if user_question:
            with st.spinner("Thinking..."):
                try:
                    docs = st.session_state.knowledge_base.similarity_search(user_question)
                    llm = OllamaLLM(model=selected_model)
                    response = create_qa_chain(llm, docs, user_question)
                    st.write("Answer:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.error("Make sure Ollama is running and the model is installed!")

    else:
        # If no PDF uploaded, clear cache
        st.session_state.pdf_hash = None
        st.session_state.embeddings = None
        st.session_state.knowledge_base = None
        st.session_state.chunks = None

main()
