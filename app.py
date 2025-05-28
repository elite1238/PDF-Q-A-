import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class LocalEmbeddings:
    def __init__(self):
        # Free local embedding model
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')#all-MiniLM-L6-v2
    def embed_documents(self, texts):
        return self.model.encode(texts)
    def embed_query(self, query):
        return self.model.encode([query])[0]

class LocalVectorStore:
    def __init__(self, texts, embeddings_model):
        self.texts = texts
        self.embeddings_model = embeddings_model
        # Create embeddings for all texts
        self.embeddings = self.embeddings_model.embed_documents(texts)
    def similarity_search(self, query, k=4):
        # Get query embedding
        query_embedding = self.embeddings_model.embed_query(query)
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        # Get top k most similar documents
        top_indices = np.argsort(similarities)[-k:][::-1]
        # Return the most relevant texts
        return [self.texts[i] for i in top_indices]

def create_qa_chain(llm, docs, question):
    """Simple QA function without langchain chains"""
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
    # Model selection
    available_models = ["llama3.2", "phi3.5", "gemma2:2b", "qwen2.5:7b"]
    selected_model = st.selectbox("Choose LLM Model:", available_models)
    pdf = st.file_uploader("Upload your PDF file", type=["pdf"])
    if pdf is not None:
        with st.spinner("Processing PDF..."):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() 
            text_splitter = CharacterTextSplitter(
                separator="\n", 
                chunk_size=1000, 
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            embeddings = LocalEmbeddings()
            knowledge_base = LocalVectorStore(chunks, embeddings)
        st.success(f"PDF processed! Split into {len(chunks)} chunks.") 
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            with st.spinner("Thinking..."):
                try:
                    docs = knowledge_base.similarity_search(user_question)    
                    llm = OllamaLLM(model=selected_model)  
                    response = create_qa_chain(llm, docs, user_question) 
                    st.write("**Answer:**")
                    st.write(response) 
                    with st.expander("View source chunks"):
                        for i, doc in enumerate(docs):
                            st.write(f"**Chunk {i+1}:**")
                            st.write(doc[:200] + "..." if len(doc) > 200 else doc)
                            st.write("---")      
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.error("Make sure Ollama is running and the model is installed!")
main()