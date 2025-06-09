import streamlit as st
from PyPDF2 import PdfReader
import re
import uuid
import hashlib
import os
from langchain_groq import ChatGroq
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import logging
import time

# Configure logging
logging.basicConfig(level=logging.WARNING)

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "Mongo_uri real")
DB_NAME = "new_api"
COLLECTION_NAME = "qa_pairs"

# Groq API Key
GROQ_API_KEY = "your_groq_api_key"

# ==================== Database Functions ====================
@st.cache_resource
def get_mongo_collection():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=22000)  
        client.admin.command('ping') 
        db = client[DB_NAME]
        return db[COLLECTION_NAME]
    except Exception as e:
        st.warning(f"MongoDB connection failed: {e}. Q&A storage/retrieval will be unavailable.")
        return None

@st.cache_resource
def get_search_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

def store_qa_pair(question, answer):
    try:
        collection = get_mongo_collection()
        if collection is None:
            st.warning("MongoDB collection not available. Cannot store Q&A pair.")
            return
        model = get_search_model()
        embedding = model.encode(question).tolist()
        doc = {
            "question": question,
            "answer": answer,
            "embedding": embedding,
            "timestamp": time.time()
        }
        collection.insert_one(doc)
    except Exception as e:
        st.warning(f"Could not store Q&A: {e}")

def find_similar_questions(query, top_k=3):
    try:
        collection = get_mongo_collection()
        if collection is None:
            st.warning("MongoDB collection not available. Cannot find similar questions.")
            return []
        model = get_search_model()
        query_emb = model.encode(query)
        docs = list(collection.find({}, {"question": 1, "embedding": 1, "answer": 1}).sort("timestamp", -1).limit(500))
        if not docs:
            return []
        emb_matrix = np.array([doc["embedding"] for doc in docs if "embedding" in doc and doc["embedding"] is not None])
        if emb_matrix.ndim == 1:
            if emb_matrix.shape[0] == query_emb.shape[0]:
                emb_matrix = emb_matrix.reshape(1, -1)
            else:
                return []
        elif emb_matrix.shape[0] == 0:
            return []
        emb_matrix = emb_matrix.astype(np.float32)
        query_emb = query_emb.astype(np.float32)
        sims = util.pytorch_cos_sim(query_emb, emb_matrix)[0].cpu().numpy()
        actual_top_k = min(top_k, len(sims))
        if actual_top_k == 0:
            return []
        top_indices = sims.argsort()[-actual_top_k:][::-1]
        return [docs[i] for i in top_indices if sims[i] > 0.7]
    except Exception as e:
        st.warning(f"Could not search MongoDB for similar questions: {e}")
        return []

# ==================== Model Loading ====================
@st.cache_resource
def load_embedding_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # st.info(f"Embedding model using device: {device}")  # IMPROVE: avoid info in cache functions
    return SentenceTransformer('all-MiniLM-L6-v2', device=device)

@st.cache_resource
def load_question_gen_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("iarfmoose/t5-base-question-generator")
        model = AutoModelForSeq2SeqLM.from_pretrained("iarfmoose/t5-base-question-generator")
        return tokenizer, model
    except Exception as e:
        return None, None

@st.cache_resource
def load_llm(model_name: str, api_key: str):
    if not api_key:
        st.error("Groq API key is not set. Please set the GROQ_API_KEY environment variable.")
        return None
    try:
        return ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=0.2)
    except Exception as e:
        st.error(f"Failed to load Groq LLM ({model_name}): {e}")
        return None

# ==================== Text Processing ====================
def extract_pdf_text(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return ""

def clean_and_preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def split_into_optimized_chunks(text, chunk_size=200, overlap=50, strategy="recursive"):
    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " "],
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = splitter.split_text(text)
    elif strategy == "sentence":
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += (" " + sentence) if current_chunk else sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
    elif strategy == "paragraph":
        paragraphs = text.split('\n\n')
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) <= chunk_size:
                chunks.append(para)
            else:
                splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
                chunks.extend(splitter.split_text(para))
    else:
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len
        )
        chunks = splitter.split_text(text)
    return [chunk for chunk in chunks if len(chunk.strip()) > 30]

def create_smart_chunks(text, min_chunk_size=200, max_chunk_size=400, overlap_ratio=0.15):
    sections = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk_text = ""
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(current_chunk_text) + len(section) + 2 <= max_chunk_size:
            current_chunk_text += ("\n\n" + section) if current_chunk_text else section
        else:
            if current_chunk_text:
                if len(current_chunk_text) >= min_chunk_size:
                    chunks.append(current_chunk_text)
                overlap_len = int(len(current_chunk_text) * overlap_ratio)
                overlap = current_chunk_text[-overlap_len:] if chunks and overlap_len > 0 else ""
                current_chunk_text = overlap + ("\n\n" + section if overlap else section)
            else:
                current_chunk_text = section
            while len(current_chunk_text) > max_chunk_size:
                part_to_add = current_chunk_text[:max_chunk_size]
                last_sentence_end = max(part_to_add.rfind("."), part_to_add.rfind("!"), part_to_add.rfind("?"))
                if last_sentence_end > min_chunk_size:
                    part_to_add = current_chunk_text[:last_sentence_end+1]
                if len(part_to_add) >= min_chunk_size:
                    chunks.append(part_to_add.strip())
                remaining_text = current_chunk_text[len(part_to_add):].strip()
                overlap_len = int(len(part_to_add) * overlap_ratio)
                overlap = part_to_add[-overlap_len:] if chunks and overlap_len > 0 else ""
                current_chunk_text = (overlap + "\n\n" + remaining_text if overlap else remaining_text).strip()
                if not remaining_text:
                    break
    if current_chunk_text and len(current_chunk_text.strip()) >= min_chunk_size:
        chunks.append(current_chunk_text.strip())
    return [chunk for chunk in chunks if len(chunk.strip()) > 30]

def generate_question_from_passage(passage, tokenizer, model):
    if tokenizer is None or model is None:
        st.info("Question generation model not loaded. Using simple fallback.")
        return generate_simple_question_from_passage(passage)
    try:
        context = passage[:512]
        input_text = f"generate question: {context}"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                early_stopping=True,
            )
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        question = question.replace("generate question:", "").strip()
        if not question or len(question) < 10:
            return generate_simple_question_from_passage(passage)
        if not question.endswith('?'):
            question += '?'
        return question
    except Exception as e:
        st.warning(f"T5 question generation failed: {e}. Using simple fallback.")
        return generate_simple_question_from_passage(passage)

def generate_simple_question_from_passage(passage):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', passage) if len(s.strip()) > 10]
    if not sentences:
        return "What is the main topic discussed in this passage?"
    first_sentence_topic = extract_key_topic(sentences[0])
    if first_sentence_topic == "this topic" or len(first_sentence_topic.split()) > 3:
        return f"What is the main idea of the following text: \"{sentences[0][:100]}...\"?"
    return f"What does the passage say about {first_sentence_topic}?"

def extract_key_topic(sentence):
    try:
        blob = TextBlob(sentence)
        if blob.noun_phrases:
            return blob.noun_phrases[0]
        nouns = [word for word, pos in blob.tags if pos.startswith('NN')]
        return nouns[0] if nouns else "this topic"
    except Exception:
        words = sentence.split()
        cap_words = [w for w in words if w.istitle() and len(w) > 3]
        if cap_words: return cap_words[0]
        if words: return max(words, key=len)
        return "this topic"

class OptimizedVectorStore:
    def __init__(self, embeddings_model, texts: list[str]):
        self.embeddings_model = embeddings_model
        self.texts = texts
        self.embeddings = None
        if self.texts:
            self._create_embeddings()
    def _create_embeddings(self):
        # st.info(f"Creating embeddings for {len(self.texts)} chunks...")  # IMPROVE: avoid info in non-UI code
        self.embeddings = self.embeddings_model.encode(self.texts, convert_to_tensor=True, show_progress_bar=True)
    def similarity_search_top_k(self, query: str, k: int = 5) -> list[dict]:
        if self.embeddings is None or len(self.texts) == 0:
            return []
        query_embedding = self.embeddings_model.encode(query, convert_to_tensor=True)
        if query_embedding.device != self.embeddings.device:
            query_embedding = query_embedding.to(self.embeddings.device)
        scores = util.cos_sim(query_embedding, self.embeddings)[0]
        actual_k = min(k, len(scores))
        if actual_k == 0:
            return []
        top_k_results = torch.topk(scores, actual_k)
        results = []
        for score, idx in zip(top_k_results.values, top_k_results.indices):
            results.append({
                'text': self.texts[idx.item()],
                'score': score.item(),
                'index': idx.item()
            })
        return results

def create_optimized_qa_chain(llm, doc_results: list[dict], question: str):
    if not llm:
        return "LLM not loaded. Cannot generate answer."
    if not doc_results:
        return "No relevant document passages found to answer the question."
    sorted_results = sorted(doc_results, key=lambda x: x.get('score', 0), reverse=True)
    context_chunks = [chunk['text'] for chunk in sorted_results[:st.session_state.get('top_k_results_for_llm', 3)]]
    context = "\n\n---\n\n".join(context_chunks)
    MAX_CONTEXT_CHARS = 15000
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n...(context truncated)"
    prompt = f"""You are a helpful AI assistant. Answer the question based *only* on the context provided below.
Be concise and accurate. If the answer is not found in the context, say "The answer is not found in the provided document context."

Context:
---
{context}
---

Question: {question}

Answer:"""
    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        st.error(f"Error generating answer with LLM: {e}")
        return f"Error generating answer: {e}"

def get_file_hash(file):
    file.seek(0)
    file_bytes = file.read()
    file.seek(0)
    return hashlib.md5(file_bytes).hexdigest()

# ==================== Main Streamlit App ====================
def main():
    st.set_page_config(
        page_title="Integrated PDF AI Assistant",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 PDF AI Assistant with Groq")
    st.markdown("Upload a PDF, ask questions, generate questions, or search previous Q&A. Powered by Groq!")

    if not GROQ_API_KEY:
        st.error("🔴 GROQ_API_KEY environment variable not set. The application will not be able to use Groq LLMs. Please set it and restart.")
    else:
        st.success("🟢 Groq API Key found.")

    with st.sidebar:
        st.header("⚙️ Configuration")
        available_models = ["llama3-8b-8192"]
        selected_model = st.selectbox("Choose Groq LLM Model:", available_models)
        st.subheader("Chunking Settings")
        chunk_strategy = st.selectbox(
            "Chunking Strategy:",
            ["smart", "recursive", "sentence", "paragraph"],
            index=0,
            help="Smart: Adaptive paragraph/sentence. Recursive: Hierarchical. Sentence/Paragraph: Basic splits."
        )
        if chunk_strategy == "smart":
            min_chunk_size = st.slider("Min Smart Chunk Size (chars)", 150, 500, 250, 10)
            max_chunk_size = st.slider("Max Smart Chunk Size (chars)", 300, 1000, 500, 10)
        else:
            chunk_size = st.slider("Chunk Size (chars)", 100, 1000, 400, 10)
            chunk_overlap = st.slider("Chunk Overlap (chars)", 0, 200, 50, 5)
        max_chunks_to_process = st.slider("Max Chunks to Process from PDF", 10, 200, 100, 10)
        st.subheader("Retrieval & Answering Settings")
        top_k_results_for_retrieval = st.slider("Top-K Chunks for Display", 1, 10, 3, 1)
        st.session_state.top_k_results_for_llm = st.slider("Top-K Chunks for LLM Context", 1, 5, 3, 1)

    llm = load_llm(selected_model, GROQ_API_KEY)

    tab1, tab2 = st.tabs(["📄 PDF Q&A", "❓ Question Generation"])

    pdf_file = st.file_uploader("Upload your PDF file", type=["pdf"], key="pdf_uploader")

    if "pdf_hash" not in st.session_state: st.session_state.pdf_hash = None
    if "pdf_text_chunks" not in st.session_state: st.session_state.pdf_text_chunks = None
    if "vector_knowledgebase" not in st.session_state: st.session_state.vector_knowledgebase = None

    if pdf_file:
        current_pdf_hash = get_file_hash(pdf_file)
        if st.session_state.pdf_hash != current_pdf_hash:
            st.session_state.pdf_hash = current_pdf_hash
            with st.spinner("🔄 Processing PDF... This may take a moment."):
                raw_text = extract_pdf_text(pdf_file)
                if raw_text:
                    processed_text = clean_and_preprocess_text(raw_text)
                    st.info(f"Selected chunking strategy: {chunk_strategy}")
                    if chunk_strategy == "smart":
                        st.session_state.pdf_text_chunks = create_smart_chunks(
                            processed_text,
                            min_chunk_size=min_chunk_size,
                            max_chunk_size=max_chunk_size
                        )[:max_chunks_to_process]
                    else:
                        st.session_state.pdf_text_chunks = split_into_optimized_chunks(
                            processed_text,
                            chunk_size=chunk_size,
                            overlap=chunk_overlap,
                            strategy=chunk_strategy
                        )[:max_chunks_to_process]
                    if st.session_state.pdf_text_chunks:
                        embedding_model_instance = load_embedding_model()
                        st.session_state.vector_knowledgebase = OptimizedVectorStore(
                            embedding_model_instance,
                            st.session_state.pdf_text_chunks
                        )
                        st.success(f"✅ PDF processed! Created {len(st.session_state.pdf_text_chunks)} chunks for Q&A.")
                        if st.session_state.pdf_text_chunks:
                            avg_chunk_len = sum(len(c) for c in st.session_state.pdf_text_chunks) / len(st.session_state.pdf_text_chunks)
                            st.info(f"Average chunk length: {avg_chunk_len:.0f} characters.")
                    else:
                        st.error("❌ No chunks were created from the PDF. It might be empty or unreadable.")
                        st.session_state.vector_knowledgebase = None
                else:
                    st.error("❌ Could not extract text from PDF.")
                    st.session_state.pdf_text_chunks = None
                    st.session_state.vector_knowledgebase = None

    with tab1:
        st.header("💬 Ask Questions About Your PDF")
        if pdf_file and st.session_state.vector_knowledgebase:
            user_question = st.text_input("Enter your question:", key="qa_question")
            if user_question:
                with st.spinner("🔍 Thinking..."):
                    similar_qa_pairs = find_similar_questions(user_question, top_k=3)
                    found_similar = False
                    sim_threshold = 0.7
                    for qa_pair in similar_qa_pairs:
                        if qa_pair.get("question") and qa_pair.get("answer"):
                            found_similar = True
                            st.success("✅ Found a similar question in the database!")
                            st.markdown(f"**Question:** {qa_pair['question']}")
                            st.markdown(f"**Stored Answer:** {qa_pair['answer']}")
                            break
                    if not found_similar:
                        doc_results = st.session_state.vector_knowledgebase.similarity_search_top_k(
                            user_question, k=top_k_results_for_retrieval
                        )
                        if doc_results:
                            response = create_optimized_qa_chain(llm, doc_results, user_question)
                            st.markdown("**Answer:**")
                            st.info(response)
                            store_qa_pair(user_question, response)
                            with st.expander(f"Relevant Passages (Top {len(doc_results)} used for context)"):
                                for i, result in enumerate(doc_results, 1):
                                    st.markdown(f"**Passage {i} (Score: {result['score']:.3f})**")
                                    st.caption(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])
                                    st.markdown("---")
                        else:
                            st.warning("Could not find relevant passages for your question in the PDF.")
        elif pdf_file and not st.session_state.vector_knowledgebase:
            st.warning("PDF processed, but knowledge base (vectors) not created. Please check PDF content or processing steps.")
        else:
            st.info("📄 Please upload a PDF to enable Q&A.")

    with tab2:
        st.header("❓ Generate Questions from PDF Content")
        if pdf_file and st.session_state.pdf_text_chunks:
            qg_topic = st.text_input("Enter a topic/keywords to find passages for question generation:", key="qg_topic")
            num_q_to_gen = st.number_input("Number of questions to generate per passage:", 1, 3, 1, key="num_q_gen")
            if qg_topic:
                with st.spinner("🧬 Generating questions..."):
                    relevant_passages_for_qg = st.session_state.vector_knowledgebase.similarity_search_top_k(
                        qg_topic, k=num_q_to_gen
                    )
                    if relevant_passages_for_qg:
                        qg_tokenizer, qg_model_instance = load_question_gen_model()
                        st.markdown("**Generated Questions & Passages:**")
                        for i, passage_data in enumerate(relevant_passages_for_qg):
                            st.markdown(f"**Source Passage {i+1} (Relevance: {passage_data['score']:.2f})**")
                            st.caption(passage_data['text'][:300] + "...")
                            generated_q = generate_question_from_passage(passage_data['text'], qg_tokenizer, qg_model_instance)
                            st.info(f"**Generated Question:** {generated_q}")
                            if st.button(f"Answer this question ({i+1})", key=f"answer_gen_q_{i}"):
                                with st.spinner("Answering generated question..."):
                                    answer_docs = st.session_state.vector_knowledgebase.similarity_search_top_k(generated_q, k=top_k_results_for_retrieval)
                                    if answer_docs:
                                        gen_q_answer = create_optimized_qa_chain(llm, answer_docs, generated_q)
                                        st.markdown(f"**Answer to '{generated_q}':**")
                                        st.success(gen_q_answer)
                                        store_qa_pair(generated_q, gen_q_answer)
                                    else:
                                        st.warning("Could not find relevant context for this generated question.")
                            st.markdown("---")
                    else:
                        st.warning(f"No relevant passages found for topic: '{qg_topic}'")
        else:
            st.info("📄 Please upload and process a PDF to generate questions from its content.")

    st.markdown("---")
    st.caption("💡 Tips: Ensure your GROQ_API_KEY is set. For best results, use specific questions. Experiment with chunking strategies and LLM models.")
    if st.session_state.get('pdf_text_chunks'):
        st.caption(f"ℹ️ Current PDF: {len(st.session_state.pdf_text_chunks)} chunks loaded.")

if __name__ == "__main__":
    main()
