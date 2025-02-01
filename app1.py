import streamlit as st
import time
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv()

# API Keys
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "2-Q&A_RAG_Chatbot"
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

# Define Prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Provide the most accurate response.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Create vector embeddings
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("research papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Streamlit UI
st.markdown("""
    <h1 style='text-align: center; color: #ff4b4b;'>📜 RAG Q&A Chatbot 🤖</h1>
    <p style='text-align: center;'>Powered by <b>Groq & Llama3</b></p>
""", unsafe_allow_html=True)

# Sidebar options
with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.5)
    top_k = st.slider("Top K Retrievals", 1, 10, 3)

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.success("✅ Vector Database is ready")

if user_prompt:
    # Ensure the vectors are created before using them
    if "vectors" not in st.session_state:
        create_vector_embedding()

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": top_k})  # Use top_k here
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start_time = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt, 'temperature': temperature})  # Pass temperature here
    end_time = time.process_time()
    
    response_time = round(end_time - start_time, 2)
    
    # AI Typing Effect
    placeholder = st.empty()
    for i in range(len(response['answer'])):
        placeholder.write(response['answer'][:i+1])
        time.sleep(0.02)
    
    st.write(f"⏳ Response Time: {response_time} sec")
    
    # Document similarity search
    with st.expander("🔍 Document similarity Search"):
        for doc in response['context']:
            st.markdown(f"""
                <div style="background:#1e1e1e;padding:10px;border-radius:10px;margin:10px;color:white;">
                    {doc.page_content[:300]}...
                </div>
            """, unsafe_allow_html=True)
