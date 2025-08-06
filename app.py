import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
from datetime import datetime

# Handle async operations
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    st.error("Please install nest_asyncio: pip install nest_asyncio")
    st.stop()

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Custom CSS for chat interface
CHAT_CSS = """
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.bot {
        background-color: #475063;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
        color: #fff;
    }
</style>
"""

def get_pdf_text(pdf_docs):
    """Extract text from PDF files"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text, model_name):
    """Split text into chunks"""
    if model_name == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=1000
        )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, model_name, api_key):
    """Create and save vector store"""
    if not api_key or not api_key.startswith('AIza'):
        st.error("❌ Invalid Google API Key")
        return None
        
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
            transport="rest"  # Force synchronous mode
        )
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"🔴 Embedding Error: {str(e)}")
        return None

def get_conversational_chain(model_name, api_key):
    """Create QA chain"""
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer isn't in the context, say "answer is not available in the context".
    
    Context:\n{context}\n
    Question: \n{question}\n

    Answer:
    """
    
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=api_key,
        transport="rest"
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def display_conversation():
    """Display chat history"""
    st.markdown(CHAT_CSS, unsafe_allow_html=True)
    
    for msg in st.session_state.conversation_history:
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>
                <div class="message">{msg['question']}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
                </div>
                <div class="message">{msg['answer']}</div>
            </div>
            <div class="info">
                {msg['timestamp']} | PDF: {msg['pdf_names']}
            </div>
            """,
            unsafe_allow_html=True
        )

def process_query(user_question, model_name, api_key, pdf_docs):
    """Process user question and generate response"""
    if not api_key or not pdf_docs:
        st.warning("Please upload PDFs and provide API key")
        return
        
    with st.spinner("Processing your question..."):
        try:
            # Process PDFs and create embeddings
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text, model_name)
            
            if not get_vector_store(text_chunks, model_name, api_key):
                return
                
            # Generate response
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key,
                transport="rest"
            )
            db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = db.similarity_search(user_question)
            
            chain = get_conversational_chain(model_name, api_key)
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            
            # Store conversation
            st.session_state.conversation_history.append({
                "question": user_question,
                "answer": response['output_text'],
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "pdf_names": ", ".join([pdf.name for pdf in pdf_docs])
            })
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

def main():
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    st.title("📚 Chat with Your PDFs")
    st.caption("Upload PDFs and ask questions about their content")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # API Key Input
        api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Get your key from https://aistudio.google.com/apikey"
        )
        
        # PDF Upload
        pdf_docs = st.file_uploader(
            "Upload PDF Files",
            type="pdf",
            accept_multiple_files=True
        )
        
        # Conversation controls
        if st.button("🧹 Clear Conversation"):
            st.session_state.conversation_history = []
            st.rerun()
            
        if st.session_state.conversation_history:
            if st.button("💾 Export Chat History"):
                df = pd.DataFrame(st.session_state.conversation_history)
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="chat_history.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    # Main chat interface
    user_question = st.chat_input("Ask a question about your PDFs...")
    
    if user_question:
        if not api_key or not api_key.startswith('AIza'):
            st.error("Please enter a valid Google API Key")
        elif not pdf_docs:
            st.error("Please upload PDF files first")
        else:
            process_query(user_question, "Google AI", api_key, pdf_docs)
            st.rerun()
    
    # Display conversation
    if st.session_state.conversation_history:
        display_conversation()

if __name__ == "__main__":
    main()