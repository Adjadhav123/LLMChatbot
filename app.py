import os
import uuid
import time
import platform
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
load_dotenv()

# Set up embeddings
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
except Exception as e:
    logger.error(f"Failed to initialize Ollama embeddings: {e}")
    raise ValueError("Could not initialize embeddings. Ensure Ollama is running with nomic-embed-text model.")

# In-memory store for chat history
store = {}

# Ensure uploads directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Groq API
groq_api_key = "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"
if not groq_api_key:
    logger.error("GROQ_API_KEY not found in environment variables")
    raise ValueError("GROQ_API_KEY not found in environment variables")

try:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
except Exception as e:
    logger.error(f"Failed to initialize Groq LLM: {e}")
    raise ValueError("Could not initialize Groq LLM. Check API key and network.")

# Global variable for RAG chain
conversational_rag_chain = None

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

@app.route('/')
def index():
    logger.info("Serving index.html")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    global conversational_rag_chain
    logger.info("Received upload request")
    start_time = time.time()

    if 'files' not in request.files:
        logger.warning("No files uploaded in request")
        return jsonify({"error": "No files uploaded"}), 400
    
    files = request.files.getlist('files')
    documents = []
    file_paths = []
    
    for file in files:
        if file and file.filename.endswith('.pdf'):
            filename = str(uuid.uuid4()) + '.pdf'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)
            
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs[:100])  # Limit to 100 pages to avoid overload
            except Exception as e:
                logger.error(f"Failed to process PDF {file.filename}: {e}")
                return jsonify({"error": f"Failed to process {file.filename}"}), 400
    
    if not documents:
        logger.warning("No valid PDF files uploaded")
        return jsonify({"error": "No valid PDF files uploaded"}), 400
    
    try:
        # Split documents
        logger.info("Splitting documents")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)
        
        # Create FAISS vector store
        logger.info("Creating FAISS vector store")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        
        # Contextualize question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        
        # Create history-aware retriever
        logger.info("Creating history-aware retriever")
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # Question-answering prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # Create RAG chain
        logger.info("Creating RAG chain")
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Create conversational RAG chain
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        logger.info("Conversational RAG chain initialized successfully")
        
        # Clean up uploaded files
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to delete file {file_path}: {e}")
        
        processing_time = time.time() - start_time
        logger.info(f"PDF(s) processed successfully in {processing_time:.2f} seconds")
        return jsonify({"message": "PDF(s) processed successfully"})
    
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global conversational_rag_chain
    logger.info("Received chat request")
    data = request.json
    user_input = data.get('input')
    session_id = data.get('session_id', 'default_session')
    
    if not user_input:
        logger.warning("No input provided in chat request")
        return jsonify({"error": "No input provided"}), 400
    
    if not conversational_rag_chain:
        logger.warning("Chat attempted before PDF upload")
        return jsonify({"error": "Please upload a PDF first"}), 400
    
    try:
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        
        session_history = get_session_history(session_id)
        messages = [
            {"type": msg.type, "content": msg.content}
            for msg in session_history.messages
        ]
        
        logger.info(f"Chat response generated for session {session_id}")
        return jsonify({
            "answer": response["answer"],
            "history": messages
        })
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    logger.info("Received clear history request")
    data = request.json
    session_id = data.get('session_id', 'default_session')
    try:
        if session_id in store:
            store[session_id] = ChatMessageHistory()
        logger.info(f"Chat history cleared for session {session_id}")
        return jsonify({"message": "Chat history cleared"})
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Disable reloader on Windows to avoid socket error
    use_reloader = False if platform.system() == "Windows" else True
    app.run(debug=True, host='0.0.0.0', port=3000, use_reloader=use_reloader)
    