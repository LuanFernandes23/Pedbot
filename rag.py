import os
import time
import streamlit as st
import sqlite3
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import CohereEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from gtts import gTTS

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

DATABASE = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                 id INTEGER PRIMARY KEY,
                 session_id TEXT,
                 role TEXT,
                 content TEXT,
                 timestamp_local TEXT DEFAULT (DATETIME('now', 'localtime', '-5 hours')))''')
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
                 session_id TEXT PRIMARY KEY,
                 start_time TEXT DEFAULT (DATETIME('now', 'localtime', '-5 hours')))''')
    conn.commit()
    conn.close()

# Salvar mensagem no banco de dados
def save_message(session_id, role, content):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
              (session_id, role, content))
    conn.commit()
    conn.close()

# Função para carregar mensagens do banco de dados
def load_messages(session_id):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM chat_history WHERE session_id = ?", (session_id,))
    messages = c.fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in messages]

# Função para salvar a sessão no banco de dados
def save_session(session_id):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("INSERT INTO sessions (session_id) VALUES (?)", (session_id,))
    conn.commit()
    conn.close()

# Função para carregar sessões do banco de dados
def load_sessions():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT session_id, start_time FROM sessions ORDER BY start_time DESC")
    sessions = c.fetchall()
    conn.close()
    return sessions

# Função para deletar uma sessão do banco de dados
def delete_session(session_id):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
    c.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

class DocumentLoader:
    def __init__(self):
        self.vectorstore_index_name = "rag-streamlit"
        self.embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=COHERE_API_KEY)
        self.gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest", temperature=0.3, google_api_key=GOOGLE_API_KEY
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        self.create_pinecone_index(self.vectorstore_index_name)
        self.vectorstore = PineconeVectorStore(
            index_name=self.vectorstore_index_name,
            embedding=self.embeddings,
            pinecone_api_key=PINECONE_API_KEY
        )
        self.rag_prompt = ChatPromptTemplate.from_template(
            """Você é um atencioso e útil assistente IA que tem a tarefa de responder as perguntas do usuário.
               Você é amigável e responde extensivamente com várias frases quando necessário. Você pode usar marcadores para resumir.
               Contexto: {context}
               Pergunta: {question}
               Resposta:"""
        )
        self.retriever = None
        self.rag_chain = None

    def create_pinecone_index(self, vectorstore_index_name):
        pc = Pinecone(api_key=PINECONE_API_KEY)
        spec = ServerlessSpec(cloud='aws', region='us-east-1')
        if vectorstore_index_name in pc.list_indexes().names():
            pc.delete_index(vectorstore_index_name)
        pc.create_index(
            vectorstore_index_name,
            dimension=1024,
            metric='dotproduct',
            spec=spec
        )
        while not pc.describe_index(vectorstore_index_name).status['ready']:
            time.sleep(1)

    def load_docs_from_pdf(self, pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return [Document(page_content=text)]

    def load_docs_from_urls(self, web_urls):
        docs = []
        for web_url in web_urls:
            loader = WebBaseLoader(web_url)
            docs.extend(loader.load())
        return [Document(page_content=doc.page_content) for doc in docs]

    def split_and_store_docs(self, docs):
        split_docs = self.text_splitter.split_documents(docs)
        self.vectorstore.add_documents(split_docs)

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def create_retrieval_chain(self):
        self.retriever = self.vectorstore.as_retriever()
        self.rag_chain = (
            {
                "context": self.retriever | self.format_docs, "question": RunnablePassthrough()
            }
            | self.rag_prompt
            | self.gemini_llm
            | StrOutputParser()
        )

    def qa(self, query, session_id):
        if not self.rag_chain:
            self.create_retrieval_chain()
        response = self.rag_chain.invoke(query)
        save_message(session_id, "user", query)
        save_message(session_id, "assistant", response)
        return response
    
    def clear_database(self):
        self.create_pinecone_index(self.vectorstore_index_name)
        self.vectorstore = PineconeVectorStore(
            index_name=self.vectorstore_index_name,
            embedding=self.embeddings,
            pinecone_api_key=PINECONE_API_KEY
        )

# Função para processar PDFs automaticamente
def process_pdfs():
    pdf_docs = st.session_state.get('uploaded_pdfs', [])
    if pdf_docs:
        loader = st.session_state.loader
        with st.sidebar:
            with st.spinner("Processando PDFs..."):
                try:
                    raw_text = loader.load_docs_from_pdf(pdf_docs)
                    loader.split_and_store_docs(raw_text)
                    loader.create_retrieval_chain()
                    st.session_state.docs_processed = True
                    st.sidebar.success("Documentos processados e banco de dados atualizado :white_check_mark:")
                except Exception as e:
                    st.sidebar.error(f"Erro ao processar PDFs: {e}")

# Função para gerar o áudio usando gTTS
def generate_audio(response, session_id):
    response = response.replace('*', '')
    tts = gTTS(response, lang='pt')
    audio_file = f"response_{session_id}.mp3"
    tts.save(audio_file)
    return audio_file

# Streamlit App
def main():
    st.set_page_config(
        page_title="Converse com seu site/PDF",
        page_icon=":orange_heart:"
    )

    st.title("Chat with PDFs/URLs :book:")
    st.write("Use it wisely, pretty :heart:")
    
    if "loader" not in st.session_state:
        st.session_state.loader = DocumentLoader()
        st.session_state.docs_processed = False
    
    loader = st.session_state.loader
    
    # Inicializa o banco de dados
    init_db()

    # Gerar ID de sessão único
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(time.time())
        save_session(st.session_state.session_id)
    
    session_id = st.session_state.session_id

    # Carregar histórico de mensagens do banco de dados
    if "messages" not in st.session_state:
        st.session_state.messages = load_messages(session_id)

    with st.sidebar:
        st.title("Menu:")
        
        st.file_uploader(
            "Faça upload dos PDFs", accept_multiple_files=True, key='uploaded_pdfs', on_change=process_pdfs)

        urls = st.text_area("Cole os URLs, um por linha:")

        if urls:
            if not st.session_state.docs_processed:
                with st.spinner("Processando URLs..."):
                    try:
                        url_list = [url.strip() for url in urls.split("\n") if url.strip()]
                        if url_list:
                            docs = loader.load_docs_from_urls(url_list)
                            loader.split_and_store_docs(docs)
                        loader.create_retrieval_chain()
                        st.session_state.docs_processed = True
                        st.sidebar.success("Documentos processados e banco de dados atualizado :white_check_mark:")
                    except Exception as e:
                        st.sidebar.error(f"Erro ao processar URLs: {e}")

        if st.button("Limpar banco de dados vetorial :x:"):
            with st.spinner("Limpando banco de dados..."):
                try:
                    loader.clear_database()
                    st.session_state.docs_processed = False
                    st.sidebar.success("Banco de dados limpo com sucesso :white_check_mark:")
                except Exception as e:
                    st.sidebar.error(f"Erro ao limpar o banco de dados: {e}")

        st.write("Histórico de Sessões")
        sessions = load_sessions()
        session_options = [f"Sessão de {session[1]}" for session in sessions]
        selected_session = st.selectbox("Selecione uma sessão", session_options)

        if selected_session:
            selected_session_id = sessions[session_options.index(selected_session)][0]
            st.session_state.session_id = selected_session_id
            st.session_state.messages = load_messages(selected_session_id)

        if sessions:
            st.write("Gerenciar Sessões")
            for session in sessions:
                session_id, start_time = session
                if st.button(f"Deletar {start_time}", key=session_id):
                    delete_session(session_id)
                    st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Adicionando a reprodução de áudio para as respostas do assistente
            if message["role"] == "assistant":
                audio_file = generate_audio(message["content"], session_id)
                audio_bytes = open(audio_file, "rb").read()
                st.audio(audio_bytes, format="audio/mp3")

    if prompt := st.chat_input():
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(time.time())
            save_session(st.session_state.session_id)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        if st.session_state.docs_processed:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = loader.qa(prompt, st.session_state.session_id)
                        full_response = response if isinstance(response, str) else response['output_text']
                        st.write(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})

                        # Gerar e reproduzir áudio da resposta do assistente
                        audio_file = generate_audio(full_response, session_id)
                        audio_bytes = open(audio_file, "rb").read()
                        st.audio(audio_bytes, format="audio/mp3")
                    except Exception as e:
                        st.error(f"Erro ao gerar resposta: {e}")
        else:
            with st.chat_message("assistant"):
                st.write("Por favor, carregue e processe os documentos antes de fazer uma pergunta.")

if __name__ == "__main__":
    main()