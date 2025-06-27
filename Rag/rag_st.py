import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from dotenv import load_dotenv
import os
import tempfile

# Carga las variables de entorno al iniciocd 
load_dotenv()

# --- Configuración de Streamlit ---
st.set_page_config(page_title="RAG con PDF y Cohere", layout="centered")
st.title("📚 RAG con tus PDFs y Cohere")
st.markdown("Sube un PDF, y luego hazme preguntas sobre su contenido.")

# --- Inicialización de modelos y funciones (Caché para evitar recargas) ---

@st.cache_resource
def load_cohere_models():
    """Carga los modelos de Cohere una sola vez."""
    try:
        cohere_api_key = os.getenv("COHERE_API_KEY") # Obtén el valor aquí
        
        if not cohere_api_key:
            st.error("Error: La variable de entorno 'COHERE_API_KEY' no está configurada o está vacía. Por favor, asegúrate de que tu archivo 'gemini/.env' contenga la clave.")
            st.stop()
            
        # PASA LA CLAVE EXPLÍCITAMENTE AQUÍ
        embeddings = CohereEmbeddings(model="embed-v4.0", cohere_api_key=cohere_api_key)
        llm = ChatCohere(model="command-a-03-2025", cohere_api_key=cohere_api_key) 
        
        return embeddings, llm
    except Exception as e:
        st.error(f"Error al cargar los modelos de Cohere: {e}")
        st.stop()

embeddings, llm = load_cohere_models()

@st.cache_resource
def process_pdf_and_create_vectordb(uploaded_file):
    """
    Procesa el PDF, crea chunks y genera un Chroma VectorDB.
    Usa tempfile para manejar archivos subidos.
    """
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        st.info("Cargando y procesando el PDF... Esto puede tardar un momento.")
        try:
            # 1. Cargar el documento
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()

            # 2. Dividir el documento en chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, # Ajustado el chunk_size para un equilibrio
                chunk_overlap=150
            )
            chunks = text_splitter.split_documents(documents)
            
            st.write(f"PDF procesado: {len(documents)} páginas, dividido en {len(chunks)} fragmentos.")

            # 3. Crear el Vector Store Chroma
            vector_db = Chroma.from_documents(chunks, embeddings)
            st.success("PDF procesado y base de datos de vectores creada.")
            return vector_db
        except Exception as e:
            st.error(f"Error al procesar el PDF: {e}")
            return None
        finally:
            # Limpiar el archivo temporal
            if 'pdf_path' in locals() and os.path.exists(pdf_path):
                os.remove(pdf_path)
    return None

# --- Interfaz de usuario para subir el PDF ---
uploaded_file = st.file_uploader("Sube tu archivo PDF aquí", type="pdf")

vector_db = None
if uploaded_file:
    vector_db = process_pdf_and_create_vectordb(uploaded_file)

# --- Interfaz de usuario para hacer preguntas ---
if vector_db:
    st.subheader("Hazme preguntas sobre el PDF:")
    user_query = st.text_input("Tu pregunta:")

    if user_query:
        # 1. Crear un "retriever" desde nuestro vector store
        retriever = vector_db.as_retriever(search_kwargs={"k": 3}) # Obtener los 3 chunks más relevantes

        # 2. Usar un prompt de la comunidad de LangChain
        prompt = hub.pull("rlm/rag-prompt")

        # 3. Crear la cadena RAG
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        with st.spinner("Buscando y generando respuesta..."):
            response = rag_chain.invoke(user_query)
            st.markdown("---")
            st.write("**Respuesta:**")
            st.info(response)
else:
    st.info("Sube un PDF para empezar a hacer preguntas.")

st.markdown("---")
st.markdown("Desarrollado con LangChain, Streamlit y Cohere.")