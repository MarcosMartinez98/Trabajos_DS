import streamlit as st
import os
import cohere
from dotenv import load_dotenv 


load_dotenv()


COHERE_API_KEY = os.environ["COHERE_API_KEY"]


if not COHERE_API_KEY:
    st.error("Error: La clave API de Cohere (COHERE_API_KEY) no está configurada. ")
    st.stop() 

try:
    
    co = cohere.ClientV2(COHERE_API_KEY)
except Exception as e:
    st.error(f"Error al inicializar el cliente de Cohere: {e}")
    st.stop() 

def api_cohere(text):
   
    system_message = (
        "Eres un asistente virtual, sé conciso y usa como máximo 200 tokens por respuesta. "
        "Intenta emular a Chiquito de la Calzada siempre empezarás tus respuestas con "
        "una frase de Chiquito de la Calzada. No te excedas de los 200 tokens."
    )
    try:
        res = co.chat(
            model="command-a-03-2025",
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )
        # Ensure the response structure matches what you expect
        if res and res.message and res.message.content and res.message.content[0] and res.message.content[0].text:
            return res.message.content[0].text
        else:
            return "¡Fistro! No he podido generar una respuesta adecuada. Inténtalo de nuevo, pecador."
    except Exception as e:
        st.error(f"Error al llamar a la API de Cohere: {e}")
        return "¡Condemor! Algo ha ido mal al intentar comunicarme con Chiquito. Prueba otra vez."


st.title("Chatbot con Chiquito de la Calzada")
st.markdown("¡Aquí tu asistente personal, pecador de la pradera!")

# Initialize chat history if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
prompt = st.chat_input("Escribe aquí tu consulta")
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Pensando, ¡al ataqueer! (Espera un poquito, pecador)..."):
            response = api_cohere(prompt)
            st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

