import os
import json
import torch
import streamlit as st
from dotenv import load_dotenv

from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    LangchainEmbedding
)
from langchain.embeddings import HuggingFaceInstructEmbeddings
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

# ---------------------------------
# Load credentials
# ---------------------------------
if "WATSONX_API_KEY" in st.secrets:
    # Running on Streamlit Cloud
    Watsonx_API = st.secrets["ApiKey-3f6f8f27-76fe-4a3c-9c72-9b4b147909ba"]
    Project_id = st.secrets["63319c07-43a6-438e-80ab-d26f1bf9ce91"]
else:
    # Running locally with .env
    load_dotenv()
    Watsonx_API = os.getenv("ApiKey-3f6f8f27-76fe-4a3c-9c72-9b4b147909ba")
    Project_id = os.getenv("63319c07-43a6-438e-80ab-d26f1bf9ce91")

if not Watsonx_API or not Project_id:
    st.error("⚠️ Missing Watsonx API credentials. Please set them in `.env` or Streamlit Secrets.")
    st.stop()

# ---------------------------------
# Basic Info
# ---------------------------------
info = {
    "Pronoun": "he",
    "Name": "Vaibhav",
    "Subject": "he",
    "Full_Name": "Vaibhav Awasthi"
}

# ---------------------------------
# Streamlit Page Setup
# ---------------------------------
st.set_page_config(page_title="💬 Chat with My AI Assistant", layout="centered")
st.title("💬 Chat with My AI Assistant")

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles/styles_chat.css")

# ---------------------------------
# Initialize Session State
# ---------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Thank you for stopping by! I’m thrilled to share my journey, projects, and passions with you. "
                   "Whether you're here to explore my work, collaborate on exciting ideas, or just get inspired—"
                   "you're in the right place. Dive in, and feel free to reach out if something sparks your interest!"
    }]

# ---------------------------------
# Sidebar
# ---------------------------------
with st.sidebar:
    st.markdown("# Chat with my AI assistant")
    with st.expander("Click here to see FAQs"):
        st.info(
            "- What are his strengths and weaknesses?\n"
            "- What is his expected salary?\n"
            "- What is his latest project?\n"
            "- When can he start to work?\n"
            "- Tell me about his professional background\n"
            "- What is his skillset?\n"
            "- What is his contact?\n"
            "- What are his achievements?"
        )

    messages = st.session_state.messages
    if messages:
        st.download_button(
            label="Download Chat",
            data=json.dumps(messages),
            file_name='chat.json',
            mime='application/json',
        )

    st.caption("© Made by Vaibhav Awasthi 2025. All rights reserved.")

# ---------------------------------
# IBM Watsonx & LlamaIndex Setup
# ---------------------------------
with st.spinner("Initiating the AI assistant. Please hold..."):
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    def init_llm():
        params = {
            GenParams.MAX_NEW_TOKENS: 512,
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
            GenParams.TEMPERATURE: 0.7,
            GenParams.TOP_K: 50,
            GenParams.TOP_P: 1
        }

        credentials = {
            'url': "https://us-south.ml.cloud.ibm.com",
            'apikey': Watsonx_API
        }

        model_id = ModelTypes.LLAMA_2_70B_CHAT
        model = Model(
            model_id=model_id,
            credentials=credentials,
            params=params,
            project_id=Project_id
        )

        llm_hub = WatsonxLLM(model=model)

        embeddings = HuggingFaceInstructEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": DEVICE}
        )

        return llm_hub, embeddings

    llm_hub, embeddings = init_llm()

    documents = SimpleDirectoryReader(input_files=["bio.txt"]).load_data()
    llm_predictor = LLMPredictor(llm=llm_hub)
    embed_model = LangchainEmbedding(embeddings)

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        embed_model=embed_model
    )

    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

# ---------------------------------
# Function to Ask Bot
# ---------------------------------
def ask_bot(user_query):
    PROMPT_QUESTION = """You are Buddy, an AI assistant dedicated to assisting {name} in {pronoun} job search by 
    providing recruiters with relevant information about {pronoun} qualifications and achievements. 
    Your goal is to support {name} in presenting {pronoun}self effectively to potential employers and 
    promoting {pronoun} candidacy for job opportunities.
    If you do not know the answer, politely admit it and let recruiters know how to contact {name} directly. 
    Don't put "Buddy" or a breakline in the front of your answer.
    Human: {input}
    """
    return index.as_query_engine().query(
        PROMPT_QUESTION.format(name=info["Name"], pronoun=info["Pronoun"], input=user_query)
    )

# ---------------------------------
# Chat Input & Display
# ---------------------------------
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            response = ask_bot(prompt)
            st.write(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})

# ---------------------------------
# Suggested Questions
# ---------------------------------
questions = [
    f'What are {info["Pronoun"]} strengths and weaknesses?',
    f'What is {info["Pronoun"]} latest project?',
    f'When can {info["Subject"]} start to work?'
]

if "disabled" not in st.session_state:
    st.session_state.disabled = False

if not st.session_state.disabled:
    for q in questions:
        if st.button(q):
            resp = ask_bot(q)
            st.session_state.messages.append({"role": "user", "content": q})
            st.session_state.messages.append({"role": "assistant", "content": resp.response})
