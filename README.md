# 🧠 My AI Agent

My AI Agent is a **Streamlit-based AI-powered portfolio and chatbot application**.  
It merges personal portfolio presentation with an interactive chatbot that leverages modern AI frameworks and cloud-based services.

---

## 📂 Project Structure

myaiagent/
├── .streamlit/ # Streamlit configuration files (e.g., config.toml)
├── images/ # Images for hobbies and CV
├── pages/ # Additional Streamlit pages (.py files)
├── style/ # CSS files for custom styling
├── utils/ # Static data and helper functions
├── Dockerfile # Docker setup for deployment
├── bio.txt # Chatbot knowledge base
├── requirements.txt # Python dependencies
└── 💼Portfolio.py # Main app entry point


---

## ⚙️ How It Works

### **Step 1: Importing Key Libraries**
At the start of the code, essential libraries are imported — each with a specific role in building and running the chatbot:

- **`streamlit`** → Builds the user interface and handles chat interactions.
- **`torch`** → Enables GPU acceleration for efficient computations, especially for deep learning.
- **`llama_index`** → Serves as the AI data access layer for advanced retrieval and querying.
- **`langchain`** → Handles embeddings, converting text into numerical vectors for ML processing.
- **`ibm_watson_machine_learning`** → Integrates with **Watsonx.ai** for IBM Watson’s ML capabilities.

These libraries form the backbone of the application, enabling an efficient, scalable, and interactive chatbot experience.

---

## 🚀 Features

- **Personalized Portfolio** – Showcases personal details, hobbies, and projects.
- **AI Chatbot** – Powered by a knowledge base (`bio.txt`) and AI embeddings.
- **Advanced Retrieval** – `llama_index` and `langchain` enable intelligent, context-aware responses.
- **Custom Styling** – Modify look and feel with CSS from the `style/` folder.
- **Docker Support** – Easy deployment with the included `Dockerfile`.
- **Streamlit Config** – `.streamlit/config.toml` controls theme and settings.

---

Vaibhav Awasthi
📧 v.a.awasthivaibhav@gmail.com
