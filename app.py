import streamlit as st
import tempfile

from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.title("🤖 AI Research Assistant")
st.subheader("Chat with PDF, Word, PPT, Text & Excel files")

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "search_history" not in st.session_state:
    st.session_state.search_history = []

if "selected_history" not in st.session_state:
    st.session_state.selected_history = None

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("🔑 Configuration")

    groq_api_key = st.text_input("Groq API Key", type="password")

    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf", "docx", "txt", "pptx", "xlsx"]
    )

    st.markdown("[Get Free Groq API Key](https://console.groq.com/keys)")

    st.divider()
    st.subheader("🕘 Previous Searches")

    if st.session_state.search_history:
        for i, item in enumerate(reversed(st.session_state.search_history)):
            if st.button(item["question"][:40], key=f"hist_{i}"):
                st.session_state.selected_history = item
    else:
        st.caption("No previous searches")

# -------------------------------------------------
# VALIDATION
# -------------------------------------------------
if not groq_api_key:
    st.warning("⬅️ Enter Groq API Key")
    st.stop()

if not uploaded_file:
    st.warning("⬅️ Upload a document")
    st.stop()

# -------------------------------------------------
# FILE SAVE
# -------------------------------------------------
with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
    tmp.write(uploaded_file.getvalue())
    file_path = tmp.name

ext = uploaded_file.name.split(".")[-1].lower()

# -------------------------------------------------
# LOAD DOCUMENT
# -------------------------------------------------
if ext == "pdf":
    loader = PyPDFLoader(file_path)
elif ext == "docx":
    loader = Docx2txtLoader(file_path)
elif ext == "txt":
    loader = TextLoader(file_path, encoding="utf-8")
elif ext == "pptx":
    loader = UnstructuredPowerPointLoader(file_path)
elif ext == "xlsx":
    loader = UnstructuredExcelLoader(file_path)
else:
    st.error("Unsupported file type")
    st.stop()

documents = loader.load()

# -------------------------------------------------
# TEXT SPLITTING + EMBEDDINGS
# -------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever()

# -------------------------------------------------
# LLM
# -------------------------------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

prompt = ChatPromptTemplate.from_template(
    """
You are an AI research assistant.
Answer ONLY from the provided context.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# -------------------------------------------------
# SHOW SELECTED HISTORY
# -------------------------------------------------
if st.session_state.selected_history:
    st.info("📜 Previous Search")
    st.markdown("**Question**")
    st.write(st.session_state.selected_history["question"])
    st.markdown("**Answer**")
    st.write(st.session_state.selected_history["answer"])
    st.divider()

# -------------------------------------------------
# CHAT UI
# -------------------------------------------------
st.header("💬 Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask something about the document...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        retrieved_docs = retriever.invoke(user_input)
        response = rag_chain.invoke(user_input)
        answer = response.content
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.search_history.append({"question": user_input, "answer": answer})
    st.session_state.selected_history = None
