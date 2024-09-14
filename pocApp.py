import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import textwrap
import warnings
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure Google Generative AI
api_key = os.getenv("GOOGLE_API_KEY")
import google.generativeai as genai
genai.configure(api_key=api_key)

# Function to create a vector store from a URL
def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    vector_store = Chroma.from_documents(document_chunks, embeddings)
    return vector_store

def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_excel_text(excel_file):
    df = pd.read_excel(excel_file, engine='openpyxl')
    return df.to_string()

def get_url_text(url):
    warnings.simplefilter('ignore', InsecureRequestWarning)
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        st.error(f"Error fetching URL content: {e}")
        return ""

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say, "answer is not available in the context". Do not provide the wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    data = response["output_text"]
    #if "answer is not available in the context".lower() not in data.lower():
    st.write("Reply: ", data)
    #else:
#         prompt_template = """
#         Answer the user's question as best as you can based on your knowledge. If the user is asking about advanced certification courses, suggest relevant courses and provide a link to a detailed blog post on Amazon that lists such certifications.

#         Question:
#         {question}

#         Answer:
# """
        #model = genai.GenerativeModel('gemini-pro')
        #prompt = prompt_template.format(question=user_question)
        #response = model.generate_content(prompt)
        #text_content = response.candidates[0].content.parts[0].text
        #st.write("Reply: ", text_content)

def main():
    st.set_page_config(page_title="Chat Bot")
    st.header("POC AI BOT")

    # Sidebar
    with st.sidebar:
        st.title("Menu:")
        input_type = st.selectbox("Select Input Type", ["PDF", "Excel", "URL"])

        if input_type == "PDF":
            pdf_file = st.file_uploader("Upload your PDF File", type=["pdf"])
            if st.button("Submit & Process PDF"):
                if pdf_file:
                    raw_text = get_pdf_text(pdf_file)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete")
                else:
                    st.error("Please upload a PDF file.")
        
        elif input_type == "Excel":
            excel_file = st.file_uploader("Upload your Excel File", type=["xlsx"])
            if st.button("Submit & Process Excel"):
                if excel_file:
                    raw_text = get_excel_text(excel_file)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete")
                else:
                    st.error("Please upload an Excel file.")
        
        elif input_type == "URL":
            url = st.text_input("Enter URL")
            if st.button("Submit & Process URL"):
                if url:
                    raw_text = get_url_text(url)
                    if raw_text:  # Check if URL content was successfully fetched
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Processing complete")
                    else:
                        st.error("Failed to fetch or process URL content.")
                else:
                    st.error("Please enter a URL.")

    # Main content
    st.subheader("Ask a Question")
    user_question = st.text_input("Your Question")
    if user_question and st.button("Submit Question"):
        user_input(user_question)

if __name__ == "__main__":
    main()