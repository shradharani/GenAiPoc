# import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate


def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks,embeddings)

    return vector_store



def get_context_retriever_chain(vector_store):
    # llm = ChatOpenAI()
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain



GOOGLE_API_KEY="AIzaSyB_VtaStDXRpaGqdahwYv-8ys-ZXHITd4s"

import os
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

prompt_template="""

Given the context provided, answer the following question accurately.
If the information needed to answer is not found in the context, simply respond with 'Not in context.'

Context: {{context}}

Question: {{question}}


"""

model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])




from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Prompt template
template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
# model = ChatOpenAI(temperature=0, model="gpt-4")
url="https://www.geeksforgeeks.org/aws-tutorial/"
vector_store=get_vectorstore_from_url(url)

retriever = vector_store.as_retriever()

# RAG pipeline
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)



# print(chain.invoke("who directed Chatrapathi movie?"))

print(chain.invoke("What is Amazon Web Service or AWS"))