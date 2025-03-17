import os
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

st.title("News research tool")
st.sidebar.title("News article urls")

urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens = 500)

file_path = 'faiss_index'

if process_url_clicked:

    # loading the data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data loading started...")
    data = loader.load()

    # chunking
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size = 1000
    )
    docs = text_splitter.split_documents(data)
    main_placeholder.text("Text splitting started...")

    # creating embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding started...")
    time.sleep(2)

    vectorstore_openai.save_local("faiss_index")

query = main_placeholder.text_input("Question: ")
if query:
    vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever = vectorstore.as_retriever())
    result = chain({'question': query}, return_only_outputs = True)
    st.header("Answer")
    st.write(result["answer"])

