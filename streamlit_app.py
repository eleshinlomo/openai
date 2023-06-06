# an ai project that allows users to interract with pdf documents

import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS  
from langchain.memory import ConversationBufferMemory
from htmlTemplates import css, bot_template, user_template
# language model
from langchain.chat_models import ChatOpenAI
# allows chatting with get_vectorstore and retrieve last conversation
from langchain.chains import ConversationalRetrievalChain



# extracts text from uploaded PDFs

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# splits the "text" into chunks to feed the database/vector store and models

def get_text_chunk(text):
    chunk_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = chunk_splitter.split_text(text)
    
    return chunks

# handling chunks, embeddings for vector store 

def get_vectorstore(text_chunk):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunk, embedding= embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages= True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever= vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain

# user question handler

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # loop through the chat history

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html= True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html= True)


# User UI for pdf upload and interractions

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDF", page_icon=":books:")

    # css wrapper

    st.write(css, unsafe_allow_html= True)

# session state initialization(Important when using session state)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("INTERRACTIVE PDF CHATBOT :books:")
    st.subheader("Never have to read through bunch of documents anymore. Ask this bot about everything in your document.")
    user_question = st.text_input("Upload your PDF Document on the left and ask a question about your document here: ")
    if user_question:
        handle_userinput(user_question)
        

    with st.sidebar:
        st.subheader("Your PDF Document")
        pdf_docs = st.file_uploader("Upload your file and click on process", accept_multiple_files = True)
        if st.button("Process"):
            with st.spinner("Processing..."):

                # get pdf text. This calls the get_pdf_text function above

                raw_text = get_pdf_text(pdf_docs)
                
                # get text chunk to feed the database

                text_chunk = get_text_chunk(raw_text)
                st.write(text_chunk)
                
                # create vector store
                vectorstore = get_vectorstore(text_chunk)

                # create conversation with memory 
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()