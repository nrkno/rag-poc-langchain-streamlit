import bs4
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    WebBaseLoader,
    DirectoryLoader,
    PyPDFDirectoryLoader,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import streamlit as st

# Requires a valid OpenAI API key to work, stored in the env variable OPENAI_API_KEY

llm = ChatOpenAI(model="gpt-4o-2024-05-13")

loader = PyPDFDirectoryLoader(
    "data/"
)  # Upload pdf files to the data folder, or switch to a different kind of loader that suits your data

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs=dict(k=20))


system_prompt = (
    # "You are an assistant for question-answering tasks. "
    # "Use the following pieces of retrieved context to answer "
    # "the question. If you don't know the answer, say that you "
    # "don't know. Use three sentences maximum and keep the "
    # "answer concise."
    "Du er en assistent som er behjelpelig med spørsmål"
    "Du skal helst svare på norsk, men kan bruke engelske begrep om det ikke finnes en god oversettelse"
    "Bruk følgende kontekst til å svare på spørsmålet."
    "Hvis du ikke vet svaret, si at du ikke vet."
    "Brukere som stille spørsmål vet ikke hvilken kontekst du har,"
    "så du må forklare svaret ditt, selv om det virker åpenbart for deg."
    "Dersom språket er svært vanskelig, slik som lov-dokumenter, kan du bruke enklere språk."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

input = st.text_input("Still et spørsmål", key="question")
if input:
    response = rag_chain.invoke({"input": input})
    st.write(response["answer"])
