import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json
import csv
from io import StringIO

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

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
    Answer the question as detailed as possible from the provided context, 
    make sure to provide all the details that are available in the context that are relevant to the question, 
    use spaces and new lines appropriately so that the user can understand the answer easily.
    If the answer is not in the provided context, just say, "answer is not available in the context", don't provide a wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}?\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def convert_to_csv(data):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Query', 'Answer'])
    writer.writerow([data['question'], data['output_text']])
    return output.getvalue()

def convert_to_json(data):
    return json.dumps(data, indent=4)

def convert_to_txt(data):
    return f"Query: {data['question']}\n\nAnswer:\n{data['output_text']}"

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: {}".format(response["output_text"]))

    # Provide export options
    data = {
        "question": user_question,
        "output_text": response["output_text"]
    }

    st.download_button(
        label="Download as CSV",
        data=convert_to_csv(data),
        file_name="query_output.csv",
        mime="text/csv"
    )

    st.download_button(
        label="Download as JSON",
        data=convert_to_json(data),
        file_name="query_output.json",
        mime="application/json"
    )

    st.download_button(
        label="Download as TXT",
        data=convert_to_txt(data),
        file_name="query_output.txt",
        mime="text/plain"
    )

def main():
    st.set_page_config("Chat PDF")
    st.header("PDF Query Application💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
