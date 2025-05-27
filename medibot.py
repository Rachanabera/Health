import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={"device": "cpu"}  # <-- this line is important
)



    db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template,input_variables=["context","question"])
    return prompt

def load_llm(huggingface_repo_id,HF_TOKEN):
    llm = HuggingFaceEndpoint(
    repo_id=huggingface_repo_id,
    temperature=0.5,
    huggingfacehub_api_token=HF_TOKEN,
    model_kwargs={"device": "cpu"}  # <-- this forces CPU instead of trying GPU
)



def main():
    st.title("Ask Medibot 🩺 ")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # Get user input
    prompt = st.chat_input("Pass your prompt here:")
    if prompt:
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
    # Get response from model
        CUSTOM_PROMPT_TEMPLATE="""

        Use the pieces of information provided in the context tpo answer user's questions.
        If you dont know the answer , just say that you dont know, dont try to make up an answer.
        Dont provide anything out of the given context
        Context:{context}
        Question:{question}
        Start the answer directly , No small talk please.
        """
        HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN=os.environ.get("HF_TOKEN")
        


        try:
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load vector store")   


            qa_chain=RetrievalQA.from_chain_type(
            llm= load_llm(HUGGINGFACE_REPO_ID,HF_TOKEN),
            chain_type= "stuff",
            retriever= vectorstore.as_retriever(search_kwargs={'k':3}),
            return_source_documents=True ,
            chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )  

            response=qa_chain.invoke({'query':prompt})
            result=response["result"]
            source_documents=response["source_documents"]
            result_to_show = result + "\n\n**Source Documents:**\n"
            for doc in source_documents:
                result_to_show += f"- Page {doc.metadata.get('page_label', doc.metadata.get('page'))} from {doc.metadata.get('source')}\n"

            st.chat_message("assistant").markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})


        except Exception as e:
            st.error(f"Error: {str(e)}")
if __name__ == "__main__":
    main()
