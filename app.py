import os 
import google.generativeai as genai
from pdfextractor import text_extractor
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# Lets configure the models
# LLM mOdels
gemini_key = os.getenv("test-project-2")
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-2.5-flash')

# Configure Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Lets create the main page 
st.title(":rainbow[CHATBOT]:  :blue[AI Assisted using RAG]")
tips = '''
Follow the steps to use the application:-
* Upload your PDF document using the sidebar.
* Write your query and start the chat.'''
st.text(tips)

# Lets create the sidebar
st.sidebar.title('UPLOAD YOUR DOCUMENT(pdf only)')
pdf_file = st.sidebar.file_uploader('Upload here', type=['pdf'])
if pdf_file:
    st.sidebar.success('File Uploaded Successfully')

    file_text = text_extractor(pdf_file)

    # Step1: Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks = splitter.split_text(file_text)

    # Step 2: Create Vector database(FAISS)
    vector_store = FAISS.from_texts(chunks, embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={'k':3})

    def generate_content(query):
        # Step 3: Retrieval (R)
        retrived_docs = retriever.invoke(query)
        context = '\n'.join([d.page_content for d in retrived_docs])


        # Step 4: Augmenting
        augmented_prompt = f'''
        <ROLE> You are a helpful assistant using RAG.
        <GOAL> Answer the question asked by the user. Here is the question {query}
        <CONTEXT> Here are the documents retrieved from the vector database to support the answer
          which you have to generate{context}.
        '''
        
        # Step 5: Generate (G)
        response = model.generate_content(augmented_prompt)
        return response.text
    
    # Create ChatBot in order to start the conversation
    # Initialize a chat create history if not created
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Display the chat history
    for message in st.session_state.history:
        if message['role'] == 'user':
            st.info(f"User:{message['text']}")
        else:
            st.success(f"CHATBOT:{message['text']}")

    
    # Input from the user using streamlit form
    with st.form('Chatbot Form ', clear_on_submit=True):
        user_query = st.text_area("Enter your question:")
        send = st.form_submit_button("Send")

        # Start the conversation and append output and query in history
        if user_query and send:
            with st.spinner("CHATBOT is typing..."):
                response = generate_content(user_query)
                st.session_state.history.append({'role':'user', 'text':user_query})
                st.session_state.history.append({'role':'bot', 'text':generate_content(user_query)})
                st.rerun()