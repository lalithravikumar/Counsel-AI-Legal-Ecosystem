import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import boto3
from botocore.exceptions import ClientError
import os
import pickle
import requests
import io
from pydantic import field_validator
st.set_page_config(page_title="Counsel AI", page_icon="https://i.postimg.cc/442yR08M/law-Symbol.png")

# Load environment variables
load_dotenv()
AWS_REGION = "ap-southeast-2"
USER_POOL_ID = os.getenv("USER_POOL_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
S3_BUCKET = os.getenv("S3_BUCKET")

# AWS Clients
cognito_client = boto3.client("cognito-idp", region_name=AWS_REGION)
s3_client = boto3.client("s3", region_name=AWS_REGION)

def register_user(username, password, email):
    try:
        response = cognito_client.sign_up(
            ClientId=CLIENT_ID,
            Username=username,
            Password=password,
            UserAttributes=[{"Name": "email", "Value": email}],
        )

        st.success("Registration successful. Please check your email for the verification code.")
    except ClientError as e:
        st.error(f"Error: {e.response['Error']['Message']}")

def confirm_user(username, verification_code):
    try:
        cognito_client.confirm_sign_up(
            ClientId=CLIENT_ID,
            Username=username,
            ConfirmationCode=verification_code,
        )
        st.success("Account verified successfully! Please log in.")
    except ClientError as e:
        st.error(f"Verification failed: {e.response['Error']['Message']}")

def login_user(username, password):
    try:
        response = cognito_client.initiate_auth(
            ClientId=CLIENT_ID,
            AuthFlow="USER_PASSWORD_AUTH",
            AuthParameters={"USERNAME": username, "PASSWORD": password},
        )
        st.session_state["auth_token"] = response["AuthenticationResult"]["AccessToken"]
        st.session_state["username"] = username
        return True
    except ClientError as e:
        st.error("Login failed. Check your credentials.")
        return False

def logout_user():
    st.session_state.clear()
    st.success("You have logged out successfully!")


def login_or_register():
    st.title("Login/Register")
    auth_option = st.selectbox("Choose an option", ["Login", "Register", "Verify Account"])

    username = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if auth_option == "Register" and st.button("Register"):
        register_user(username, password, username)
    elif auth_option == "Login" and st.button("Login"):
        if login_user(username, password):
            st.session_state["logged_in"] = True
            st.success("Login successful!")
    elif auth_option == "Verify Account":
        verification_code = st.text_input("Verification Code")
        if st.button("Verify"):
            confirm_user(username, verification_code)


def main_app():
    st.write(css, unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.image("https://i.postimg.cc/JnR9z2Bf/law-Symbol.png", width=50)
    st.sidebar.title("Counsel AI")
    nav_option = st.sidebar.radio("Go to", ["Chat with your Data", "Consult a Legal Expert", "File Manager"])
    

    if nav_option == "Chat with your Data" :  # Hide the upload section for Legal Expert page
        st.sidebar.subheader("Select Files from File Manager or Upload Your Documents")
        folders = list_folders()
        selected_folder = st.sidebar.selectbox("Choose a folder", folders, key="folder_selection")
        
        if selected_folder:
            files = list_files_in_folder(selected_folder)
            selected_files = st.sidebar.multiselect("Select files to process", files, key="file_selection")

        pdf_docs = st.sidebar.file_uploader("Or Upload your PDFs here", accept_multiple_files=True, key="pdf_upload")

        # Process files based on user choice
        if st.sidebar.button("Process"):
            if pdf_docs:
                with st.spinner("Processing uploaded files..."):
                    raw_text = get_pdf_text(pdf_docs)
                    process_text_chunks(raw_text)
            elif selected_files:
                with st.spinner("Processing selected files from File Manager..."):
                    raw_text = get_text_from_s3_files(selected_files)
                    process_text_chunks(raw_text)
            else:
                st.warning("Please upload or select at least one PDF file.")

    # Main Content
    if nav_option == "Chat with your Data":
        st.header("Chat with your data :books:")

        user_question = st.text_input("Ask a question about your data:")
        if user_question:
            handle_userinput(user_question)

    elif nav_option == "Consult a Legal Expert":
        legal_expert_page()

    elif nav_option == "File Manager":
        file_manager()
    with st.sidebar.empty():
        st.sidebar.markdown("---")  # Line separator
        st.sidebar.markdown("### User Profile")
        st.sidebar.write(f"Logged in as: **{st.session_state.get('username')}**")
        if st.sidebar.button("Logout"):
            logout_user()
        

def upload_file(file, folder_name):
    try:
        s3_client.upload_fileobj(file, S3_BUCKET, f"{folder_name}/{file.name}")
        st.success(f"{file.name} uploaded to {folder_name} successfully.")
    except ClientError as e:
        st.error(f"Could not upload {file.name}: {e.response['Error']['Message']}")

def create_folder(folder_name):
    try:
        s3_client.put_object(Bucket=S3_BUCKET, Key=(folder_name + '/'))
        return True
    except ClientError as e:
        st.error(f"Could not create folder: {e.response['Error']['Message']}")
        return False
    st.sidebar.markdown("User Profile")
    st.sidebar.write(f"Logged in as: **{st.session_state.get('username')}**")
    if st.sidebar.button("Logout"):
        logout_user()

# Helper Functions
def list_folders():
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Delimiter='/')
    return [prefix['Prefix'].rstrip('/') for prefix in response.get('CommonPrefixes', [])]

def list_files_in_folder(folder_name):
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{folder_name}/")
    return [obj['Key'] for obj in response.get('Contents', []) if obj['Key'] != f"{folder_name}/"]

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

def get_text_from_s3_files(file_keys):
    text = ""
    for file_key in file_keys:
        file_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
        pdf_bytes = io.BytesIO(file_obj['Body'].read())
        pdf_reader = PdfReader(pdf_bytes)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def process_text_chunks(raw_text):
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    st.session_state.conversation = get_conversation_chain(vectorstore)

# Define a helper function to check and load conversation history from session state
def load_conversation_history():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    return st.session_state["chat_history"]

# Update the chat history after every user input
def handle_userinput(user_question):
    # Get the chat history (if it exists) or create a new one
    chat_history = load_conversation_history()

    # Process the user input with your conversational retrieval chain
    response = st.session_state.conversation({'question': user_question})

    # Append the new message to the chat history
    chat_history.append({'user': user_question, 'bot': response['answer']})

    # Save it back to session_state
    st.session_state["chat_history"] = chat_history

    # Display the chat history
    for message in chat_history:
        # Assuming user_template and bot_template are defined for styling
        st.write(user_template.replace("{{MSG}}", message['user']), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", message['bot']), unsafe_allow_html=True)


def legal_expert_page():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    embeddings_file = "LawEmbeddings.pkl"
    vectorstore = load_embeddings(embeddings_file)
    st.session_state.conversation = get_conversation_chain(vectorstore)

    st.header("Legal Expert") 
    
    user_question = st.text_input("Ask your query:")
    if user_question:
        handle_userinput(user_question)

def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        vectorstore = pickle.load(f)
    return vectorstore

#New file manager
def file_manager():
    st.header("File Manager")
    
    # List and display folders with popup options
    folders = list_folders()
    selected_folder = st.selectbox("Select a folder to view its contents", folders)

    if selected_folder:
        view_folder_contents(selected_folder)

    # Popup for creating a new folder
    with st.expander("Create New Folder"):
        new_folder_name = st.text_input("Enter new folder name")
        if st.button("Create Folder"):
            if create_folder(new_folder_name):
                st.success(f"Folder '{new_folder_name}' created successfully!")
            else:
                st.error("Failed to create folder. Please try again.")

    # Upload files to selected folder
    with st.expander("Upload Files"):
        folder_to_upload = st.selectbox("Select folder for upload", folders)
        uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)
        if st.button("Upload Files"):
            if uploaded_files:
                for file in uploaded_files:
                    upload_file(file, folder_to_upload)
                st.success("Files uploaded successfully!")

def view_folder_contents(folder_name):
    st.subheader(f"Contents of {folder_name}")
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{folder_name}/")
    files = [obj['Key'] for obj in response.get('Contents', [])]

    for file_key in files:
        st.write(file_key)
        if file_key.endswith('.pdf'):
            if st.button(f"Open PDF: {file_key}"):
                # Fetch the PDF file from S3
                file_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
                
                # Use BytesIO to read the file content
                pdf_bytes = io.BytesIO(file_obj['Body'].read())
                pdf_reader = PdfReader(pdf_bytes)  # Initialize PdfReader with the BytesIO object
                
                # Assuming single-page for demo
                if len(pdf_reader.pages) > 0:
                    pdf_text = pdf_reader.pages[0].extract_text()
                    st.text_area("PDF Content:", pdf_text, height=300)
                else:
                    st.warning("The PDF has no pages.")
        elif file_key.endswith(('.jpg', '.jpeg', '.png')):
            st.image(s3_client.generate_presigned_url('get_object', Params={'Bucket': S3_BUCKET, 'Key': file_key}), use_column_width=True)
        elif file_key.endswith('.docx'):
            st.write("DOCX file preview not available. [Download it here](%s)" % s3_client.generate_presigned_url('get_object', Params={'Bucket': S3_BUCKET, 'Key': file_key}))

def list_folders():
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Delimiter='/')
    return [prefix['Prefix'].rstrip('/') for prefix in response.get('CommonPrefixes', [])]

def create_folder(folder_name):
    try:
        s3_client.put_object(Bucket=S3_BUCKET, Key=(folder_name + '/'))
        return True
    except ClientError as e:
        st.error(f"Could not create folder: {e}")
        return False

def upload_file(file, folder_name):
    try:
        s3_client.upload_fileobj(file, S3_BUCKET, f"{folder_name}/{file.name}")
        st.success(f"{file.name} uploaded to {folder_name} successfully.")
    except ClientError as e:
        st.error(f"Could not upload {file.name}: {e}")

# Entry point for the application
if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        login_or_register()
    else:
        main_app()