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
import io
from PyPDF2 import PdfReader

st.set_page_config(page_title="Counsel AI", page_icon=r"C:\Users\bjlal\Documents\KMS J\KMS\data\lawSymbol.png")


load_dotenv()
AWS_REGION = "ap-southeast-2"  
USER_POOL_ID = os.getenv("USER_POOL_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
S3_BUCKET = os.getenv("S3_BUCKET")
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

    st.sidebar.title("Navigation")
    nav_option = st.sidebar.radio("Go to", ["Chat with your Data", "Consult a Legal Expert", "File Manager"])

    if nav_option == "Chat with your Data":
        st.header("Chat with your data :books:")
        user_question = st.text_input("Ask a question about your data:")
        if user_question:
            handle_userinput(user_question)

        with st.sidebar:
            st.subheader("Your documents")
            pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
            if st.button("Process"):
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)

    elif nav_option == "Consult a Legal Expert":
        legal_expert_page()

    elif nav_option == "File Manager":
        file_manager()

    st.sidebar.markdown("User Profile")
    st.sidebar.image("https://img.icons8.com/material-outlined/24/000000/user.png", use_column_width=False)
    st.sidebar.write(f"Logged in as: **{st.session_state.get('username')}**")
    if st.sidebar.button("Logout"):
        logout_user()

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

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def legal_expert_page():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    embeddings_file = "embeddings.pkl"
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

def file_manager():
    st.header("File Manager")
    
    folders = list_folders()
    selected_folder = st.selectbox("Select a folder to view its contents", folders)

    if selected_folder:
        view_folder_contents(selected_folder)

    new_folder_name = st.text_input("Create a new folder:")
    if st.button("Create Folder"):
        if create_folder(new_folder_name):
            st.success(f"Folder '{new_folder_name}' created successfully!")
        else:
            st.error("Failed to create folder. Please try again.")

    folder_to_upload = st.selectbox("Select folder to upload files", folders)
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
                file_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
                
              
                pdf_bytes = io.BytesIO(file_obj['Body'].read())
                pdf_reader = PdfReader(pdf_bytes) 
       
                if len(pdf_reader.pages) > 0:
                    pdf_text = pdf_reader.pages[0].extract_text()
                    st.text_area("PDF Content:", pdf_text, height=300)
                else:
                    st.warning("The PDF has no pages.")
        elif file_key.endswith(('.jpg', '.jpeg', '.png')):
            st.image(s3_client.generate_presigned_url('get_object', Params={'Bucket': S3_BUCKET, 'Key': file_key}), use_column_width=True)
        elif file_key.endswith('.docx'):
            st.write("DOCX file preview not available. [Download it here](%s)" % s3_client.generate_presigned_url('get_object', Params={'Bucket': S3_BUCKET, 'Key': file_key}))

def delete_folder(folder_name):
    objects = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{folder_name}/").get('Contents', [])
    for obj in objects:
        s3_client.delete_object(Bucket=S3_BUCKET, Key=obj['Key'])
    st.success(f"Folder '{folder_name}' and its contents have been deleted.")

def list_folders():
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Delimiter='/')
    folders = [prefix['Prefix'].rstrip('/') for prefix in response.get('CommonPrefixes', [])]
    return folders

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

if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        login_or_register()
    else:
        main_app()
