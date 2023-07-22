import streamlit as st
import base64
import textract
from streamlit_extras.colored_header import colored_header
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message
from sidebar import side_bar
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter


st.set_page_config(
    page_title="Document QA Chatbot",
    page_icon="🧊",
    layout="wide"
)

side_bar()

st.header(":open_book: Document QA Chatbot")
st.markdown(
    ":arrow_forward:  **Document QA chatbot** is an intelligent virtual assistant that leverages NLP and ML to provide accurate and efficient answers to users' questions about specific documents. By analyzing the content, structure, and context of documents, the chatbot can extract relevant information and deliver prompt responses, saving users' time and effort in searching through lengthy texts.\n"
    "\n"
    ":arrow_forward:  To show you how this Document QA chatbot :robot_face: works, I will be using paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) as the document we want to retrieve information from, and build a interactive chatbot on top of it!  "
)

# Upload the document
st.markdown("##### **Step 1:** Please upload the document you want to interact with below:")
st.text("")

File = st.file_uploader(label = "Upload file", type=["pdf","docx"])
    # Submit = st.form_submit_button(label='Submit')

if File:
    st.markdown("The file is sucessfully uploaded.")
    st.markdown("It may take a while to preprocess the document...")
    
    save_path = "Sample_Docs/" + File.name
    doc = textract.process(save_path)
    with open("tmp.txt", 'w') as f:
        f.write(doc.decode('utf-8'))

    text = doc.decode('utf-8')
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 524,
        chunk_overlap  = 0,
        length_function = count_tokens,
    )
    chunks = text_splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.5), db.as_retriever())

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hi! How may I help you?"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ['Hi!']

    st.markdown("##### **Step 2:** Type in any questions regarding the document you just uploaded:")
    st.text("")
    
    input_container = st.container()
    colored_header(label='', description='', color_name='blue-30')
    response_container = st.container()

    # User input
    ## Function for taking user provided prompt as input
    def get_text():
        input_text = st.text_input("Ask questions here: ", "", key="input")
        return input_text

    ## Applying the user input box
    
    with input_container:
        user_input = get_text()

    # Response output
    ## Function for taking user prompt as input followed by producing AI generated responses
    chat_history = []
    def generate_response(prompt):
        result = qa({"question": user_input, "chat_history": chat_history})
        chat_history.append((user_input, result['answer']))
        return result['answer']

    with response_container:
        if user_input:
            response = generate_response(user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response)

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="adventurer")
                message(st.session_state['generated'][i], key=str(i))