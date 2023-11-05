# Second Brain App to store notes, ideas, tasks and schedules
#
# Idea:
# A normal Chat-Bot like Application with chating like ChatGPT
# with extra funtions to get information from a vector database and store
# information to a vector database
#
# Functions
# Streamlit Chat
# Store information information into a file and index it to a vector 
# Ask for information out of a vector
#

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import openai
import streamlit as st
import json

from datetime import datetime
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

#------------------------------------------------------------------------------
# OpenAI Settings
#------------------------------------------------------------------------------

openai.api_key = st.secrets["OPENAI_API_KEY"]
embedding_model = st.secrets["OPENAI_API_EMBEDDING_MODEL"]
gpt_model = st.secrets["OPENAI_MODEL"]

#------------------------------------------------------------------------------
# GPT Function Descriptions
#------------------------------------------------------------------------------    

function_descriptions = [
    {
        "name": "store_information",
        "description": "Storing information to vector",
        "parameters": {
            "type": "object",
            "properties": {
                "information": {
                    "type": "string",
                    "description": "The information to store in vector",
                },
            },
            "required": ["information"],
        },
    },
    {
        "name": "get_information_from_memory_db",
        "description": "Reveive Information from Vector",
        "parameters": {
            "type": "object",
            "properties": {
                "information": {
                    "type": "string",
                    "description": "Question to ask on the vector db",
                },
            },
            "required": ["information"],
        },
    },
]

#------------------------------------------------------------------------------
# Function Call: Store Information
#------------------------------------------------------------------------------
def store_information(information):

    # Definitions for data storing
    file_path = "data/memory.txt"
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

    # Storing content into memory.txt
    # IMPROVEMENTS could be: Cloud, SQL DB
    with open(file_path, 'a') as f:
        f.write(f"{date_time};{information}\n")
    
    # Indexing the memory.txt into Vector
    loader = TextLoader(file_path, encoding="latin1")
    docs = loader.load_and_split()
    embedding = OpenAIEmbeddings(
        model=embedding_model,
        chunk_size=1
    )

    # Storing the Vector
    # IMPROVEMENTS: Cloud
    db = FAISS.from_documents(documents=docs, embedding=embedding)
    db.save_local("data/memoryDB")

    return "Information stored"

#------------------------------------------------------------------------------
# Function Call: Get Information
#------------------------------------------------------------------------------

def get_information_from_memory_db(information):

    # Large Language Model
    llm = ChatOpenAI(
        model=gpt_model,
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        temperature=0.0
    )

    # Embeddings
    embedding = OpenAIEmbeddings(
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        model=embedding_model,
        chunk_size=1
    )

    # Used Vectorstore
    vector_store = FAISS.load_local("data/memoryDB", embedding)

    # QA Retrievel
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Execute Query and store
    responese = qa({"query": information})
    answer = responese["result"]

    return answer   

#------------------------------------------------------------------------------
# OpenAI GPT Processing
#------------------------------------------------------------------------------

def request_gpt(user_prompt, message_chain):

# General GPT processing whith included function calling
# Three Ways are possible:
#   1. General Conversation
#   2. Retrieve Information from Vectorstore
#   3. Store information into Vectorstore

    # stored temporary for function call
    information = user_prompt

    # GPT Call
    completion = openai.ChatCompletion.create(
        model=gpt_model,
        messages=[      
            {"role": m["role"], "content": m["content"]}
            for m in message_chain
        ],
        functions=function_descriptions,
        function_call="auto",
    )
    output = completion.choices[0].message

    return output

def get_answer_from_gpt(output, prompt):

# If a function was called the the content of the choosen function
# will be used to get the clear output from GPT about what it was doing

    # prep. of answer
    params = json.loads(output.function_call.arguments)
    chosen_function = eval(output.function_call.name)
    the_answer = chosen_function(**params)

    # The key is to add the function output back to the messages with role: function
    second_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "function", "name": output.function_call.name, "content": the_answer},
        ],
        functions=function_descriptions,
    )

    # None Handling
    if second_completion.choices[0].message.content is not None:
        response = second_completion.choices[0].message.content
    else:
        response = "Information stored: " + prompt

    return response

#------------------------------------------------------------------------------
# MAIN PROCESSING

st.set_page_config(
    page_title="BrainGPT App",
    page_icon="assets/favicon.png",
    #layout="wide"
)

#------------------------------------------------------------------------------
# Chat Processing Streamlit
#------------------------------------------------------------------------------

system_message = """
You are my personal Task and Notes Assistant. We can eiter just chat, or I can ask for some information from the memory database with functions: get_information_from_memory_db, or give you new information, and you use this: store_information. For any kind of tasks use the prefix: Task, and for any kind of notes use the prefix: Notes.
Before you execute the function to store a information, make sure to gather some more details about the task or note to capture more details to store.
"""

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = gpt_model

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": system_message})

for message in st.session_state.messages:
    if message["role"] != "system":    
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input(" "):

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar="👨‍🦱"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧛‍♂️"):

        message_chain = st.session_state.messages
        #print(message_chain)

        message_placeholder = st.empty()
        full_response = ""

        # executing the request
        output = request_gpt(prompt, message_chain)
        #print(output)

        # check if we got a function call or just a conversation
        if output['content'] is None:
            full_response = get_answer_from_gpt(output, prompt)
        else:
            full_response = output['content'] 
            
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})