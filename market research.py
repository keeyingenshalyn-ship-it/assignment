# %% [markdown]
# # Introduction 
# In this notebook, I will develop a generative AI market research assistant for business analysts. The goal is to create an intelligent system that helps conduct market research

# %% [markdown]
# # Install the packages

# %%

import streamlit as st
import pandas as pd
import numpy as np
import pickle  #for saving and loading ML models
import time
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# %%
st.title("Industry research assistant")

# %%
# --- Sidebar
# This keeps the key hidden from view and doesn't hardcode it.
with st.sidebar:
    st.title("Settings")
    user_api_key = st.text_input(
        "OpenAI API Key", 
        type="password", 
        help="The key is not stored and remains in your browser session."
    )
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
    model = st.sidebar.selectbox("Model", ["gpt-4", "gpt-3.5-turbo"])


# %% [markdown]
# ## Build the assistant 

# %%
# initialize model
max_output_tokens = 100
llm = ChatOpenAI(
    temperature=temperature,
    max_output_tokens=max_output_tokens,
    openai_api_key=user_api_key
)

# %%
# set system prompt
system_prompt = "YOUR TEXT HERE"

######
system_prompt = '''
You are a professional business analyst at a large corporation.
Your goal is to provide a concise market research report.
Strictly adhere to a limit of 500 words. "
Base your report ONLY on the provided Wikipedia data.
If you do not know the exact answer, apologize and say that you do not know.
'''

# %%
# set user prompt
user_prompt = "YOUR TEXT HERE"

######
user_prompt =f"""
I am a business analyst conducting market research on the {industry}industry. 

Based ONLY on the provided Wikipedia data below, please generate a comprehensive industry report. 

Retrieved Wikipedia Data:
{context}

Requirements:
1. The report must be less than 500 words.
2. Use professional business language.
3. Focus on key industry trends, major players, and market structure found in the text.
"""

# %%
# Main area keeps the chat
st.header("4. Chatbot")

# Session state is a dictionary that saves the data that persists across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):  # "user" or "assistant"
        st.write(msg["content"])


