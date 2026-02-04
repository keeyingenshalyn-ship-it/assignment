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
retriever = WikipediaRetriever()
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# %%
st.title("Industry research assistant")

# %%
# --- Sidebar
# This keeps the key hidden from view and doesn't hardcode it.
with st.sidebar:
    st.title("Settings")
    user_api_key = st.text_input(
        "API Key", 
        type="password", 
        help="The key is not stored and remains in your browser session."
    )
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
    model = st.sidebar.selectbox("Model", ["gpt-3.5-turbo"])


# %% [markdown]
# ## Build the assistant 
# Initialize Gemini
llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=user_api_key,
            temperature=0.3) #

# --- STEP 1: Industry Input & Validation ---
industry = st.text_input("Enter an industry to research:")

if st.button("Generate Report"):
    if not industry:
        st.warning("Please provide an industry to proceed.")
    else:
        # --- STEP 2: Wikipedia Search [cite: 48] ---
        with st.spinner("Searching Wikipedia..."):
            docs = retriever.invoke(industry)
            # Take top 5 as required
            top_5_docs = docs[:5]
            
            st.subheader("Top 5 Wikipedia Sources")
            for doc in top_5_docs:
                st.write(f"- {doc.metadata['source']}")

        # --- STEP 3: Report Generation ---
        with st.spinner("Generating Industry Report..."):
            # Combine content from the 5 docs
            context = "\n\n".join([doc.page_content for doc in top_5_docs])
            
            prompt = ChatPromptTemplate.from_template("""
            You are a business analyst. Based on the following Wikipedia data, 
            write a market research report for the {industry} industry. 
            The report must be less than 500 words.
            
            Data: {context}
            """)
            chain = prompt | llm
            report = chain.invoke({"industry": industry, "context": context})
            
            st.subheader("Industry Report")
            st.write(report.content)
            
            # Check word count 
            word_count = len(report.content.split())
            st.caption(f"Word count: {word_count}")

try:
    report = chain.invoke({"industry": industry, "context": context})
    st.write(report.content)
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Tip: Check your OpenAI quota or try again in a moment.")
