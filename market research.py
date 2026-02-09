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
# --- Sidebar ---
with st.sidebar:
    st.title("Settings")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
    model_choice = st.sidebar.selectbox("Model", ["Gemini 2.5 Flash-Lite"])

# --- Build the assistant ---
# Initialize Gemini using the Secret directly
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=st.secrets["GOOGLE_API_KEY"], 
    temperature=temperature 
)
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
retriever = WikipediaRetriever()

# 2. Logic to handle results (Step 2 of your assignment)
try:
    docs = retriever.invoke(industry)
    
    # Check if we found fewer than 5 sources
    if len(docs) < 5:
        # Option A: Show a manual warning directly
        st.warning(f"Note: Only {len(docs)} relevant Wikipedia sources were found for '{industry}'.")
        
        # Option B: Or manually "raise" an error to trigger the except block below
        # raise ValueError("Insufficient data found on Wikipedia.")

    # Proceed only with what was found (max 5)
    docs = docs[:5]
    
except Exception as e:
    # This block ONLY runs if there is a real crash (e.g., no internet, API error)
    st.error(f"A system error occurred: {e}")
