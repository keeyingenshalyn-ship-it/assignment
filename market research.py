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
# --- SIDEBAR (Requirement Q0) ---
with st.sidebar:
    st.title("Settings")
    # (a) Dropdown for selecting the LLM
    model_choice = st.selectbox("Select LLM", ["gemini-1.5-flash", "gemini-1.5-pro"])
    # (b) Text field for entering our API key
    user_api_key = st.text_input("Enter Google API Key", type="password")
    
    # Optional but recommended for Q5
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.3)

# --- INITIALIZATION ---
# Initialize session state variables to fix NameErrors
if 'is_valid' not in st.session_state:
    st.session_state.is_valid = False

# --- STEP 1: INDUSTRY SELECTION & VALIDATION ---
st.header("Step 1: Industry Selection")
industry = st.text_input("Enter the industry you wish to research:", placeholder="e.g., Renewable Energy")

# Validation Function (The Guardrail)
def validate_industry(llm, query):
    # A simple, low-cost prompt to check intent
    prompt = f"Is '{query}' a legitimate business industry or sector? Answer only 'Yes' or 'No'."
    response = llm.invoke(prompt).content.strip().lower()
    return "yes" in response

# Execution Trigger
is_valid = False
if st.button("Generate Market Report"):
    if not industry.strip():
        # Q1 Requirement: Ask for update if empty
        st.warning("Please provide an industry name to proceed.")
    else:
        # Run Step 1 Validation
        with st.spinner("Validating industry intent..."):
            is_valid = validate_industry(llm, industry)
            
        if is_valid:
            st.success(f"Confirmed: '{industry}' is a valid sector.")
            
            # --- Proceed to STEP 2: Wikipedia Retrieval ---
            st.header("Step 2: Source Retrieval")
            # (Your retriever code goes here)
            
        else:
            # Handle invalid input like "hello"
            st.error(f"'{industry}' does not appear to be a valid business industry. Please try a more specific term.")

if is_valid:
    st.success(f"Confirmed: '{industry}' is a valid sector.")
    
    # --- STEP 2: SOURCE RETRIEVAL ---
    st.header("Step 2: Source Retrieval")
    
    with st.spinner("Searching Wikipedia for industry data..."):
        try:
            retriever = WikipediaRetriever()
            # Return exactly 5 relevant pages
            docs = retriever.invoke(industry)[:5]
            
            if len(docs) == 0:
                st.error("No relevant Wikipedia pages found. Please refine your industry name.")
            else:
                if len(docs) < 5:
                    st.warning(f"Note: Only {len(docs)} sources were found. 5 are typically required.")
                
                st.subheader("Top 5 Wikipedia Sources")
                # Displaying URLs satisfies Q2 requirement
                for doc in docs:
                    st.write(f"- {doc.metadata['source']}")
                
                # Store content for Step 3
                context_data = "\n\n".join([d.page_content for d in docs])
                
        except Exception as e:
            st.error(f"Retrieval error: {e}")
    
    # --- STEP 3: MARKET RESEARCH REPORT ---
    st.header("Step 3: Industry Report")
    
    with st.spinner("Synthesizing market report..."):
        # Prompt engineering to ensure persona and word count compliance
        report_prompt = (
            f"You are a professional Business Analyst. Write a market research report "
            f"for the '{industry}' industry based on this context: {context_data}. "
            f"The report must be professional and strictly under 500 words."
        )
        
        try:
            # Generate the report
            report = llm.invoke(report_prompt)
            st.markdown(report.content)
            
            # Simple word count check to ensure Q3 compliance
            word_count = len(report.content.split())
            st.caption(f"Report Length: {word_count} words.")
            
        except Exception as e:
            st.error(f"Generation error: {e}")
