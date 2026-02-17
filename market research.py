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
from langchain_core.messages import SystemMessage, HumanMessage

# --- 1. INITIALIZATION ---
# Initialize session state to prevent NameErrors
if 'is_valid' not in st.session_state:
    st.session_state.is_valid = False

# --- 2. SIDEBAR CONFIGURATION (Requirement Q0) ---
with st.sidebar:
    st.title("Settings")
    # (a) Dropdown for selecting the LLM
    model_choice = st.selectbox("Select LLM", ["gemini-2.5-flash-lite"])
    # (b) Text field for entering the API key
    user_api_key = st.text_input("Enter Google API Key", type="password")
    # Temperature for performance improvement
    fixed_temperature = 0.3
st.title("Industry Research Assistant")

# --- 3. STEP 1: INDUSTRY SELECTION & GUARDRAIL ---
st.header("Step 1: Industry Selection")
industry = st.text_input("Enter the industry you wish to research:", placeholder="e.g., Fintech")

if st.button("Generate Market Report"):
    # FIX for srsearch error: Never search with empty or whitespace input
    if not industry.strip():
        st.warning("Please provide an industry name to proceed.")
        st.session_state.is_valid = False
    elif not user_api_key:
        st.error("Please enter your API key in the sidebar.")
    else:
        try:
            # Initialize model safely
            llm = ChatGoogleGenerativeAI(model=model_choice, google_api_key=user_api_key, temperature=fixed_temperature)
            
            # LLM Guardrail Validation
            # Use a low-cost check to filter nonsense like "hello"
            check_prompt = f"Is '{industry}' a valid business sector or industry? Answer only 'Yes' or 'No'."
            response = llm.invoke(check_prompt).content.lower()
            
            if "yes" in response:
                st.session_state.is_valid = True
            else:
                st.session_state.is_valid = False
                st.error(f"'{industry}' is not recognized as a valid industry. Please try again.")
        except Exception as e:
            st.error(f"Validation Error: {e}")

if st.session_state.is_valid and industry.strip():
    st.success(f"Confirmed: '{industry}' is a valid sector.")
    
    # --- STEP 2: SOURCE RETRIEVAL ---
    st.header("Step 2: Source Retrieval")
    try:
        retriever = WikipediaRetriever()
        # Using .strip() prevents the srsearch parameter error
        docs = retriever.invoke(industry.strip())
        
        if not docs:
            st.error("No sources found.")
        else:
            docs = docs[:5]
            st.subheader(f"Top Wikipedia Sources ({len(docs)})")
            for doc in docs:
                st.write(f"- {doc.metadata['source']}")
    except Exception as e:
        st.error(f"Retrieval error: {e}")

# --- STEP 3: INDUSTRY REPORT GENERATION ---
if st.session_state.is_valid:
    st.header("Step 3: Industry Report")
    
    with st.spinner("Synthesizing market insights..."):
        # Explicit System Prompt to enforce constraints
        system_message = (
            "You are a professional Business Analyst. "
            "Write a market research report based ONLY on the provided context. "
            "CRITICAL REQUIREMENT: The report must be under 500 words. "
            "Structure: Use professional headings (e.g., Market Overview, Key Trends)."
        )

        # User Prompt providing the Wikipedia data
        context_text = "\n\n".join([d.page_content for d in docs])
        user_message = f"Write a report for the '{industry}' industry using this context: \n\n {context_text}"

        try:
            # Combining prompts for the Gemini model
            from langchain_core.messages import SystemMessage, HumanMessage
            llm_gen = ChatGoogleGenerativeAI(model=model_choice, google_api_key=user_api_key, temperature=temp)
            report = llm_gen.invoke([
                SystemMessage(content=system_message),
                HumanMessage(content=user_message)
            ])
            
            st.markdown(report.content)
            
            # --- WORD COUNT VALIDATION ---
            # This demonstrates "critical awareness" for your reflection
            actual_word_count = len(report.content.split())
            if actual_word_count > 500:
                st.warning(f"Note: Report is {actual_word_count} words. Please refine your prompt.")
            else:
                st.caption(f"Success: Report length is {actual_word_count} words.")

        except Exception as e:
            st.error(f"Error during report generation: {e}")

# Add a Reset button to clear validation state for new searches
if st.session_state.is_valid:
    if st.button("Start New Search"):
        st.session_state.is_valid = False
        st.rerun()
