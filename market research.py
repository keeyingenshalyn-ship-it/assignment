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

# --- 1. INITIALIZATION ---
# Initialize session state to prevent NameErrors
if 'is_valid' not in st.session_state:
    st.session_state.is_valid = False

# --- 2. SIDEBAR CONFIGURATION (Requirement Q0) ---
with st.sidebar:
    st.title("Settings")
    # (a) Dropdown for selecting the LLM
    model_choice = st.selectbox("Select LLM", ["gemini-1.5-flash", "gemini-1.5-pro","gemini-2.5-flash-lite","gemini-2.5-pro"])
    # (b) Text field for entering the API key
    user_api_key = st.text_input("Enter Google API Key", type="password")
    # Temperature slider for performance improvement
    temp = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.3)

st.title("Industry Research Assistant")

# --- 3. STEP 1: INDUSTRY SELECTION & GUARDRAIL ---
st.header("Step 1: Industry Selection")
industry = st.text_input("Enter the industry you wish to research:", placeholder="e.g., Fintech")

if st.button("Generate Market Report"):
    # Fix for Wikipedia crash: Ensure input is not empty
    if not industry.strip():
        st.warning("Please provide an industry name to proceed.") # Satisfies Q1
    elif not user_api_key:
        st.error("Please enter your API key in the sidebar.")
    else:
        try:
            # Initialize model safely
            llm = ChatGoogleGenerativeAI(model=model_choice, google_api_key=user_api_key, temperature=temp)
            
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

# --- 4. CONDITIONAL RENDERING (Fixes Duplicate Headings) ---
# This block only runs if Step 1 validation passes
if st.session_state.is_valid:
    st.success(f"Confirmed: '{industry}' is a valid sector.")
    
    # --- STEP 2: SOURCE RETRIEVAL ---
    st.header("Step 2: Source Retrieval")
    
    with st.spinner("Searching Wikipedia..."):
        retriever = WikipediaRetriever()
        # Return exactly 5 relevant pages
        docs = retriever.invoke(industry)[:5]
        
        if not docs:
            st.error("No Wikipedia sources found for this topic.")
        else:
            # Displaying URLs satisfies Q2 requirement
            st.subheader("Top 5 Wikipedia Sources")
            for doc in docs:
                st.write(f"- {doc.metadata['source']}")
            
            # --- STEP 3: INDUSTRY REPORT ---
            st.header("Step 3: Industry Report")
            
            with st.spinner("Generating professional report..."):
                # System Prompt defines the "Business Analyst" persona
                system_msg = (
                    "You are a professional Business Analyst. "
                    "Write an objective market research report based on the context provided. "
                    "The report must be professional and strictly under 500 words."
                )
                
                # User Prompt provides the data
                context_data = "\n\n".join([d.page_content for d in docs])
                human_msg = f"Write a report for the '{industry}' industry using this context: \n\n {context_data}"
                
                messages = [
                    SystemMessage(content=system_msg),
                    HumanMessage(content=human_msg)
                ]
                
                # Final LLM invocation
                llm = ChatGoogleGenerativeAI(model=model_choice, google_api_key=user_api_key, temperature=temp)
                report = llm.invoke(messages)
                
                st.markdown(report.content)
                
                # Word count check for Q3 compliance
                word_count = len(report.content.split())
                st.caption(f"Report Length: {word_count} words.")

# Add a Reset button to clear validation state for new searches
if st.session_state.is_valid:
    if st.button("Start New Search"):
        st.session_state.is_valid = False
        st.rerun()
