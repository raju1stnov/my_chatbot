import streamlit as st
import pandas as pd
import difflib  # For fuzzy matching

# Define knowledge, operations, and request cost data
knowledge_data = {
    "How to search for a document?": "Use the `/docsearch` API endpoint with your query parameters. Check our GitHub repo's 'search-examples' folder for sample code.",
    "How to register a model?": "1. Navigate to Model Registry UI 2. Click 'Register New Model' 3. Fill in metadata 4. Upload model artifacts. [Example code](github.com/mlops/models#registration)",
    "How to perform an SQL generation operation?": "Use the `sqlgen.generate()` method with your natural language query. Ensure you have access to the EDM database. [Docs](github.com/mlops/sqlgen#usage)",
    "How to trigger a pipeline?": "Use the `PipelineRunner` class from our SDK. Required parameters: config_path and dataset_version. [Example](github.com/mlops/pipelines#triggering)",
}

operations_data = {
    "PCI-1234": "Status: Failed ❌\nError: MemoryLimitExceeded\nLast Updated: 2m ago\nSuggested Fix: Increase memory allocation to 8GB in pipeline config",
    "DOC-5678": "Status: Completed ✅\nProcessed 1,234 documents\nDuration: 12m 34s\nOutput: gs://processed-docs/5678/",
    "MOD-9012": "Status: Pending ⏳\nIn queue position: 3\nEstimated start: 15m\nTrack at registry.mlops/status/9012",
    "SQL-3456": "Status: Running 🏃\nGenerated 45 queries\nProgress: 78%\nETA: 3m",
}

request_cost_data = {
    "DOC-5678": "📊 **Cost & Token Details:**\n- Token Usage: **12,500 tokens**\n- Execution Cost: **$0.45**",
    "SQL-3456": "📊 **Cost & Token Details:**\n- Token Usage: **9,800 tokens**\n- Execution Cost: **$0.30**",
    "SEARCH-345": "📊 **Cost & Token Details:**\n- Token Usage: **15,200 tokens**\n- Execution Cost: **$0.55**",
    "MOD-9012": "📊 **Cost Details:**\n- Execution Cost: **$2.10**",
}

# Function to check if the question mentions cost or tokens
def check_request_cost(user_input, cost_data):
    if "cost" in user_input.lower() or "token" in user_input.lower():
        for key in cost_data.keys():
            if key in user_input:
                return key
    return None

# Function to check if the question contains any operation ID
def find_operation_match(user_input, operations_dict):
    for key in operations_dict.keys():
        if key in user_input:
            return key
    return None

# Function to find the best matching key in knowledge data
def find_best_match(user_input, data_dict):
    if not user_input:
        return None
    matches = difflib.get_close_matches(user_input, data_dict.keys(), n=1, cutoff=0.5)
    return matches[0] if matches else None

# Page Configuration
st.set_page_config(page_title="AIPlatform - Model Dashboard", layout="wide")

# Two Column Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<h1 style='color: #D62728;'>AIPlatform</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #B22222;'>Model Dashboard</h2>", unsafe_allow_html=True)
    st.write("### Welcome, Kaushik!")

with col2:
    with st.expander("MLOps Assistant Agent - Click to Expand"):
        st.markdown("<span style='color: red; font-weight: bold;'>MLOps Assistant Agent</span>", unsafe_allow_html=True)
        st.markdown("""
        **General Queries**  
        - How to search for a document?  
        - How to register a model?  
        - How to perform an SQL generation operation?  
        - How to trigger a pipeline?  

        **Troubleshooting Assistance**  
        - What is the current state of my `training_pipeline`?  
        - Was my `document_search` run successful?  
        - Why is my model registration pending?  
        - How to debug a failed pipeline execution?  

        **Request Cost Queries**  
        - What is the number of token usage for a request_id?  
        - What is the execution cost of the request request_id?  
        """, unsafe_allow_html=True)

    # Chat Help Section
    st.markdown("### 💬 Agent Help")
    user_question = st.text_input("Ask your question here:", key="user_input")

    if user_question:
        cost_match = check_request_cost(user_question, request_cost_data)
        if cost_match:
            response = request_cost_data[cost_match]
        else:
            operation_match = find_operation_match(user_question, operations_data)
            if operation_match:
                response = operations_data[operation_match]
            else:
                best_match_knowledge = find_best_match(user_question, knowledge_data)
                if best_match_knowledge:
                    response = knowledge_data[best_match_knowledge]
                else:
                    response = "🤔 Sorry, I don't have an answer for that. Please check the documentation or contact support."

        st.markdown(f"**Response:**\n\n{response}")

# Tabs for Active, Archive, and All
tab1, tab2, tab3 = st.tabs(["Active (94)", "Archive (2)", "All (96)"])

# Dummy Data for Table
data = {
    "Model Name": [
        "abcdefgh10a1-dc87-4102-bd3b-abcdefgh10a1",
        "bcdefghidc071e60-f701-430a-b1b2-bcdefghidc071e60",
        "cdefghijb56dc637-9e29-4a98-8e98-cdefghijb56dc637"
    ],
    "Model Owner(s)": ["Kaushik Das", "Kaushik Das", "Kaushik Das"],
    "Version": ["1.0.1", "1.0.1", "1.0.1"],
    "Model Repo": ["No link available", "No link available", "No link available"],
    "Status": ["Pending", "Pending", "Pending"]
}

df = pd.DataFrame(data)

# Display Table in "Active" Tab
with tab1:
    st.write("#### Active Models")
    st.dataframe(df, width=1200, height=200)

with tab2:
    st.write("#### Archive Models")
    st.write("No archived models.")

with tab3:
    st.write("#### All Models")
    st.dataframe(df, width=1200, height=200)

# Add Model Button
st.button("+ Add model", key="add_model", help="Click to add a new model")

# Footer
st.markdown("---")
st.write("© AIPlatform | Documentation | Reporting | Support")
