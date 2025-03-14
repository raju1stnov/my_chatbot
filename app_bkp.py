import streamlit as st
import pandas as pd
import difflib  # For fuzzy matching

# Define knowledge and operations data
knowledge_data = {
    "How to search for a document?": "Use the `/docsearch` API endpoint with your query parameters. Check our GitHub repo's 'search-examples' folder for sample code.",
    "How to register a model?": "1. Navigate to Model Registry UI 2. Click 'Register New Model' 3. Fill in metadata 4. Upload model artifacts. [Example code](github.com/mlops/models#registration)",
    "How to perform an SQL generation operation?": "Use the `sqlgen.generate()` method with your natural language query. Ensure you have access to the EDM database. [Docs](github.com/mlops/sqlgen#usage)",
    "How to trigger a pipeline?": "Use the `PipelineRunner` class from our SDK. Required parameters: config_path and dataset_version. [Example](github.com/mlops/pipelines#triggering)",
    "Where to find pipeline templates?": "All templates are in the `pipeline-templates` repo under 'vertexai-templates' directory.",
    "How to monitor pipeline runs?": "Access the Pipeline Dashboard UI or use the `get_pipeline_status(pipeline_id)` SDK method.",
    "What authentication is needed for model registry?": "You need GCP credentials with 'mlops-model-editor' role. [See setup guide](github.com/mlops/auth#model-registry)",
    "How to format training data?": "Data must be in TFRecord format with schema defined in protobufs. [Schema examples](github.com/mlops/data#formatting)",
    "Where are the API docs hosted?": "All Swagger docs are available at api.mlops-platform.com/swagger",
    "How to contribute new features?": "1. Fork the repo 2. Create feature branch 3. Submit PR with tests. [Contribution guide](github.com/mlops/contributing)"
}

operations_data = {
    "PCI-1234": "Status: Failed ❌\nError: MemoryLimitExceeded\nLast Updated: 2m ago\nSuggested Fix: Increase memory allocation to 8GB in pipeline config",
    "DOC-5678": "Status: Completed ✅\nProcessed 1,234 documents\nDuration: 12m 34s\nOutput: gs://processed-docs/5678/",
    "MOD-9012": "Status: Pending ⏳\nIn queue position: 3\nEstimated start: 15m\nTrack at registry.mlops/status/9012",
    "SQL-3456": "Status: Running 🏃\nGenerated 45 queries\nProgress: 78%\nETA: 3m",
    "TRAIN-7890": "Status: Completed ✅\nAccuracy: 92.4%\nArtifacts: gs://models/v12.3\nLogs: ops.mlops/train-7890",
    "INGEST-123": "Status: Failed ❌\nError: InvalidSchema\nFailed at: Data validation step\nContact: data-team@mlops",
    "VALID-456": "Status: Running 🏃\nProcessed: 56k records\nThroughput: 1.2k rec/s\nETA: 8m",
    "DEPLOY-789": "Status: Rolling Out 🌐\nCurrent version: v3.4\nTarget: 100% by 15:00 UTC\nHealth: 98.7% success",
    "INDEX-012": "Status: Completed ✅\nIndexed 89k documents\nSize: 45GB\nOptimized: true",
    "SEARCH-345": "Status: Failed ❌\nError: Timeout\nRetry Attempt: 2/3\nNext try: 2m"
}

# Function to check if the question contains any operation ID
def find_operation_match(user_input, operations_dict):
    for key in operations_dict.keys():
        if key in user_input:  # If the key exists in the user input
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
col1, col2 = st.columns([2, 1])  # Left column is wider

with col1:
    st.markdown("<h1 style='color: #D62728;'>AIPlatform</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #B22222;'>Model Dashboard</h2>", unsafe_allow_html=True)
    st.write("### Welcome, Kaushik!")

with col2:
    # Expander for MLOps Assistant Agent
    with st.expander("ℹ️ **MLOps Assistant Agent - Click to Expand**"):
        st.markdown("""
        Use the [MLOps Assistant Agent](#) for various queries and troubleshooting related to model operations. Below are some key areas where it can assist:

        #### 🔍 **General Queries**
        1. **How to search for a document?**  
        2. **How to register a model?**  
        3. **How to perform an SQL generation operation?**  
        4. **How to trigger a pipeline?**  

        #### 🛠️ **Troubleshooting Assistance**
        1. **What is the current state of my `training_pipeline`?**  
        2. **Was my `document_search` run successful?**  
        3. **Why is my model registration pending?**  
        4. **How to debug a failed pipeline execution?**  
        """, unsafe_allow_html=True)

    # Chat Help Section
    st.markdown("### 💬 Agent Help")
    user_question = st.text_input("Ask your question here:", key="user_input")

    if user_question:
        # First check if any operations_data key is in the user input
        operation_match = find_operation_match(user_question, operations_data)
        if operation_match:
            response = operations_data[operation_match]
        else:
            # If no operation key found, search in knowledge_data
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

# Archive & All Tabs Placeholder
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
