import streamlit as st
import pandas as pd
import difflib  # For fuzzy matching

# Define knowledge, operations, request cost, comparison, and anomaly data
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

comp_data = {
    "document_search_cost": """
📊 **Document Search Cost Analysis - Q1 2024**

- Your Total Spend: **$1,450.75**
- Team Average: **$890.20** (+63% above average)
- Organization Percentile: **95th**

🚩 **Key Findings:**
1. Storage costs **45% higher** than similar teams
2. Indexing operations cost **2.1x team avg**

💡 **Optimization Recommendations:**
1. **Cold Storage Archiving**  
   Potential savings: **$420/mo** (29% reduction)  
   `gsutil lifecycle set config.json gs://your-bucket`

2. **Batch Processing**  
   Reduce indexing frequency from real-time to hourly  
   Estimated savings: **$310/mo**

3. **Query Optimization**  
   23% of queries use expensive wildcard patterns  
   [See optimization guide](https://...)
""",
    "sql_gen_cost": """
📊 **SQL Generation Cost Analysis - Q1 2024**

- Your Total Spend: **$3,210.50**
- Team Average: **$2,450.30** (+31% above average)
- Organization Percentile: **88th**

🚩 **Key Findings:**
1. Frequent joins increase compute cost by **57%**
2. Data scan size **3.5x team avg**

💡 **Optimization Recommendations:**
1. **Materialized Views**  
   Precompute frequent queries  
   Expected savings: **$780/mo**

2. **Partitioned Tables**  
   Reduce scan overhead by 41%  
   `ALTER TABLE my_table SET OPTIONS (partition_expiration_days=30);`

3. **Index Optimization**  
   Improve performance of repetitive queries  
   [See indexing guide](https://...)
""",
    "pipeline_cost": """
📊 **ML Pipeline Cost Analysis - Q1 2024**

- Your Total Spend: **$5,675.80**
- Team Average: **$4,210.90** (+34% above average)
- Organization Percentile: **91st**

🚩 **Key Findings:**
1. GPU instances idle **28% of the time**
2. Model training iterations **2.8x team avg**

💡 **Optimization Recommendations:**
1. **Auto-scaling Nodes**  
   Reduce idle GPU costs  
   Expected savings: **$950/mo**

2. **Efficient Model Checkpoints**  
   Reduce redundant training runs  
   `export MODEL_CHECKPOINT_FREQUENCY=5`

3. **Preemptible VMs**  
   Use low-cost spot instances  
   [See GCP Preemptible VMs](https://...)
""",
    "storage_cost": """
📊 **Cloud Storage Cost Analysis - Q1 2024**

- Your Total Spend: **$980.45**
- Team Average: **$710.30** (+38% above average)
- Organization Percentile: **89th**

🚩 **Key Findings:**
1. Large unaccessed datasets consuming **62% of storage**
2. Redundant backups **2.3x team avg**

💡 **Optimization Recommendations:**
1. **Auto-delete Policies**  
   Remove old logs automatically  
   `gsutil lifecycle set config.json gs://your-bucket`

2. **Cold Storage for Rarely Used Data**  
   Migrate to Archive storage  
   `gcloud storage buckets update --storage-class ARCHIVE`

3. **Deduplicating Backups**  
   Compress redundant logs  
   [Backup Optimization Guide](https://...)
"""
}

anomaly_data = {
    "document_search_anomaly": """
📊 **Cost Anomaly - DOC_SEARCH**

- request_id: **DOC-003**
- actual_cost: **$10.00** (+1076.47% more than last month)
- predicted_cost: **$4.80**
- reason: **Actual cost is 2.08x higher than predicted for the given size and duration.**
"""
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

# Function to check if the question is a comparison request
def check_comparison(user_input, comp_data):
    if "compare" in user_input.lower() or "cost analysis" in user_input.lower():
        for key in comp_data.keys():
            if key in user_input:
                return key
    return None

# Function to check if the question is an anomaly request
def check_anomaly(user_input, anomaly_data):
    if "anomaly" in user_input.lower():
        for key in anomaly_data.keys():
            if key.split("_anomaly")[0] in user_input.lower():
                return key
    return None

# Page Configuration
st.set_page_config(page_title="AIPlatform - Model Dashboard", layout="wide")

# Layout
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

        **Cost Comparison Requests**  
        - Compare document search cost  
        - Compare SQL execution cost  
        - Compare pipeline costs  

        **Anomaly Detection Queries**  
        - Anomaly in document search cost  
        """, unsafe_allow_html=True)

    st.markdown("### 💬 Agent Help")
    user_question = st.text_input("Ask your question here:", key="user_input")

    if user_question:
        response = "🤔 Sorry, I don't have an answer for that. Please check the documentation or contact support."

        anomaly_match = check_anomaly(user_question, anomaly_data)
        if anomaly_match:
            response = anomaly_data[anomaly_match]
        else:
            comp_match = check_comparison(user_question, comp_data)
            if comp_match:
                response = comp_data[comp_match]
            else:
                cost_match = check_request_cost(user_question, request_cost_data)
                if cost_match:
                    response = request_cost_data[cost_match]
                else:
                    op_match = find_operation_match(user_question, operations_data)
                    if op_match:
                        response = operations_data[op_match]
                    else:
                        kb_match = find_best_match(user_question, knowledge_data)
                        if kb_match:
                            response = knowledge_data[kb_match]

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

with tab1:
    st.write("#### Active Models")
    st.dataframe(df, width=1200, height=200)

with tab2:
    st.write("#### Archive Models")
    st.write("No archived models.")

with tab3:
    st.write("#### All Models")
    st.dataframe(df, width=1200, height=200)

st.button("+ Add model", key="add_model", help="Click to add a new model")

st.markdown("---")
st.write("© AIPlatform | Documentation | Reporting | Support")