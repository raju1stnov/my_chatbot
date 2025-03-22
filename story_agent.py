'''
StoryGenerator: Uses Gemini to create a short story idea.
SentimentAnalyzer: Analyzes the sentiment of the story idea.
TitleSuggester: Suggests a title based on the story idea.

set AUTOGEN_USE_DOCKER=False
python story_agent.py
'''

import os
from dotenv import load_dotenv
import google.generativeai as genai
from autogen import AssistantAgent, UserProxyAgent

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("OOGLE_API_KEY")

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Define the Gemini model
model = genai.GenerativeModel("gemini-pro")

# AutoGen configuration for Gemini API
config_list = [
    {
        "model": "gemini-pro",
        "api_key": GEMINI_API_KEY,
        "api_type": "google"
    }
]

# Task 1: Story Idea Generator Agent
story_generator = AssistantAgent(
    name="StoryGenerator",
    llm_config={"config_list": config_list},
    system_message="You are a creative writer. Generate a short story idea in 2-3 sentences."
)

# Task 2: Sentiment Analyzer Agent
sentiment_analyzer = AssistantAgent(
    name="SentimentAnalyzer",
    llm_config={"config_list": config_list},
    system_message="You are a sentiment analysis expert. Analyze the sentiment of the given text (positive, negative, neutral) and explain why."
)

# Task 3: Title Suggester Agent
title_suggester = AssistantAgent(
    name="TitleSuggester",
    llm_config={"config_list": config_list},
    system_message="You are a title creation expert. Suggest a catchy title based on the story idea provided."
)

# User Proxy to interact with agents
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    code_execution_config={"work_dir": "coding", "use_docker": False},  # Disable Docker
)

# Function to run the workflow
def run_story_workflow():
    # Step 1: Generate story idea
    story_idea = user_proxy.initiate_chat(
        story_generator,
        message="Generate a short story idea."
    )
    story_text = story_generator.last_message()["content"]
    print(f"Story Idea: {story_text}\n")

    # Step 2: Analyze sentiment
    sentiment_result = user_proxy.initiate_chat(
        sentiment_analyzer,
        message=f"Analyze the sentiment of this text: '{story_text}'"
    )
    sentiment_text = sentiment_analyzer.last_message()["content"]
    print(f"Sentiment Analysis: {sentiment_text}\n")

    # Step 3: Suggest a title
    title_result = user_proxy.initiate_chat(
        title_suggester,
        message=f"Suggest a title for this story idea: '{story_text}'"
    )
    title_text = title_suggester.last_message()["content"]
    print(f"Suggested Title: {title_text}")

# Run the workflow
if __name__ == "__main__":
    run_story_workflow()