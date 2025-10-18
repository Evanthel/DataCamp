

# Import required libraries
import pandas as pd
from llama_cpp import Llama
# Load the email dataset
emails_df = pd.read_csv('data/email_categories_data.csv')
# Display the first few rows of our dataset
print("Preview of our email dataset:")
emails_df.head(2)
# Set the model path
model_path = "/files-integrations/files/c9696c24-44f3-45f7-8ccd-4b9b046e7e53/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"

llm = Llama(model_path)
prompt = """
You are an AI email classification assistant.

Your task is to classify each email from the dataset 'data/email_categories_data.csv' 
into one of the following three categories:

1. Priority  — Important or time-sensitive messages that require user attention or action.
2. Updates   — Notifications or informational messages related to existing services, accounts, or subscriptions.
3. Promotions — Marketing, advertisements, or sales-related messages.

Use the subject and body text of each email to decide the most appropriate category.
"""
