

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
def process_message(llm, message, prompt):
    response = llm(prompt, max_tokens= 5, temperature=0)
    return response['choices'][0]['text'].strip()
test_emails = emails_df.head(2)
results = []
for idx, row in test_emails.iterrows():
    email_content = row['email_content']
    expected_category = row['expected_category']
    
    # Get model's classification
    result = process_message(llm, email_content, prompt)
    
    # Store results
    results.append({
        'email_content': email_content,
        'expected_category': expected_category,
        'model_output': result
    })
results_df = pd.DataFrame(results)

result1 = results_df['model_output'].iloc[0]
result2 = results_df['model_output'].iloc[1]

print(f"Result 1: `{result1}`\nResult 2: `{result2}`")