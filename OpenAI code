pip install openai 

import openai
import pandas as pd

# Load training data
df = pd.read_csv('training_data.csv')

# Define OpenAI API key
openai.api_key = "YOUR_API_KEY"

# Define a function to classify severity
def classify_severity(deficiency_text):
    prompt = f"""
    You are an expert in classifying deficiencies based on severity levels.
    Here are some examples:

    1. "{df.iloc[0]['Deficiency Text']}" -> {df.iloc[0]['Consensus Severity']}
    2. "{df.iloc[1]['Deficiency Text']}" -> {df.iloc[1]['Consensus Severity']}
    3. "{df.iloc[2]['Deficiency Text']}" -> {df.iloc[2]['Consensus Severity']}
    
    Based on the above examples, classify the following deficiency:
    "{deficiency_text}"
    """

    # Use OpenAI GPT model to classify
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50
    )

    # Extract and return severity from the response
    return response.choices[0].text.strip()

# Test the function with a new description
new_description = "Description of a new deficiency that involves critical issues."
severity = classify_severity(new_description)
print(f"Predicted Severity: {severity}")
