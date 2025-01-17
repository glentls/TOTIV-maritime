from openai import OpenAI
import pandas as pd
import openpyxl
import numpy as np
import faiss
import streamlit as st
import os
from sklearn.model_selection import train_test_split
from google.colab import files

uploaded = files.upload()
# Load your dataset (replace 'your_file.csv' with your actual dataset file)
df = pd.read_csv(r"consensus_severity_results.zip")

# Split the dataset into training and validation sets
train, validation = train_test_split(df, test_size=0.2, random_state=42)

# Save the training set to a CSV file
train.to_csv('train.csv', index=False)

# Save the validation set to a CSV file
validation.to_csv('validation.csv', index=False)

# Download the files to your local machine
from google.colab import files
files.download('train.csv')
files.download('validation.csv')

# Load training data
df = pd.read_csv('train.csv')
df2 = pd.read_csv(r"Datasets/psc_severity_test.csv")
workbook = openpyxl.load_workbook(r"Datasets/submission_template.xlsx")
sheet = workbook.active

# Define OpenAI API key
client = OpenAI(
  api_key="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"
)

# Define a function to classify severity
def classify_severity(deficiency_text):
    prompt = "You are an expert in classifying deficiencies based on severity levels. " + \
        "\n Here are some examples:" + \
            "\n 1. " + df.iloc[0]['def_text'] + " -> " + df.iloc[0]['Consensus Severity'] + \
                "\n 2. " + df.iloc[1]['def_text'] + " -> " + df.iloc[1]['Consensus Severity'] + \
                    "\n 3. " + df.iloc[2]['def_text'] + " -> " + df.iloc[2]['Consensus Severity'] + \
                        "Based on the above examples, classify the following deficiency: " + \
                            deficiency_text + "in one word: Low, Medium, High"

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extract and return severity from the response
    return completion.choices[0].message.content

# Load your CSV file
file_path = "your_dataset.csv"  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Ensure the dataset contains a 'Deficiency Text' column
print(df.head())

# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

# Function to get embeddings
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=text, model=model)
    return response["data"][0]["embedding"]

# Generate embeddings for the 'Deficiency Text' column
df["embedding"] = df["Deficiency Text"].apply(lambda x: get_embedding(x))

# Convert embeddings to a NumPy array
embeddings = np.array(df["embedding"].tolist()).astype("float32")

# Create a FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
index.add(embeddings)  # Add embeddings to the index

print(f"Added {index.ntotal} embeddings to the index.")

def retrieve_relevant_chunks(query, k=3):
    query_embedding = np.array(get_embedding(query)).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    results = df.iloc[indices[0]]
    return results

def generate_response(query, retrieved_chunks):
    context = "\n".join(retrieved_chunks["Deficiency Text"].tolist())
    prompt = f"""
    Context:
    {context}

    Question: {query}
    Answer:
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()

# Example query
query = "How to address deficiency related to system downtime?"
relevant_chunks = retrieve_relevant_chunks(query)
response = generate_response(query, relevant_chunks)

print("Generated Response:", response)

def rag_pipeline(query, k=3):
    # Retrieve relevant chunks
    retrieved_chunks = retrieve_relevant_chunks(query, k)
    # Generate response
    response = generate_response(query, retrieved_chunks)
    return response

# Example usage
query = "Explain a deficiency related to employee training gaps."
result = rag_pipeline(query)
print("Generated Response:", result)

row_count = len(df)
for i in range(row_count):
    # Test the function with a new description
    new_description = df2.iloc[i]['def_text']
    severity = classify_severity(new_description)
    print(f"Predicted Severity: {severity}")
    if i % 100 == 0:
        print(i)
    sheet.insert_cols(3)
    sheet["C2"] = df2.iloc[i - 1]['PscInspectionId']
    sheet["C3"] = df2.iloc[i - 1]['deficiency_code']
    sheet["C4"] = severity

workbook.save(r"Datasets/submission_template.xlsx")

# Test the function with a new description
new_description = "Description of a new deficiency that involves critical issues."
severity = classify_severity(new_description)
print(f"Predicted Severity: {severity}")


st.sidebar.image(image="/Users/yixin/Downloads/Logo (1).png")

st.sidebar.markdown("<h2 style='text-align: left; color: violet; '>Instructions!</h2",unsafe_allow_html=True)

st.sidebar.markdown("<h5 style='text-align: left; color: black; '>Key in the deficiency description into the text box and the Chatbot will provide you with the severity of the deficiency.</h5",unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: violet;'>TOTIV Chatypie</h1>",unsafe_allow_html=True)

st.markdown("<h5 style='text-align: center; color: black;'>How can I help today? :)</h5>",unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input(label="", key="input", placeholder= "Type your message here...")

if user_input: 
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[
            {"role": "user", "content": "user_input"}
        ]
    )

    chatbot_response = completion.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": chatbot_response})

for message in st.session_state.messages:
    role = "User" if message["role"] == "user" else "Chatbot"
    st.markdown(f"**{role}:** {message['content']}")