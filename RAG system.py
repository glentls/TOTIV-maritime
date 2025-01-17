pip install openai faiss-cpu pandas

import pandas as pd

# Load your CSV file
file_path = "your_dataset.csv"  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Ensure the dataset contains a 'Deficiency Text' column
print(df.head())

import openai

# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

# Function to get embeddings
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=text, model=model)
    return response["data"][0]["embedding"]

# Generate embeddings for the 'Deficiency Text' column
df["embedding"] = df["Deficiency Text"].apply(lambda x: get_embedding(x))


import numpy as np
import faiss

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

