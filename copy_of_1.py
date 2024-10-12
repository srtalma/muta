import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from ibm_watsonx_ai.foundation_models import Model
import os
import getpass

# Streamlit title
st.title('Arabic Poetry Generator')

# Load dataset from uploaded file
uploaded_file = st.file_uploader("Upload your poetry dataset (CSV or JSON)")
if uploaded_file:
    df = pd.read_csv(uploaded_file)  # Assuming it's a CSV file
    st.write("Data preview:")
    st.write(df.head())
else:
    st.warning("Please upload a poetry dataset to proceed.")
    st.stop()

# Initialize FAISS and BM25 models
st.text("Initializing FAISS and BM25 models...")

# Initialize Sentence-BERT model for FAISS
model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
poem_texts = [poem for poem in df['البيت']]
poem_embeddings = model.encode(poem_texts, convert_to_tensor=False)
poem_embeddings = np.array(poem_embeddings)

# Initialize FAISS index for fast retrieval
dimension = poem_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(poem_embeddings)

# BM25 Initialization
df['combined_text'] = df[['الشاعر', 'البحر']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
tokenized_corpus = [doc.split(" ") for doc in df['combined_text'].values]
bm25 = BM25Okapi(tokenized_corpus)

# IBM Watson Model Credentials
st.text("Initializing IBM Watson model...")
def get_credentials():
    return {
        "url": "https://eu-de.ml.cloud.ibm.com",
        "apikey": getpass.getpass("Enter your API key: ")
    }

credentials = get_credentials()

model_id = "sdaia/allam-1-13b-instruct"
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 900,
    "repetition_penalty": 1.0
}
project_id = "65870abf-b0eb-4dce-9b63-eeed50e3a3d0"
space_id = os.getenv("SPACE_ID")

try:
    watson_model = Model(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id,
        space_id=space_id
    )
    st.success("Watson model initialized successfully!")
except Exception as e:
    st.error(f"Error initializing the model: {e}")
    st.stop()

# Collect user inputs for poet, meter, and emotion
poet_name = st.selectbox("Select the Poet", df['الشاعر'].unique())
meter = st.selectbox("Select the Meter", df['البحر'].unique())
emotion = st.selectbox("Select the Emotion", ["Pride", "Love", "Sadness", "Satire", "Praise"])
context = st.text_input("Enter Context", "Example: Afkhar bi akhi (I am proud of my brother)")

# Trigger poem generation when the user clicks the button
if st.button("Generate Poem"):

    # Step 1: Retrieve Metaphor and Context using Watson model
    st.text("Retrieving metaphors and context...")
    
    def teacher_provide_metaphor_and_emotion_context(emotion, user_context):
        prompt = f"Provide metaphors related to the feeling of {emotion}. {user_context}"
        response = watson_model.generate_text(prompt=prompt, guardrails=False)
        return response.strip()

    emotion_context = teacher_provide_metaphor_and_emotion_context(emotion, context)

    # Step 2: Retrieve poems using BM25 for poet and meter
    st.text("Retrieving poems using BM25...")
    
    def retrieve_poems_bm25(poet, meter, num_results=3):
        query = f"{poet} {meter}"
        tokenized_query = query.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_results = df.loc[bm25_scores.argsort()[::-1]].head(num_results)
        return top_bm25_results['البيت'].tolist()

    bm25_poems = retrieve_poems_bm25(poet_name, meter)

    # Step 3: Retrieve poems using FAISS based on emotion
    st.text("Retrieving poems using FAISS...")

    def retrieve_poems_faiss(query, num_results=3):
        query_embedding = model.encode([query], convert_to_tensor=False)
        distances, indices = index.search(np.array(query_embedding), num_results)
        return [poem_texts[i] for i in indices[0]]

    faiss_poems = retrieve_poems_faiss(emotion)

    # Step 4: Generate the first draft of the poem using Watson model
    st.text("Generating the poem...")
    
    def student_generate_first_poem(emotion_context, retrieved_poems):
        prompt = f"Based on the following context: {emotion_context}, and examples: {retrieved_poems}, generate a poem."
        response = watson_model.generate_text(prompt=prompt, guardrails=False)
        return response.strip()

    generated_poem = student_generate_first_poem(emotion_context, bm25_poems + faiss_poems)
    st.text("Generated Poem:")
    st.write(generated_poem)

    # Step 5: Meter correction using Watson model
    st.text("Correcting meter and weight...")
    
    def teacher_fix_meter_and_weight(student_poem, meter):
        prompt = f"Fix the meter and weight of this poem: {student_poem} to match the {meter} meter."
        response = watson_model.generate_text(prompt=prompt, guardrails=False)
        return response.strip()

    corrected_poem = teacher_fix_meter_and_weight(generated_poem, meter)
    st.text("Meter Corrected Poem:")
    st.write(corrected_poem)

    # Step 6: Final rhyme adjustment
    st.text("Finalizing rhyme...")
    
    def teacher_fix_rhyme_and_finalize(revised_poem, rhyme="د"):
        prompt = f"Adjust the rhyme of this poem: {revised_poem} to match the {rhyme} rhyme."
        response = watson_model.generate_text(prompt=prompt, guardrails=False)
        return response.strip()

    final_poem = teacher_fix_rhyme_and_finalize(corrected_poem)
    st.text("Final Poem:")
    st.write(final_poem)
