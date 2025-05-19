# sementic_search
searching semantic cosine function approach. 

1. Define the Goal
    - Build an application that:
    - Accepts a CSV with text1 and text2 columns.
    - Calculates how semantically similar each text pair is.
    - Outputs a similarity score between 0 and 1.

2. Choose a Semantic Embedding Model
    - Use all-MiniLM-L6-v2 from SentenceTransformers:
    - Fast and efficient
    - Pre-trained on semantic textual similarity tasks
    - Ideal for short-to-medium-length sentences

3. Design the Streamlit Application
    - Accept CSV uploads via the interface.
    - Validate that the file contains the required columns.
    - Preload the model once for efficiency.
    - Encode both text columns into vector embeddings.
    - Compute cosine similarity for each row pair.
    - Display and allow download of the results.

4. Similarity Score Calculation
    - Convert both text1 and text2 into embeddings.
    - Use cosine similarity to quantify similarity.
    - Score:
        - 1.0: Highly similar
        - 0.0: Completely dissimilar
        - (Negative scores are rare but may indicate model noise or poorly structured input)

5. CSV Handling Logic
    - Process the entire CSV or a subset (e.g., first 50 rows) for efficiency.
    - Store the results by adding a new column: similarity score.

6. API Readiness (Optional)
    - Store results in a structured format (CSV, JSON, or database).
    - Use a FastAPI backend (optional) to:
    - Load precomputed results
    - Accept input text1, text2 via API
    - Return the similarity score if match found (or compute on-the-fly)