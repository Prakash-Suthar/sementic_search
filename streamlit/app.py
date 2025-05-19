import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import io
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set title
st.title(" Semantic Textual Similarity (STS) App")

# Load model once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# File upload
uploaded_file = st.file_uploader("Upload CSV file with `text1` and `text2` columns", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Validate column names
        if "text1" not in df.columns or "text2" not in df.columns:
            st.error("CSV must contain 'text1' and 'text2' columns.")
        else:
            st.success("CSV loaded successfully!")

            # Limit to first 50 rows
            df = df[["text1", "text2"]].head(50)

            # Show sample
            st.subheader("Preview of First 50 Rows")
            st.dataframe(df)

            # Compute embeddings and similarity
            st.subheader(" Computing Semantic Similarity for First 50 Rows...")
            with st.spinner("Processing..."):
                embeddings1 = model.encode(df["text1"].tolist(), convert_to_tensor=True)
                embeddings2 = model.encode(df["text2"].tolist(), convert_to_tensor=True)

                similarity_scores = cosine_similarity(embeddings1.cpu(), embeddings2.cpu()).diagonal()
                df["similarity score"] = similarity_scores.round(3)

            st.subheader("Results (First 50 Rows with Similarity Score)")
            st.dataframe(df)

            # Download results
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button("📥 Download Results as CSV", csv_buffer.getvalue(), "similarity_results.csv", "text/csv")

    except Exception as e:
        st.error(f"Error reading file: {e}")

# Footer
st.markdown("---")
st.caption("Developed with using Sentence Transformers and Streamlit")
