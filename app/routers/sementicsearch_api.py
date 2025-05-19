from fastapi import APIRouter, HTTPException
from app.models.sementicsearchModel import SementicModel
from sentence_transformers import SentenceTransformer   
from app.utils.computesimilarity import SimilarityCache
router = APIRouter()

# Load and precompute on startup
similarity_cache = SimilarityCache(r"E:\codified\dataneutron\dataNeutron_sementic\DataNeuron_Text_Similarity.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

@router.post("/score_checker")
def generatingAnswer(data: SementicModel):
    try:
        input_text1 = data.text1
        input_text2 = data.text2
        # model_card = get_model()
        similar = {}
        print("heyyy -- - -")
        similarity_score = similarity_cache.get_similarity(input_text1, input_text2)
        similar["similarity_score"] = similarity_score
        print("corrected score --->",similarity_score)
        
        return similar

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
