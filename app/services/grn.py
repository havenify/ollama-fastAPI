from pymongo import MongoClient
from app.utils.similarity import cosine_similarity

MONGO_URI = "mongodb+srv://erp_staging:eEGi8pRF3dUDpIOO@cleanzy.fvlqgga.mongodb.net/erp?retryWrites=true&w=majority"
DB_NAME = "erp"
COLLECTION_NAME = "grns"

def fetch_and_rank_grns(query_embedding, top_k=3):
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    grns = list(collection.find({ "embedding": { "$exists": True, "$ne": [] }}))
    ranked = []
    for grn in grns:
        score = cosine_similarity(query_embedding, grn["embedding"])
        summary = grn.get("summary", "Unknown GRN")
        ranked.append({ "grn": grn, "score": score, "summary": summary })
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]

def build_prompt(top_docs, user_question):
    context = "\n".join([f"{i+1}. {doc['summary']}" for i, doc in enumerate(top_docs)])
    return f"You are a helpful assistant that answers based on GRN data.\n\nRelevant GRNs:\n{context}\n\nQuestion: {user_question}\nAnswer:"