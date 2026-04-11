import os
import time
from fastembed import TextEmbedding, LateInteractionTextEmbedding
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import Batch
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
import sys

# Usage:
# python qd_client_query.py <collection-name> <query-text> <'norerank'>
# sys.argv[0] = qd_client_query.py
# sys.argv[1] = <collection-name>
# sys.argv[2] = <query-text>
# sys.argv[3] = <"norerank">

if len(sys.argv) < 3:
    print("Usage: qd_query.py <collection-name> <query-text> <'norerank'>")
    exit()

collection_name = sys.argv[1]
query_text = sys.argv[2]
norerank   = False
if len(sys.argv) > 3:
    if sys.argv[3].lower() == "norerank":
        norerank = True
    else:
        print("Usage: qd_query.py <collection-name> <query-text> <'norerank'>")
        exit()


def rerank_hits(query_text, top_hits, doc_names, sections, page_numbers, chunk_numbers):
    # Re-rank top_hits using gina colbert v2
    reranker = LateInteractionTextEmbedding(reranker_name)
    hits_embedding = list(reranker.embed(top_hits))
    query_embedding = list(reranker.embed([query_text]))
    
    # Debug: Print embedding shapes
    if __debug__:
        print(f"\nDebug - Query embedding shape: {query_embedding[0].shape}")
        print(f"Debug - First hit embedding shape: {hits_embedding[0].shape}")

    #Calculate maxsim scores for each hit and print them
    start_time = time.time()
    reranked_scores = [calculate_maxsim(query_embedding, [hit_emb]) for hit_emb in hits_embedding]
    elapsed = time.time() - start_time
    if __debug__:
        print(f"Time for reranking {elapsed:.4f}s (numpy/CPU)")
        for i, score in enumerate(reranked_scores):
            print(f"\nMaxSim score for result {i+1}: {score}")

    # Sort top_hits and doc_names together by reranked_scores
    sorted_results = sorted(zip(reranked_scores, top_hits, doc_names, sections, page_numbers, chunk_numbers), reverse=True)
    top_hits = [x for _, x, _, _, _, _ in sorted_results]
    doc_names = [x for _, _, x, _, _, _ in sorted_results]
    sections = [x for _, _, _, x, _, _ in sorted_results]
    page_numbers = [x for _, _, _, _, x, _ in sorted_results]
    chunk_numbers = [x for _, _, _, _, _, x in sorted_results]

    return top_hits, doc_names, sections, page_numbers, chunk_numbers
#done rerank_hits

def calculate_maxsim(query_emb, doc_emb):
    # Calculate the maxsim score between query and document embeddings
    # query_emb and doc_emb are lists of multi-vector embeddings
    maxsim_score = 0.0
    for q_emb in query_emb:  # Iterate over each query embedding
        for d_emb in doc_emb:  # Iterate over each document embedding
            score = 0.0
            for q_vec in q_emb:  # Each query token vector
                max_sim = -float('inf')
                for d_vec in d_emb:  # Each document token vector
                    sim = np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec))
                    max_sim = max(max_sim, sim)
                score += max_sim
            maxsim_score = max(maxsim_score, score)
    return maxsim_score
#done calculate_maxsim


if __name__ == "__main__":
    client = QdrantClient(host="localhost", port=6333)
    #model_name = "BAAI/bge-small-en"
    #model_name = "snowflake/snowflake-arctic-embed-m-long"
    #model_name = "BAAI/bge-large-en-v1.5"
    #model_name = "mixedbread-ai/mxbai-embed-large-v1"
    model_name  = "jinaai/jina-embeddings-v3"
    max_length = 2048
    reranker_name = "jinaai/jina-colbert-v2"

    print(f"\nQuerying for: '{query_text}' in Collection: {collection_name}")
    print(f"Using embedding model: {model_name} (Model size: {client.get_embedding_size(model_name)})")
    if not norerank:
        print(f"Using reranker model: {reranker_name} (Model size: {client.get_embedding_size(reranker_name)})")

    if not client.collection_exists(collection_name):
        print("Collection does not exist")
        exit()

    #client.set_model(model_name)
    embedder = TextEmbedding(model_name, max_length=max_length)
    query_vector = list(embedder.embed([query_text]))[0].tolist()

    # Perform a similarity search
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=25
    )

    top_hits = []
    doc_names = []
    sections = []
    page_numbers = []
    chunk_numbers = []
    if norerank:
        print(f"\nResults of query without re-ranking (Top 10): ")
        print(f"="*45)

    for i, point in enumerate(results.points):
        if norerank:
            #Print first 10 results with score and document name
            if i <= 9:
                print(f"\nResult {i+1}:")
                print(f"Score: {point.score}")
                print(f"#Document: {point.payload['document_name']} (Chunk: {point.payload['chunk']})")
                print(f"##Section: {point.payload['Section']}", f"##Page: {point.payload['Page#']}\n")
                print(f"Text: {point.payload['text']}")
                print("---------------\n")
        top_hits.append(point.payload['text'])
        doc_names.append(point.payload['document_name'])
        sections.append(point.payload['Section'])
        page_numbers.append(point.payload['Page#'])
        chunk_numbers.append(point.payload['chunk'])
        
    if norerank:
       print(f"="*40)

    if norerank == False:
        print(f"\nResults of query after re-ranking (Top 10):")
        print(f"=" * 43)
        reranked_top_hits, reranked_doc_names, reranked_sections, reranked_page_numbers, reranked_chunk_numbers = \
            rerank_hits(query_text, top_hits, doc_names, sections, page_numbers, chunk_numbers)
        # Print first 10 reranked results with document name
        for i, (hit, doc) in enumerate(zip(reranked_top_hits, reranked_doc_names)):
            if i >= 10:
                break
            print(f"\nResult {i+1}:")
            print(f"#Document: {doc} (Chunk: {reranked_chunk_numbers[i]})")
            print(f"##Section: {reranked_sections[i]}", f"##Page: {reranked_page_numbers[i]}\n")
            print(f"Text: {hit}")
            print("---------------\n")
        print(f"="*40)
#done __main__

