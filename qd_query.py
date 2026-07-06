import os
import time
import html
from urllib.parse import quote
from fastembed import TextEmbedding, LateInteractionTextEmbedding
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import Batch
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
import sys

# Usage:
# python qd_query.py <collection-name> <query-text> [norerank] [top_n]
# sys.argv[0] = qd_query.py
# sys.argv[1] = <collection-name>
# sys.argv[2] = <query-text>
# optional args (any order): "norerank", top_n (integer)

if len(sys.argv) < 3:
    print("Usage: qd_query.py <collection-name> <query-text> [norerank] [top_n]")
    exit()

collection_name = sys.argv[1]
query_text = sys.argv[2]
norerank   = False
top_n      = 10
for arg in sys.argv[3:]:
    if arg.lower() == "norerank":
        norerank = True
    elif arg.isdigit():
        top_n = int(arg)
    else:
        print("Usage: qd_query.py <collection-name> <query-text> [norerank] [top_n]")
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


def query_text_to_filename(query):
    """Convert first three words of query text to a safe filename base."""
    words = query.strip().split()
    base = " ".join(words[:3]) if words else "query"
    return base.replace("/", "-").replace("\\", "-")


def unique_output_path(base_name, output_dir, ext=".html"):
    """Return a unique file path, appending [1], [2], ... if the file already exists."""
    candidate = os.path.join(output_dir, f"{base_name}{ext}")
    if not os.path.exists(candidate):
        return candidate
    counter = 1
    while True:
        candidate = os.path.join(output_dir, f"{base_name}[{counter}]{ext}")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def generate_html(data):
    """Generate HTML from parsed data"""
    query = html.escape(data['query'])
    collection = html.escape(data['collection'])
    embed_model = html.escape(data['embed_model'])
    embed_size = html.escape(data['embed_size'])
    reranker_model = html.escape(data['reranker_model']) if data['reranker_model'] else None
    reranker_size = html.escape(data['reranker_size'])
    top_n = html.escape(data['top_n'])
    rerank_label = "after re-ranking" if data['reranked'] else "without re-ranking"

    # Build result cards
    result_cards = []
    for i, r in enumerate(data['results'], 1):
        doc = html.escape(r['document'])
        doc_url = 'file://' + quote(r['document'])
        chunk = html.escape(r['chunk'])
        section = html.escape(r['section'])
        page = html.escape(r['page'])
        text = html.escape(r['text'])

        score_html = ""
        if r['score']:
            score_html = f'<span class="tag"><strong>Score:</strong> {html.escape(r["score"])}</span>'

        card = f"""  <!-- Result {i} -->
  <div class="result-card">
    <div class="result-header">
      <div class="rank-badge">{i}</div>
      <div class="doc-title"><a href="{doc_url}" target="_blank">{doc}</a> <span class="chunk">(Chunk {chunk})</span></div>
    </div>
    <div class="tags">
      <span class="tag"><strong>Section:</strong> {section}</span>
      <span class="tag"><strong>Page:</strong> {page}</span>
      {score_html}
    </div>
    <div class="result-text">{text}</div>
    <hr class="separator">
  </div>
"""
        result_cards.append(card)

    reranker_meta = ""
    if reranker_model:
        reranker_meta = f'<span><span class="icon">🔀</span> Reranker: <strong>{reranker_model}</strong> ({reranker_size})</span>'

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Search Results — {query}</title>
  <style>
    :root {{
      --bg: #f5f7fa;
      --card-bg: #ffffff;
      --accent: #2563eb;
      --accent-light: #dbeafe;
      --text: #1e293b;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --rank-bg: #2563eb;
      --rank-text: #ffffff;
      --tag-bg: #f1f5f9;
      --tag-text: #475569;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding: 2rem 1rem;
    }}

    .container {{
      max-width: 960px;
      margin: 0 auto;
    }}

    header {{
      margin-bottom: 2rem;
    }}

    header h1 {{
      font-size: 1.6rem;
      font-weight: 700;
      color: var(--text);
      margin-bottom: 0.5rem;
    }}

    .query-info {{
      background: var(--accent-light);
      border-left: 4px solid var(--accent);
      padding: 0.75rem 1rem;
      border-radius: 0 8px 8px 0;
      margin-bottom: 1rem;
    }}

    .query-info .label {{ font-weight: 600; color: var(--accent); }}
    .query-info .value {{ color: var(--text); }}

    .meta-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 1.5rem;
      font-size: 0.85rem;
      color: var(--text-muted);
    }}

    .meta-bar span {{ display: flex; align-items: center; gap: 0.3rem; }}
    .meta-bar .icon {{ font-size: 1rem; }}

    .result-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 1rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
      transition: box-shadow 0.2s;
    }}

    .result-card:hover {{
      box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }}

    .result-header {{
      display: flex;
      align-items: flex-start;
      gap: 0.75rem;
      margin-bottom: 0.75rem;
    }}

    .rank-badge {{
      flex-shrink: 0;
      background: var(--rank-bg);
      color: var(--rank-text);
      font-weight: 700;
      font-size: 0.8rem;
      width: 2rem;
      height: 2rem;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    .doc-title {{
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--text);
      word-break: break-all;
    }}

    .doc-title a {{
      color: var(--accent);
      text-decoration: none;
    }}

    .doc-title a:hover {{
      text-decoration: underline;
    }}

    .doc-title .chunk {{
      font-weight: 400;
      color: var(--text-muted);
      font-size: 0.85rem;
    }}

    .tags {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-bottom: 0.75rem;
    }}

    .tag {{
      background: var(--tag-bg);
      color: var(--tag-text);
      font-size: 0.78rem;
      padding: 0.2rem 0.6rem;
      border-radius: 4px;
    }}

    .tag strong {{ font-weight: 600; }}

    .result-text {{
      font-size: 0.9rem;
      color: var(--text);
      line-height: 1.7;
      white-space: pre-wrap;
    }}

    .separator {{
      border: none;
      border-top: 1px dashed var(--border);
      margin: 0.75rem 0 0;
    }}

    footer {{
      text-align: center;
      margin-top: 2rem;
      font-size: 0.8rem;
      color: var(--text-muted);
    }}
  </style>
</head>
<body>

<div class="container">
  <header>
    <h1>Semantic Search Results</h1>
    <div class="query-info">
      <span class="label">Query:</span>
      <span class="value">&ldquo;{query}&rdquo;</span>
    </div>
    <div class="meta-bar">
      <span><span class="icon">📂</span> Collection: <strong>{collection}</strong></span>
      <span><span class="icon">🧠</span> Embedding: <strong>{embed_model}</strong> ({embed_size})</span>
      {reranker_meta}
      <span><span class="icon">📊</span> Top <strong>{top_n}</strong> results ({rerank_label})</span>
    </div>
  </header>

{"".join(result_cards)}
  <footer>
    Semantic search powered by Qdrant &middot; Embeddings: {embed_model} &middot; {"Reranker: " + reranker_model if reranker_model else "No reranking"}
  </footer>
</div>

</body>
</html>"""
    return html_content


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
        print(f"\nResults of query without re-ranking (Top {top_n}): ")
        print(f"="*45)

    for i, point in enumerate(results.points):
        if norerank:
            #Print first top_n results with score and document name
            if i < top_n:
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

    # Get embed and reranker model sizes
    embed_size = str(client.get_embedding_size(model_name))
    reranker_size = str(client.get_embedding_size(reranker_name)) if not norerank else "?"

    if norerank == False:
        print(f"\nResults of query after re-ranking (Top {top_n}):")
        print(f"=" * 43)
        reranked_top_hits, reranked_doc_names, reranked_sections, reranked_page_numbers, reranked_chunk_numbers = \
            rerank_hits(query_text, top_hits, doc_names, sections, page_numbers, chunk_numbers)
        results_list = []
        # Print first 10 reranked results with document name
        for i, (hit, doc) in enumerate(zip(reranked_top_hits, reranked_doc_names)):
            if i >= top_n:
                break
            print(f"\nResult {i+1}:")
            print(f"#Document: {doc} (Chunk: {reranked_chunk_numbers[i]})")
            print(f"##Section: {reranked_sections[i]}", f"##Page: {reranked_page_numbers[i]}\n")
            print(f"Text: {hit}")
            print("---------------\n")
            results_list.append({
                "document": doc,
                "chunk": str(reranked_chunk_numbers[i]),
                "section": reranked_sections[i],
                "page": str(reranked_page_numbers[i]),
                "score": None,
                "text": hit,
            })
        print(f"="*40)

        # Generate HTML 
        data = {
            'query': query_text,
            'collection': collection_name,
            'embed_model': model_name,
            'embed_size': embed_size,
            'reranker_model': reranker_name,
            'reranker_size': reranker_size,
            'reranked': True,
            'top_n': str(top_n),
            'results': results_list,
        }
        html_content = generate_html(data)
        output_dir = os.path.join(os.path.dirname(__file__), "testset")
        os.makedirs(output_dir, exist_ok=True)
        output_file = unique_output_path(query_text_to_filename(query_text), output_dir)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML output written to: {output_file}")
    else:
        results_list = []
        for i, point in enumerate(results.points):
            if i >= top_n:
                break
            results_list.append({
                "document": point.payload['document_name'],
                "chunk": str(point.payload['chunk']),
                "section": point.payload['Section'],
                "page": str(point.payload['Page#']),
                "score": str(point.score),
                "text": point.payload['text'],
            })

        # Generate HTML 
        data = {
            'query': query_text,
            'collection': collection_name,
            'embed_model': model_name,
            'embed_size': embed_size,
            'reranker_model': None,
            'reranker_size': "?",
            'reranked': False,
            'top_n': str(top_n),
            'results': results_list,
        }
        html_content = generate_html(data)
        output_dir = os.path.join(os.path.dirname(__file__), "testset")
        os.makedirs(output_dir, exist_ok=True)
        output_file = unique_output_path(query_text_to_filename(query_text), output_dir)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML output written to: {output_file}")
#done __main__

