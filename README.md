# local-neural-retriever: Semantic Search with Qdrant & FastEmbed

A semantic search system built on **Qdrant** vector database and **FastEmbed** embedding models. Supports ingesting multi-format documents (PDF, DOCX, PPTX, XLSX, Markdown), chunking them intelligently, generating embeddings, and performing semantic search with optional **ColBERT v2 reranking**.

It is a self-contained, lightweight system primarily intended to be used for easily searching on local laptop/desktop content, where privacy matters. It inherits the low-footprint, Rust-memory-safe attributes of Qdrant vector DB which can be run local machin. Client scripts connect on HTTP interface on TCP/6333 for ingsting content and querying for search text.

It can also be used by running Qdrant instance remotely, where larger storage is needed for ingesting larger amount of data.

---

## Architecture

```
DATA INGESTION                          SEMANTIC SEARCH

Source Files                            Natural Language Query
(PDF, DOCX, PPTX, XLSX, MD)                    │
        │                               TextEmbedding (Jina v3)
Docling DocumentConverter                       │
        │                               Qdrant Cosine Similarity
HybridChunker                            (Top-25 candidates)
        │                                       │
Contextualization +                     ColBERT v2 Reranking
Metadata Extraction                      (MaxSim scoring)
        │                                       │
TextEmbedding (Jina v3)                 Ranked Results (Top-10)
        │                               Score │ Doc │ Section │
Qdrant Upsert (batch)                   Page │ Text
```

## Scripts

| Script | Description |
|--------|-------------|
| `qd_embed_macbook_files.py` | Ingests documents (PDF, DOCX, PPTX, XLSX, MD) using Docling for parsing and HybridChunker for chunking. Extracts section hierarchy, page numbers, and metadata. |
| `qd_query.py` | Semantic search with optional ColBERT v2 reranking. Displays results with scores, document names, sections, and page numbers. |
| `qd_list_collections.py` | Lists all collections with point counts, vector dimensions, and distance metrics. |
| `qd_delete_collection.py` | Deletes a specified collection. |
| `check_fastembed_support.py` | Lists all embedding models and rerankers supported by FastEmbed. |

## Models

| Role | Model | Details |
|------|-------|---------|
| **Embedding** | `jinaai/jina-embeddings-v3` | Primary model, max 2048 tokens |
| **Embedding** | `BAAI/bge-large-en-v1.5`    | Alternative model |
| **Reranking** | `jinaai/jina-colbert-v2`    | Late interaction ColBERT model, MaxSim scoring |

## Usage

### Prerequisites

- Python 3.10+
- Qdrant running locally on `localhost:6333`
  (Can be downloaded from https://qdrant.tech/documentation/operations/installation/)

### Install Dependencies

```bash
pip install qdrant-client fastembed docling numpy
```

### Embed Documents

```bash
# Embed files listed in a text file (one path per line)
python qd_embed_macbook_files.py <collection-name> <file-list.txt> 

### Query

```bash
# Semantic search with ColBERT v2 reranking (default)
python qd_query.py <collection-name> "your search query"

# Semantic search (without reranking)
python qd_query.py <collection-name> "your search query" norerank
```

### Manage Collections

```bash
# List all collections
python qd_list_collections.py

# Delete a collection
python qd_delete_collection.py <collection-name>
```

## Configuration

| Parameter | Default |
|-----------|---------|
| Distance metric | Cosine |
| Search candidates | 25 |
| Displayed results | 10 |
| Qdrant host | `localhost:6333` |
| Point IDs | UUID v5 (deterministic, based on content) |

## Samples

- **`samples/`** — Sample queries output

## Key Features

- **Multi-format ingestion** — PDF, DOCX, PPTX, XLSX, and Markdown via Docling
- **Intelligent chunking** — HybridChunker preserves document structure and section hierarchy
- **Rich metadata** — Document name, section headings, page numbers, and chunk indices stored as payloads
- **Two-stage retrieval** — Dense vector search followed by optional ColBERT v2 reranking for higher precision
- **Local & Cloud** — Works with local Qdrant instances; can be extended to use Qdrant Cloud
