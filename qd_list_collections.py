import collections
from logging import info
import os
from fastembed import TextEmbedding
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import Batch
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
import sys

client = QdrantClient(host="localhost", port=6333)
#model_name = "BAAI/bge-small-en"
#model_name = "BAAI/bge-large-en-v1.5"
#print("Model size:" + str(client.get_embedding_size(model_name)))

print("List of collections:")
print("=" * 20)
response = client.get_collections()
for collection in response.collections:
    name = collection.name
    collection_info = client.get_collection(name)

    vectors_config = collection_info.config.params.vectors
    # Check if it's a single vector config or multiple
    if hasattr(vectors_config, 'size'):
        size = vectors_config.size
        distance = vectors_config.distance
    else:
        # If using multiple named vectors (like Jina + Sparse)
        size = "Multiple"
        distance = "Multiple"

    print(f"Collection: {name}")
    print(f"  - Vectors Count: {collection_info.points_count}")
    print(f"  - Vector Size:   {size}")
    print(f"  - Distance:      {distance}")
    print("-" * 30)

#print(client.get_collections())

