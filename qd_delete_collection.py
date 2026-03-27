import os
from fastembed import TextEmbedding
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import Batch
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
import sys

# sys.argv[0] = qd_client_blogs.txt
# sys.argv[1] = <collecttion-name>>

if len(sys.argv) < 2:
    print("Usage: qd_client_delete_collection.py <collection-name>")
    exit()

collection_name = sys.argv[1]

client = QdrantClient(host="localhost", port=6333)
#model_name = "BAAI/bge-small-en"
model_name = "BAAI/bge-large-en-v1.5"
print("Model size:" + str(client.get_embedding_size(model_name)))

if not client.collection_exists(collection_name):
    print("Collection does not exist")
    exit()

client.delete_collection(
    collection_name=collection_name
)

print("List of remaining collections:")
print(client.get_collections())

