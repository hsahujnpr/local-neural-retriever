import os
import uuid
import argparse
import numpy as np
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import Batch
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
import sys
#from pypdf import PdfReader

if len(sys.argv) < 3:
    print("Usage: qd_embed_macbook_files.py <collection_name> <file_paths>")
    exit()

def embed_and_populate():
    with open(args.file_paths, "r", encoding="utf-8") as f:
        file_list = [line.strip() for line in f if line.strip()]

    for filename in file_list:
        all_docs      = []
        all_metadata = []
        all_ids       = []

        supported_extensions = {".md", ".pdf", ".pptx", ".docx", ".xlsx"}
        try:
            content = ""
            if Path(filename).suffix.lower() in supported_extensions:
                print(f"Processing: {filename}")
                try:
                    # Use docling document converter
                    result = converter.convert(filename)
                    
                    #markdown_text = result.document.export_to_markdown()
                    # Print and store the extracted markdown text for debug
                    #with open(filename+".md", "w", encoding="utf-8") as f:
                        #f.write(markdown_text)
                    #print("Docling extracted the following from the file")
                    #print("===")
                    #print(markdown_text)
                    #print("===")

                except Exception as e:
                    print(f"Error reading PDF {filename}: {e}")
                    continue
            else:
                #Handle other file types (TBD)
                print("Unsupported file format")
                continue

            #Chunk the data for embedding
            chunks = list(chunker.chunk(result.document))
            #metadata = [chunk.meta.export_json_dict() for chunk in chunks]
       
            #Construct the embedding vectors and Points to 'upsert()'
            points=[]
            for i,chunk in enumerate(chunks):
                # Create a unique index 
                point_index = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}+{i}"))

                # Chunk the file content and contextualize it using Docling's Chunker 
                text_to_embed = chunker.contextualize(chunk)
                page_number   = chunk.meta.doc_items[0].prov[0].page_no if chunk.meta.doc_items else "NA"

                # Prefix the filename to the text to provide more context to the model. 
                # Strip the filename of any path and extension.
                filename_only = Path(filename).stem
                text_to_embed = f"{filename_only}: {text_to_embed}"

                if __debug__:
                    print("Chunk:"+str(i))
                    print(text_to_embed)
                    print("-----")

                #vector = list(model.embed([chunk.text]))[0]
                vector = list(model.embed([text_to_embed]))[0]

                #Contruct the point structure
                points.append(
                    PointStruct(
                        id = point_index,
                        vector = vector,
                        payload = {
                            "text":chunk.text,
                            "document_name":filename,
                            "Section": ">".join(chunk.meta.headings) if chunk.meta.headings else "root",
                            "Page#": page_number,
                            "chunk":i}
                    )
                )

            # Check if points is not empty before upserting
            if not points:
                print(f"No valid chunks found for {filename}, skipping upsert.")
                continue

            #Insert into the DB
            client.upsert(
                collection_name=collection_name,
                points=points
            )   
            print(f"Inserted {len(points)} points from {filename} into collection"+ collection_name)
            print("====")

        except FileNotFoundError:
            print(f"Skipping: {filename} (File not found)") 
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed sentences to local Qdrant")
    parser.add_argument("collection_name", help="Name of the collection")
    parser.add_argument("file_paths", help="Files to embed")
    args = parser.parse_args()

    #model_name = "BAAI/bge-small-en"
    #model_name = "BAAI/bge-large-en-v1.5"
    #model_name = "snowflake/snowflake-arctic-embed-m-long"
    #model_name  = "mixedbread-ai/mxbai-embed-large-v1"
    model_name = "jinaai/jina-embeddings-v3"
    max_length = 2048
    client     = QdrantClient(host="localhost", port=6333)
    model      = TextEmbedding(model_name = model_name, max_length=max_length)

    print("Using Model: " + model_name + "Model size:" + str(client.get_embedding_size(model_name)))

    #Instantiate docling objects
    converter = DocumentConverter()
    chunker   = HybridChunker()

    #collection_name="my_macbook_files"
    collection_name=args.collection_name
    if not client.collection_exists(collection_name):
        print("Collection does not exist - creating:"+collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=client.get_embedding_size(model_name), 
                distance=Distance.COSINE),
        )

    print("List of collections:")
    print(client.get_collections())

    embed_and_populate()
#done
