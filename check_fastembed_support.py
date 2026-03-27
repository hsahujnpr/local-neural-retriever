from fastembed import TextEmbedding, LateInteractionTextEmbedding

supported_models    = TextEmbedding.list_supported_models()
supported_rerankers = LateInteractionTextEmbedding.list_supported_models()

#Query for specific models
#found = False
print("Supported models:")
for model in supported_models:
    print(f"- {model['model']}")

print("Supported Re-rankers:")
for reranker in supported_rerankers:
    print(f"- {reranker['model']}")

#    if model['model'] == "BAAI/bge-m3":
#        found = True
# 
#if found:
#    print("\nBAAI/bge-m3 is supported.")
#else:
#    print("\nBAAI/bge-m3 is NOT found in the list.")
#
# Try to instantiate it just in case list is manual but it works dynamically
#    try:
#        model = TextEmbedding(model_name="BAAI/bge-m3")
#        print("Successfully instantiated BAAI/bge-m3")
#    except Exception as e:
#print(f"Failed to instantiate BAAI/bge-m3: {e}")

