import asyncio
import os
from llama_index.core import StorageContext, load_index_from_storage
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv(dotenv_path=".env.local")

# Explicitly set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


async def test_vector_db():
    # First, check if the vector index directory exists
    if not os.path.exists("./vector_index"):
        print("Error: Vector index not found. Please run vectorDbHandler.py first.")
        return

    try:
        # Load the existing vector index
        storage_context = StorageContext.from_defaults(persist_dir="./vector_index")
        index = load_index_from_storage(storage_context)

        # Create query engine
        query_engine = index.as_query_engine(similarity_top_k=3)

        # Test queries
        test_queries = [
            "Jorge Urioste",
            "Nicolita numberg Burger King",
            "Ahora soy nancy velazquez de la upsa"
        ]

        for query in test_queries:
            enhanced_query = f"Find information about {query}, if its a client it should match the enterprise name given with the client name given or could be a solution context that should match the description of a protocol"
            response = query_engine.query(enhanced_query)
            print(f"\nQuery: {query}")
            print(f"Response: {response}")
            print("-" * 50)

        # Print stats about the vector database
        doc_count = len(index.docstore.docs)
        print(f"\nVector index contains {doc_count} documents")
        print(f"Sample document IDs: {list(index.docstore.docs.keys())[:3]}")

    except Exception as e:
        print(f"Error testing vector database: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_vector_db())