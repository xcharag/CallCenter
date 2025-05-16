from livekit.agents import function_tool, RunContext
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=".env.local")

@function_tool
async def query_info(context: RunContext, query: str, top_k: int) -> str:
    """
    Search through documents using vector similarity to answer questions.

    Args:
        context: The runtime context
        query: The search query or question to find information about
        top_k: Number of top results to consider
    """
    client = OpenAI()

    enhanced_query = f"Find information about {query}, if its a client it should match the enterprise name given with the client name given or could be a solution context that should match the description of a protocol"
    # Handle default inside the function instead of in the signature
    if top_k is None or top_k <= 0:
        top_k = 3

    try:
        # Check if vector index exists
        if not os.path.exists("./vector_index"):
            return "Error: Vector index not found. Please run vectorDbHandler.py first to create the index."

        # Load the existing vector index from storage
        storage_context = StorageContext.from_defaults(
            persist_dir="./vector_index"
        )
        index = load_index_from_storage(storage_context)

        # Create a query engine
        query_engine = index.as_query_engine(
            similarity_top_k=(top_k,5),
        )

        # Execute the query
        response = query_engine.query(enhanced_query)

        return str(response)
    except FileNotFoundError:
        return "Error: Vector index files not found. Run vectorDbHandler.py to create the index."
    except Exception as e:
        return f"Error searching documents: {str(e)}"