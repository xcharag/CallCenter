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
    Search through documents using vector similarity to find clients, companies, or protocols.

    Args:
        context: The runtime context
        query: The search query (client name and/or company name)
        top_k: Number of top results to consider
    """
    client = OpenAI()

    if top_k is None or top_k <= 0:
        top_k = 3

    try:
        if not os.path.exists("./vector_index"):
            return "ERROR: Vector index not found."

        # Make sure we're using the same embedding model that created the index
        from llama_index.embeddings.openai import OpenAIEmbedding
        embed_model = OpenAIEmbedding(model_name="text-embedding-ada-002")

        storage_context = StorageContext.from_defaults(persist_dir="./vector_index")
        index = load_index_from_storage(storage_context, embed_model=embed_model)

        # Split name and company for better search
        enhanced_query = f"Find information about {query}. Match client name with company name if both are provided."

        query_engine = index.as_query_engine(similarity_top_k=top_k)
        response = query_engine.query(enhanced_query)

        return f"SEARCH RESULTS:\n{str(response)}"
    except Exception as e:
        return f"ERROR: {str(e)}"