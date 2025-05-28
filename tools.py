from livekit.agents import function_tool, RunContext
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
import os
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine
import pandas as pd

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
        top_k = 5

    try:
        if not os.path.exists("./vector_index"):
            return "ERROR: Vector index not found."

        # Make sure we're using the same embedding model that created the index
        from llama_index.embeddings.openai import OpenAIEmbedding
        embed_model = OpenAIEmbedding(model_name="text-embedding-ada-002")

        storage_context = StorageContext.from_defaults(persist_dir="./vector_index")
        index = load_index_from_storage(storage_context, embed_model=embed_model)

        # Split name and company for better search
        enhanced_query = f"""Find information about {query}. There are this posibilities, match with the client name, the enterprise name, 
                            the service description or name if you find a match with the client name the enterprise name gave should match the one associated with the client provided must have."""

        query_engine = index.as_query_engine(similarity_top_k=top_k)
        response = query_engine.query(enhanced_query)

        return f"SEARCH RESULTS:\n{str(response)}"
    except Exception as e:
        return f"ERROR: {str(e)}"

def save_transcript_database(filepath: str, room_name: str):
    engine = create_database_connection()

    if engine is None:
        print("Failed to connect to the database. Transcript not saved.")
        return

    try:
        with engine.connect() as connection:
            from sqlalchemy import text
            query = text(""" INSERT INTO Calls (CodigoLlamada, HoraInicio, HoraFin, Grabacion) 
            VALUES (:room_name, NOW(), NOW(), :filepath)
            """)
            connection.execute(query, {
                "room_name": room_name,
                "filepath": filepath,
            })
            connection.commit()
        print(f"Transcript saved to database.")
    except Exception as e:
        print(f"Error saving transcript to database: {e}")



def create_database_connection(
        host=None, port=None, user=None, password=None, db_name=None, test_connection=True
):
    """
    Create and return a SQLAlchemy database engine connection.

    Args:
        host: Database host (defaults to DB_HOST env variable)
        port: Database port (defaults to DB_PORT env variable)
        user: Database user (defaults to DB_USER env variable)
        password: Database password (defaults to DB_PASSWORD env variable)
        db_name: Database name (defaults to DB_NAME env variable)
        test_connection: Whether to test the connection

    Returns:
        SQLAlchemy engine if successful, None if connection fails
    """
    print("Connecting to database...")
    host = host or os.environ.get("MYSQL_HOST")
    port = port or os.environ.get("MYSQL_PORT")
    user = user or os.environ.get("MYSQL_USER")
    password = password or os.environ.get("MYSQL_PASSWORD")
    db_name = db_name or os.environ.get("MYSQL_DATABASE")

    connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"
    engine = create_engine(connection_string)

    if test_connection:
        try:
            tables = pd.read_sql("SHOW TABLES", engine)
            print("Database connection successful!")
            return engine
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return None

    return engine