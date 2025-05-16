import json
import os
import pandas as pd
from sqlalchemy import create_engine
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.local")

def export_db_to_files(db_connection, export_dir="vectordb/knowledge_base"):
    """Export database tables to files for vector storage"""

    os.makedirs(export_dir, exist_ok=True)
    file_paths = []

    # Export enterprises
    try:
        enterprises_df = pd.read_sql("SELECT * FROM Companies", db_connection)

        for i, enterprise in enterprises_df.iterrows():
            enterprise_doc = {
                "enterprise_id": str(enterprise["Codigo"]),
                "name": enterprise["Nombre"],
                "type": "Company",
                "phoneNumber": enterprise.get("Telefono", ""),
                "email": enterprise.get("Email", ""),
                "city": enterprise.get("Ciudad", ""),
                "address": enterprise.get("Direccion", "")
            }

            file_path = f"{export_dir}/enterprise_{enterprise['Id']}.json"
            with open(file_path, "w") as f:
                json.dump(enterprise_doc, f, indent=2)
            file_paths.append(file_path)
    except Exception as e:
        print(f"Error exporting enterprises: {e}")

    # Export clients
    try:
        clients_df = pd.read_sql("""
SELECT 
    c.*,
    GROUP_CONCAT(e.Nombre SEPARATOR ', ') as enterprise_names,
    GROUP_CONCAT(e.Id SEPARATOR ', ') as enterprise_ids
FROM 
    Clients c
    JOIN ClientCompanies cc ON c.Id = cc.ClienteId
    JOIN Companies e ON e.Id = cc.EmpresaId
GROUP BY 
    c.CodigoCliente, c.Id, c.NombreCompleto
                                 """, db_connection)
        print(f"Exported {len(clients_df)} clients")

        # Optional: If too many clients, group by enterprise
        if len(clients_df) > 1000:
            # Group clients by enterprise
            for enterprise_id, group in clients_df.groupby("enterprise_id"):
                clients_list = []
                for _, client in group.iterrows():
                    clients_list.append({
                        "client_id": str(client["Id"]),
                        "name": client["NombreCompleto"],
                        "type": "client",
                        "carnetIdentidad": client.get("Ci", ""),
                        "email": client.get("Email", ""),
                        "phoneNumber": client.get("NumeroTelf", ""),
                        "address": client.get("Direccion", "")
                    })

                # Save as one file per enterprise
                file_path = f"{export_dir}/clients_enterprise_{enterprise_id}.json"
                with open(file_path, "w") as f:
                    json.dump({
                        "enterprise_ids": str(enterprise_id),
                        "enterprises_names": group.iloc[0]["enterprise_name"],
                        "clients": clients_list
                    }, f, indent=2)
                file_paths.append(file_path)
        else:
            # Individual client files
            for i, client in clients_df.iterrows():
                client_doc = {
                    "client_id": str(client["Id"]),
                    "name": client["NombreCompleto"],
                    "enterprise_id": str(client["enterprise_id"]),
                    "enterprise_name": client["enterprise_name"],
                    "type": "client",
                    "carnetIdentidad": client.get("Ci", ""),
                    "email": client.get("Email", ""),
                    "phoneNumber": client.get("NumeroTelf", ""),
                    "address": client.get("Direccion", "")
                }

                file_path = f"{export_dir}/client_{client['Id']}.json"
                with open(file_path, "w") as f:
                    json.dump(client_doc, f, indent=2)
                file_paths.append(file_path)
    except Exception as e:
        print(f"Error exporting clients: {e}")

    # Export services
    try:
        services_df = pd.read_sql("""
                                  SELECT s.*, e.Nombre as enterprise_name, e.Id as enterprise_id
                                  FROM Services s
                                           JOIN Companies e ON s.CompanyId = e.id
                                  """, db_connection)
        print(f"Exported {len(services_df)} services")

        # Group services by enterprise
        for enterprise_id, group in services_df.groupby("enterprise_id"):
            services_list = []
            for _, service in group.iterrows():
                services_list.append({
                    "service_id": str(service["Id"]),
                    "name": service["NombreServicio"],
                    "description": service.get("Descripcion", ""),
                    "type": "service"
                })

            # Save as one file per enterprise
            file_path = f"{export_dir}/services_enterprise_{enterprise_id}.json"
            with open(file_path, "w") as f:
                json.dump({
                    "enterprise_id": str(enterprise_id),
                    "enterprise_name": group.iloc[0]["enterprise_name"],
                    "services": services_list
                }, f, indent=2)
            file_paths.append(file_path)
    except Exception as e:
        print(f"Error exporting services: {e}")

    # Export protocols
    try:
        protocols_df = pd.read_sql("""SELECT P.Id,
                                             P.NombreProtocolo,
                                             P.Descripcion,
                                             R.CodigoMotivo,
                                             R.NombreMotivo                                         as reason_name,
                                             R.Descripcion                                          as reason_description,
                                             R.Id                                                   as reason_id,
                                             GROUP_CONCAT(CONCAT(PS.CodigoPasoProtocolo, '. ', PS.PasoProtocolo) ORDER
                                                          BY PS.CodigoPasoProtocolo SEPARATOR '\n') as steps
                                      FROM Protocols P
                                               JOIN Reasons R ON P.ReasonId = R.Id
                                               LEFT JOIN ProtocolSteps PS ON P.Id = PS.ProtocolId
                                      GROUP BY P.Id, P.NombreProtocolo, P.Descripcion, R.CodigoMotivo, R.NombreMotivo,
                                               R.Descripcion, R.Id
                                   """, db_connection)
        print(f"Exported {len(protocols_df)} protocols")

        for i, protocol in protocols_df.iterrows():
            # Create markdown content for each protocol
            protocol_content = f"""# {protocol['NombreProtocolo']}

## Description
{protocol['Descripcion']}

## When to Use
{protocol.get('reason_description', 'This protocol should be used when a customer calls about this specific issue.')}

## Steps to Follow
{protocol.get('steps', '1. Greet the customer n2. Identify the issue n3. Follow the appropriate resolution path')}

## Resolution Options
{protocol.get('resolution_options', '- Option 1: Resolve immediately n- Option 2: Escalate to supervisor n- Option 3: Create a ticket for follow-up')}

## Notes
{protocol.get('notes', '')}
"""

            # Save as markdown file
            file_path = f"{export_dir}/protocol_{protocol['Id']}.md"
            with open(file_path, "w") as f:
                f.write(protocol_content)
            file_paths.append(file_path)
    except Exception as e:
        print(f"Error exporting protocols: {e}")

    print(f"Successfully exported {len(file_paths)} files to {export_dir}/")
    return file_paths


def build_vector_index(data_dir="vectordb/knowledge_base", index_dir="./vector_index"):
    """Build a vector index from the exported files"""
    os.makedirs(index_dir, exist_ok=True)

    print(f"Building vector index from {data_dir}")
    documents = SimpleDirectoryReader(data_dir).load_data()
    print(f"Loaded {len(documents)} documents")

    # Explicitly set OpenAI API key for embeddings
    from llama_index.embeddings.openai import OpenAIEmbedding
    import openai

    # Get API key from environment (which should be loaded from .env.local)
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if openai_api_key:
        print("Using OpenAI embeddings")
        openai.api_key = openai_api_key
        embed_model = OpenAIEmbedding(api_key=openai_api_key)
    else:
        # Fallback to local embeddings if no API key available
        print("OpenAI API key not found, using local embeddings instead")
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        # Make sure to install: pip install llama-index-embeddings-huggingface transformers sentence-transformers

    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model
    )

    index.storage_context.persist(index_dir)
    print(f"Vector index built and saved to {index_dir}")

    return index


def main():
    # Database connection setup
    print("Connecting to database...")
    DB_HOST = 'lgg2gx1ha7yp2w0k.cbetxkdyhwsb.us-east-1.rds.amazonaws.com'
    DB_PORT = '3306'
    DB_USER = 'k2me3e6bdmo8hn7u'
    DB_PASSWORD = 'psgz7tfu1ldlfd43'
    DB_NAME = 'up05ekhg5c3zj0wt'

    connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(connection_string)

    # Test connection
    try:
        tables = pd.read_sql("SHOW TABLES", engine)
        print("Database connection successful!")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

    # Export files
    file_paths = export_db_to_files(engine)

    # Build vector index
    build_vector_index()

    print("Vector database creation completed successfully!")


if __name__ == "__main__":
    main()