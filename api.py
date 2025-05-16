from flask import Flask, jsonify, request
import subprocess
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env.local")

# Create Flask app
app = Flask(__name__)

@app.route('/api/rebuild-vector-index', methods=['GET'])
def rebuild_vector_index():

    try:
        # Get path to vectorDbHandler.py
        script_path = os.path.abspath("vectorDbHandler.py")

        # Execute the script
        result = subprocess.run(["python", script_path],
                              capture_output=True,
                              text=True,
                              check=True)

        return jsonify({
            "status": "success",
            "message": "Vector index rebuilt successfully",
            "details": result.stdout
        })

    except subprocess.CalledProcessError as e:
        return jsonify({
            "status": "error",
            "message": "Failed to rebuild vector index",
            "details": e.stderr
        }), 500

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }), 500

def start_api_server(host='0.0.0.0', port=5000):
    """Start the Flask API server"""
    print(f"API server started on http://{host}:{port}")
    app.run(host=host, port=port)

if __name__ == '__main__':
    start_api_server()