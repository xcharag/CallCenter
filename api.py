from flask import Flask, jsonify, request, send_file
import subprocess
from dotenv import load_dotenv
from fpdf import FPDF
import json
import os
import tempfile
import vectorDbHandler
from tools import create_database_connection
from sqlalchemy import text
from openai import OpenAI

# Load environment variables
load_dotenv(dotenv_path=".env.local")

# Create Flask app
app = Flask(__name__)

@app.route('/api/rebuild-vector-index', methods=['GET'])
def rebuild_vector_index():

    try:
        vectorDbHandler.main()

        return jsonify({
            "status": "success",
            "message": "Vector index rebuilt successfully"
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


@app.route('/api/evaluate', methods=['POST'])
def evaluate_call():
    try:
        # Get request data
        data = request.json
        if not data or 'call_id' not in data or 'evaluation_prompt' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing required fields: call_id and evaluation_prompt"
            }), 400

        call_id = data['call_id']
        evaluation_prompt = data['evaluation_prompt']

        # Connect to database
        engine = create_database_connection()
        if engine is None:
            return jsonify({
                "status": "error",
                "message": "Failed to connect to database"
            }), 500

        # Get transcript file path
        with engine.connect() as connection:
            query = text("SELECT Grabacion FROM Calls WHERE Id = :call_id")
            result = connection.execute(query, {"call_id": call_id}).fetchone()

        if not result:
            return jsonify({
                "status": "error",
                "message": f"Call with ID {call_id} not found"
            }), 404

        transcript_path = result[0]

        # Load transcript content
        with open(transcript_path, 'r') as f:
            transcript_data = json.load(f)

        # Format conversation for evaluation
        conversation = []
        for item in transcript_data.get("items", []):
            if "content" in item and item["content"]:
                role = "Assistant" if item["role"] == "assistant" else "User"
                content = item["content"][0] if item["content"] else ""
                conversation.append(f"{role}: {content}")

        conversation_text = "\n".join(conversation)

        # Call OpenAI for evaluation
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are an expert call evaluator. Analyze the conversation and provide detailed feedback."},
                {"role": "user",
                 "content": f"Evaluation instructions: {evaluation_prompt}\n\nConversation transcript:\n{conversation_text}"}
            ]
        )

        evaluation_result = response.choices[0].message.content

        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Add title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Reporte de Evaluacion de Llamadas", ln=True, align='C')
        pdf.ln(10)

        # Add evaluation details
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=f"Id de Llamada: {call_id}", ln=True)
        pdf.cell(200, 10, txt="Criterios de Evaluacion:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(190, 10, txt=evaluation_prompt)
        pdf.ln(10)

        # Add evaluation results
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Evaluation Results:", ln=True)
        pdf.set_font("Arial", size=12)

        # Split by paragraphs and add to PDF
        for paragraph in evaluation_result.split("\n\n"):
            pdf.multi_cell(190, 10, txt=paragraph)
            pdf.ln(5)

        # Save PDF to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            pdf_path = tmp.name

        pdf.output(pdf_path)

        # Send PDF file
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=f"evaluation_{call_id}.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Evaluation failed: {str(e)}"
        }), 500

def start_api_server(host='0.0.0.0', port=5555):
    """Start the Flask API server"""
    print(f"API server started on http://{host}:{port}")
    app.run(host=host, port=port)

if __name__ == '__main__':
    start_api_server()