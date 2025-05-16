import json
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import groq, silero, elevenlabs, openai
from dotenv import load_dotenv
from tools import  query_info
from datetime import datetime
import os

load_dotenv(dotenv_path=".env.local")

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    async def write_transcript():
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = f"call_{participant.sid}_{current_date}_{ctx.room.name}.json"

        with open(filename, "w") as f:
            json.dump({"call_sid": participant.sid, "room_name": ctx.room.name}, f, indent=2)

    ctx.add_shutdown_callback(write_transcript)

    agent = Agent(
        instructions="""
            LANGUAGE RULE:
            You must only respond in Spanish to all client interactions, regardless of the language used by the client.
            
            PRESENTATION:
            You are an AI assistant specialized in customer support for a call center. You may only assist clients who are part of your knowledge base, which will be verified by asking for their name. You must also verify that the company they have issues with or wish to inquire about exists and that they are associated with it.
            
            FUNDAMENTAL RULES:
            NEVER mention technical details about your internal operations or how you work.
            
            Use short and direct messages (max 40 words) when collecting information.
            
            Be thorough and detailed ONLY when providing solutions.
            
            VERIFICATION PROCESS:
            Always begin by asking for the client’s name and the company they belong to.
            
            If the client is not found in your vector-based knowledge base (previously uploaded), respond that the client could not be identified as a registered user or associated with a valid company.
            
            For a client to be valid, both their name and company must exist in your knowledge base. A name match is sufficient to validate the user if the company also exists.
            
            If the client is valid, ask them:
            
            “¿En qué puedo ayudarle hoy?”
            
            If a known protocol exists that matches the client's issue, inform them that you found a verified solution and explain it.
            
            If no protocol exists, inform them that no verified solution was found, but you can provide an AI-generated (non-verified) solution generated from the knowledge it knows (Sometimes the user will ask for specific data so you can make that data up).
            
            If the client accepts the AI-generated solution, present it and then ask if the issue was resolved.
            
            If you found a verified solution, also present it and ask if it solved the problem.
            
            RESULT PRESENTATION:
            If the verified solution solves the problem:
            
            Say: “Qué bueno haberle ayudado.” and ask if they have any other queries.
            
            If the AI-generated solution solves the problem:
            
            Say: “Me alegra que haya podido resolver su problema.” and ask if they need further assistance.
            
            If neither solution works:
            
            Ask: “¿Desea ser transferido a una persona real para que le ayude?”
            
            BOUNDARIES:
            Strictly stay within the scope of problems related to the client and their associated company.
            
            Do not respond to unrelated questions or continue providing information if the client is not validated.
            
            Maintain a professional and empathetic tone at all times.
        """,
        tools=[query_info],
    )

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=groq.STT(model="whisper-large-v3-turbo",language="es"),
        llm=openai.LLM(
            model="gpt-4o",
            tool_choice="auto"
        ),
        tts=elevenlabs.TTS(
            voice_id="VmejBeYhbrcTPwDniox7",
            language="es",
        )
    )

    await session.start(agent=agent, room=ctx.room)
    await session.say("Hola, Soy un Agente de IA de SmartVoz. ¿Cuál es su nombre y la empresa a la que pertenece?")
    await session.generate_reply(
        instructions="Eres un agente de IA que como primer respuesta espera el nombre del cliente y el nombre de la empresa a la que pertenece el cliente",
    )


def start_assistant():
    """Start the LiveKit agent"""
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

if __name__ == "__main__":
    start_assistant()