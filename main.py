import threading
import sys
import os
from api import start_api_server


def main():
    """Main entry point that runs both the API server and LiveKit agent"""
    # Start API server in a background thread
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()

    print("Both services starting. Use Ctrl+C to stop.")

    # Add the "start" argument for the LiveKit CLI
    sys.argv = [sys.argv[0], "start"]

    # Import assistant here to avoid circular imports
    from assistant import start_assistant

    # Start LiveKit agent in the main thread
    start_assistant()


if __name__ == "__main__":
    main()