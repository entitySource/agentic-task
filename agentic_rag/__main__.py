"""python -m agentic_rag [--port 8501] [--host 127.0.0.1]"""

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(prog="agentic_rag", description="Run the Agentic RAG web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    from .server import create_app

    print(f"Agentic RAG is starting on http://localhost:{args.port}")
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
