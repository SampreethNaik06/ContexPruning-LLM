import argparse
import json
import sys
from src.query_parser import process_query_input


def main():
    parser = argparse.ArgumentParser(description="Query Parser CLI Entrypoint")
    parser.add_argument("query", type=str, nargs="?", help="Input query string")
    parser.add_argument("--no-log", action="store_true", help="Disable saving detailed log to data/parsed_queries.json")
    args = parser.parse_args()

    query_input = args.query.strip() if args.query else input("Enter Query: ").strip()

    if not query_input:
        print(json.dumps({"error": "No query provided."}, indent=4))
        sys.exit(1)

    # Process query
    anchors = process_query_input(
        query=query_input, 
        save_log=not args.no_log, 
        filepath="data/parsed_queries.json"
    )

    print("\n[Pipeline In-Memory Anchors]:")
    print(json.dumps(anchors, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()