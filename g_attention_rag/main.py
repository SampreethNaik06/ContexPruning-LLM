# End-to-end pipeline CLI / runner script
import argparse
import json
import sys
from src.query_parser import process_query_input


def main():
    parser = argparse.ArgumentParser(description="Query Parser CLI Entrypoint")
    parser.add_argument("query", type=str, nargs="?", help="Input query string")
    parser.add_argument("--no-log", action="store_true", help="Disable saving detailed query log to data/parsed_queries.json")
    args = parser.parse_args()

    # Get query string from command line argument or user prompt
    query_input = args.query.strip() if args.query else input("Enter Query: ").strip()

    if not query_input:
        print(json.dumps({"error": "No query provided."}, indent=4))
        sys.exit(1)

    # Call query parser function
    # Returns nested dictionary in-memory & saves detailed logs to data/parsed_queries.json
    anchors = process_query_input(
        query=query_input, 
        save_log=not args.no_log, 
        filepath="data/parsed_queries.json"
    )

    # Print clean in-memory nested dictionary structure
    print("\n[Pipeline In-Memory Anchors]:")
    print(json.dumps(anchors, indent=4, ensure_ascii=False))
    
    if not args.no_log:
        print("\n[Disk Log]: Detailed query logs updated in 'data/parsed_queries.json'")


if __name__ == "__main__":
    main()