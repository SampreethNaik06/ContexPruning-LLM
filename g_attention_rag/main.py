"""Command-line orchestration for graph building and retrieval."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

if __package__:
    from .graph_visualization import display_name, load_graph, plot_graph
    from .src.graph_builder import build_graph, export_json
    from .src.graph_traversal import retrieve_with_traversal_details
    from .src.query_parser import StandaloneQueryParser
else:
    from graph_visualization import display_name, load_graph, plot_graph
    from src.graph_builder import build_graph, export_json
    from src.graph_traversal import retrieve_with_traversal_details
    from src.query_parser import StandaloneQueryParser


SCRIPT_DIR = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="UTF-8 text document; if omitted, loads existing data/knowledge_graph.pkl",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--max-chunk-chars", type=int, default=1200)
    parser.add_argument(
        "--layout",
        choices=["shell", "hierarchical", "spring", "kamada"],
        default="shell",
    )
    parser.add_argument("--title", default="ECAP-RAG Knowledge Graph")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--full-graph-plot", action="store_true")
    parser.add_argument("--hide-edge-labels", action="store_true")
    parser.add_argument(
        "--query", type=str, default=None, help="Query string for PathRAG traversal"
    )
    parser.add_argument(
        "--retrieve",
        action="store_true",
        help="Run PathRAG query parsing & claim retrieval",
    )
    args = parser.parse_args(argv)
    if args.max_chunk_chars < 100:
        parser.error("--max-chunk-chars must be at least 100")

    query_text = (args.query or "").strip()
    while not query_text:
        try:
            query_text = input("Enter your query: ").strip()
        except EOFError:
            parser.error("No query received. Enter a query or pass --query '<question>'.")
        if not query_text:
            print("Please enter a non-empty query.")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    graph_path = output / "knowledge_graph.pkl"

    if args.input:
        if not args.input.exists():
            parser.error(f"Input file not found: {args.input}")
        text = args.input.read_text(encoding="utf-8")
        print(f"Building knowledge graph from {args.input.name} ({len(text)} chars)...")
        state = build_graph(
            text, args.input.name,
            args.model, args.max_chunk_chars,
        )
        with graph_path.open("wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        export_json(state, output / "processed_graph/graph_output.json")
    elif graph_path.exists():
        print(f"Loading existing knowledge graph from {graph_path}...")
        with graph_path.open("rb") as handle:
            state = pickle.load(handle)
    else:
        parser.error(
            "No input document provided and no existing data/knowledge_graph.pkl found.\n"
            "Usage: python main.py <document.txt> [--query '...'] [--retrieve]"
        )

    print(json.dumps(state["diagnostics"], indent=2))

    print(f"\n--- PathRAG Traversal for Query: '{query_text}' ---")
    parser_engine = StandaloneQueryParser(args.model)
    parse_result = parser_engine.parse_query(query_text)
    anchors = parse_result["anchors"]
    print(f"Entity Anchors: {anchors['entity_anchors']}")
    print(f"Relation Anchors: {anchors['relation_anchors']}")

    retrieval_res = retrieve_with_traversal_details(
        state,
        entity_anchors=anchors["entity_anchors"],
        relation_anchors=anchors["relation_anchors"],
    )
    claims_out = output / "processed_graph/retrieved_claims.json"
    claims_out.parent.mkdir(parents=True, exist_ok=True)
    claims_out.write_text(
        json.dumps(retrieval_res, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    stats = retrieval_res["stats"]
    print(
        f"Retrieved {len(retrieval_res['claims'])} deduplicated claims "
        f"(dropped {stats['duplicates_dropped']} duplicates across "
        f"{stats['paths_explored']} paths)"
    )
    for i, claim in enumerate(retrieval_res["claims"], 1):
        reliability = claim.get("path_reliability", 1.0)
        print(f"  {i}. [Reliability: {reliability:.3f}] {claim.get('fact', '')}")
    print(f"Saved retrieved claims to {claims_out}")

    plot_graph(
        graph_path, output / "processed_graph/knowledge_graph.png", args.title,
        layout=args.layout, show_edge_labels=not args.hide_edge_labels, show=args.show,
    )
    if args.full_graph_plot:
        plot_graph(
            graph_path, output / "processed_graph/knowledge_graph_full.png", args.title,
            use_traversal=False, layout=args.layout,
            show_edge_labels=not args.hide_edge_labels,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
