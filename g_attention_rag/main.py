"""Command-line orchestration for graph building and retrieval."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

if __package__:
    from .src.graph_builder import build_graph, export_json
    from .src.graph_constants import SCHEMA_VERSION
    from .src.graph_traversal import retrieve_with_traversal_details
    from .src.query_parser import StandaloneQueryParser
else:
    from src.graph_builder import build_graph, export_json
    from src.graph_constants import SCHEMA_VERSION
    from src.graph_traversal import retrieve_with_traversal_details
    from src.query_parser import StandaloneQueryParser


SCRIPT_DIR = Path(__file__).resolve().parent


def _graph_entity_mentions(state: dict, query: str) -> list[str]:
    """Recover lowercase multiword entity mentions that spaCy NER can miss."""
    query_words = re.findall(r"[\w'-]+", query.casefold())
    matches: set[str] = set()
    graph = state.get("graph")
    if graph is None:
        return []

    for _, attributes in graph.nodes(data=True):
        names = [attributes.get("canonical_name", ""), *attributes.get("aliases", [])]
        for name in names:
            words = re.findall(r"[\w'-]+", str(name).casefold())
            for size in range(len(words), 1, -1):
                found = False
                for start in range(len(words) - size + 1):
                    phrase_words = words[start:start + size]
                    phrase = " ".join(phrase_words)
                    # Match inflected final words such as "products" to a graph
                    # display normalized as "product".
                    exact = any(
                        query_words[i:i + size] == phrase_words
                        for i in range(len(query_words) - size + 1)
                    )
                    singularized = (
                        len(phrase_words[-1]) > 3
                        and phrase_words[-1].endswith("s")
                        and any(
                            query_words[i:i + size - 1] == phrase_words[:-1]
                            and query_words[i + size - 1] == phrase_words[-1][:-1]
                            for i in range(len(query_words) - size + 1)
                        )
                    )
                    if exact or singularized:
                        matches.add(phrase)
                        found = True
                if found:
                    break

    if not matches:
        return []
    longest = max(len(phrase.split()) for phrase in matches)
    return sorted(phrase for phrase in matches if len(phrase.split()) == longest)


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
        "--query", "--querry", dest="query", type=str, default=None,
        help="Query string for PathRAG traversal",
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
    retrieve = args.retrieve or bool(query_text)
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
        if state.get("schema_version") != SCHEMA_VERSION:
            parser.error(
                "The saved graph uses an older schema. Rebuild it from its source "
                "document before loading it."
            )
    else:
        parser.error(
            "No input document provided and no existing data/knowledge_graph.pkl found.\n"
            "Usage: python main.py <document.txt> [--query '...'] [--retrieve]"
        )

    print(json.dumps(state["diagnostics"], indent=2))

    if retrieve:
        while not query_text:
            try:
                query_text = input("Enter your query: ").strip()
            except EOFError:
                parser.error(
                    "No query received. Enter a query or pass --query '<question>'."
                )
            if not query_text:
                print("Please enter a non-empty query.")

        print(f"\n--- PathRAG Traversal for Query: '{query_text}' ---")
        parser_engine = StandaloneQueryParser(args.model)
        parse_result = parser_engine.parse_query(query_text)
        anchors = parse_result["anchors"]
        graph_entities = _graph_entity_mentions(state, query_text)
        entity_anchors = list(anchors["entity_anchors"])
        for mention in graph_entities:
            if mention.casefold() not in {item.casefold() for item in entity_anchors}:
                entity_anchors.append(mention)
        entity_tokens = {
            word.rstrip("s")
            for mention in graph_entities
            for word in re.findall(r"[\w'-]+", mention.casefold())
        }
        relation_anchors = [
            relation for relation in anchors["relation_anchors"]
            if relation.casefold().rstrip("s") not in entity_tokens
        ]
        print(f"Entity Anchors: {entity_anchors}")
        if graph_entities:
            print(f"Graph entity matches: {graph_entities}")
        print(f"Relation Anchors: {relation_anchors}")

        retrieval_res = retrieve_with_traversal_details(
            state,
            entity_anchors=entity_anchors,
            relation_anchors=relation_anchors,
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
        evidence_chunks = retrieval_res.get("evidence_chunks", [])
        print(f"Matched source chunks: {len(evidence_chunks)}")
        for i, evidence in enumerate(evidence_chunks, 1):
            print(
                f"  Evidence {i}. {evidence.get('source_name')} "
                f"(chunk {evidence.get('chunk_index')}, "
                f"entities={evidence.get('matched_entities')}, "
                f"relations={evidence.get('matched_relations')})"
            )
            print(f"    {evidence.get('text', '')}")
        for i, claim in enumerate(retrieval_res["claims"], 1):
            reliability = claim.get("path_reliability", 1.0)
            print(f"  {i}. [Reliability: {reliability:.3f}] {claim.get('fact', '')}")
        print(f"Saved retrieved claims to {claims_out}")

    try:
        if __package__:
            from .graph_visualization import plot_graph
        else:
            from graph_visualization import plot_graph
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib" and not (exc.name or "").startswith("matplotlib."):
            raise
        print(
            "Skipping graph image: matplotlib is not installed. "
            "Install it with 'python -m pip install matplotlib' to render plots."
        )
    else:
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
