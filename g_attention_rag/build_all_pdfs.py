"""Build and serialize one traversable knowledge graph from extracted PDF text."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from src.graph_builder import build_graph_from_documents, export_json


SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text-dir", type=Path, default=SCRIPT_DIR / "data/raw/all_pdfs_text",
        help="Directory containing UTF-8 text files extracted from the source PDFs",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR / "data",
        help="Directory for the graph pickle and JSON export",
    )
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--max-chunk-chars", type=int, default=1200)
    args = parser.parse_args()

    text_files = sorted(args.text_dir.glob("*.txt"))
    if not text_files:
        parser.error(f"No extracted .txt files found in {args.text_dir}")
    documents = {
        path.stem + ".pdf": path.read_text(encoding="utf-8")
        for path in text_files
    }
    if any(not text.strip() for text in documents.values()):
        empty = [name for name, text in documents.items() if not text.strip()]
        parser.error(f"Empty extracted documents: {', '.join(empty)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building a corpus graph from {len(documents)} PDF documents...")
    builder = build_graph_from_documents(
        documents,
        output_path=None,
        spacy_model=args.model,
        max_chunk_chars=args.max_chunk_chars,
    )
    state = builder.state
    if state is None:
        raise RuntimeError("The graph builder returned no graph state")

    graph_path = args.output_dir / "knowledge_graph.pkl"
    with graph_path.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    json_path = args.output_dir / "processed_graph/graph_output.json"
    export_json(state, json_path)
    print(f"Sources: {len(state['documents'])}")
    print(f"Nodes: {state['graph'].number_of_nodes()}")
    print(f"Edges: {state['graph'].number_of_edges()}")
    print(f"Claims: {len(state['claims'])}")
    print(f"Graph: {graph_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
