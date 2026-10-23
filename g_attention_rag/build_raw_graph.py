"""Build the existing knowledge graph from all raw PDFs, using OCR for scans."""

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent / ".pdf_dependencies"))

import pymupdf

from eval.evaluate import evaluate
from main import plot_graph
from src.graph_builder import KnowledgeGraphBuilder, export_json
from src import graph_builder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Use only the first N PDFs in filename order")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    output = ROOT / "data/processed_graph"
    output.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT / "data/raw").rglob("*.pdf"))
    if args.limit is not None:
        sources = sources[:args.limit]
    if not sources:
        raise ValueError("No PDFs found in data/raw")
    builder = KnowledgeGraphBuilder()
    graph_builder.canonical_form = lru_cache(maxsize=50000)(graph_builder.canonical_form)
    manifest = []
    for index, path in enumerate(sources, 1):
        pages = []
        ocr_pages = []
        with pymupdf.open(path) as document:
            for number, page in enumerate(document, 1):
                text = page.get_text(sort=True)
                if len(text.strip()) < 30:
                    textpage = page.get_textpage_ocr(
                        language="eng", dpi=200, full=True,
                        tessdata=str(ROOT.parent / ".pdf_dependencies/tessdata"),
                    )
                    text = page.get_text(textpage=textpage, sort=True)
                    ocr_pages.append(number)
                pages.append(re.sub(r"[^\S\n]+", " ", text).strip())
        text = "\n\n".join(pages)
        print(f"[{index}/{len(sources)}] {path.name}: {len(text)} chars, {len(ocr_pages)} OCR pages", flush=True)
        builder.add_chunk(path.name, text)
        manifest.append({"file": path.name, "pages": len(pages), "ocr_pages": ocr_pages, "characters": len(text)})
        print(f"  {builder.stats()}", flush=True)
    assert len(builder.state["documents"]) == len(sources)
    builder.save(str(ROOT / "data/knowledge_graph.pkl"))
    export_json(builder.state, output / "graph_output.json")
    (output / "pdf_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = evaluate(builder.state)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    plot_graph(
        ROOT / "data/knowledge_graph.pkl", output / "knowledge_graph.png",
        f"Financial reports | {len(sources)} PDFs | traversal backbone",
        layout="spring", show_edge_labels=False,
    )
    print(json.dumps(builder.state["diagnostics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
