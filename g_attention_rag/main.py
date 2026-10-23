"""Build, evaluate, and plot the Module 2 ECAP knowledge graph."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

if __package__:
    from .eval.evaluate import evaluate
    from .src.graph_builder import (
        CONTEXT_RELATIONS, CO_OCCUR_RELATION, build_graph, build_graph_from_documents, export_json,
    )
else:
    from eval.evaluate import evaluate
    from src.graph_builder import (
        CONTEXT_RELATIONS, CO_OCCUR_RELATION, build_graph, build_graph_from_documents, export_json,
    )

SCRIPT_DIR = Path(__file__).resolve().parent

def import_hotpotqa_dataset(
    sample_idx: int = 0,
    supporting_only: bool = True,
    local_path: str = "data/raw/hotpotqa_sample.json",
) -> tuple[dict[str, str], dict]:
    """
    Imports and extracts documents from the HotpotQA dataset.
    Loads from local cache (data/raw/hotpotqa_sample.json) if available,
    otherwise downloads samples using the `datasets` library.
    """
    samples = []
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
    else:
        print("[HotpotQA] Local sample cache not found. Fetching from Hugging Face 'hotpotqa/hotpot_qa'...")
        from datasets import load_dataset
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", streaming=True)
        for i, item in enumerate(ds):
            if i >= 10:
                break
            ctx_titles = list(item["context"]["title"])
            ctx_sentences = [list(s) for s in item["context"]["sentences"]]
            samples.append({
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "type": item["type"],
                "level": item.get("level", ""),
                "supporting_facts": {
                    "title": list(item["supporting_facts"]["title"]),
                    "sent_id": [int(x) for x in item["supporting_facts"]["sent_id"]],
                },
                "context": {
                    "title": ctx_titles,
                    "sentences": ctx_sentences,
                },
            })
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)
        print(f"[HotpotQA] Saved {len(samples)} samples to {local_path}")

    if not samples:
        raise ValueError("HotpotQA sample cache is empty")

    # Select requested sample
    idx = max(0, min(sample_idx, len(samples) - 1))
    sample = samples[idx]

    gold_titles = set(sample["supporting_facts"]["title"])
    docs = {}
    titles = sample["context"]["title"]
    sentences_list = sample["context"]["sentences"]

    for title, sentences in zip(titles, sentences_list):
        if supporting_only and title not in gold_titles:
            continue
        docs[title] = f"{title}. " + " ".join(sentences)

    return docs, sample


ENT_TYPE_COLORS = {
    "PERSON": "#E89344",
    "ORG": "#94C6D4",
    "GPE": "#F0AD4E",
    "LOC": "#F0AD4E",
    "DATE": "#C94A42",
    "TIME": "#C94A42",
    "EVENT": "#F3C151",
    "WORK_OF_ART": "#F3C151",
    "NORP": "#B565A7",
    "FAC": "#8FBF8F",
    None: "#B0C4DE",
}


def load_graph(path: Path, use_traversal: bool = True) -> nx.MultiDiGraph:
    with path.open("rb") as handle:
        state = pickle.load(handle)
    graph = state.get("traversal_graph", state["graph"]) if use_traversal else state["graph"]
    if isinstance(graph, nx.MultiDiGraph):
        return graph
    # Backward compatibility with the user's earlier DiGraph pickle.
    upgraded = nx.MultiDiGraph()
    upgraded.add_nodes_from(graph.nodes(data=True))
    for source, target, data in graph.edges(data=True):
        relations = data.get("relations", {data.get("relation", "related_to")})
        for relation in relations:
            upgraded.add_edge(source, target, **{**data, "relation": relation})
    return upgraded


def display_name(graph: nx.MultiDiGraph, node: str) -> str:
    label = graph.nodes[node].get("canonical_name", node)
    return "\n".join(textwrap.wrap(str(label), width=22))


def plot_graph(
    graph_path: Path,
    output_path: Path,
    title: str,
    use_traversal: bool = True,
    layout: str = "kamada",
    show_edge_labels: bool = True,
    show_cooccurrence: bool = True,
    seed: int = 42,
    figsize: tuple[int, int] = (24, 18),
    show: bool = False,
) -> None:
    if not show:
        plt.switch_backend("Agg")
    graph = load_graph(graph_path, use_traversal=use_traversal)
    view = graph.copy()
    if not show_cooccurrence:
        weak = [(u, v, k) for u, v, k, d in view.edges(keys=True, data=True)
                if d.get("relation") == CO_OCCUR_RELATION]
        view.remove_edges_from(weak)
        view.remove_nodes_from(list(nx.isolates(view)))

    if view.number_of_nodes() == 0:
        raise ValueError("The graph has no plottable nodes.")

    if layout == "hierarchical":
        root = view.graph.get("topic_node", next(iter(view)))
        levels = dict(nx.single_source_shortest_path_length(view.to_undirected(), root))
        layers = defaultdict(list)
        for node in view:
            layers[levels.get(node, max(levels.values(), default=0) + 1)].append(node)
        pos = {node: (index - (len(nodes) - 1) / 2, -level)
               for level, nodes in layers.items()
               for index, node in enumerate(nodes)}
    elif layout == "spring":
        spacing = max(0.7, 2.2 / max(view.number_of_nodes() ** 0.5, 1))
        pos = nx.spring_layout(view, seed=seed, k=spacing, iterations=350, weight="weight")
    elif layout == "shell":
        topic = view.graph.get("topic_node")
        if topic in view:
            distances = nx.single_source_shortest_path_length(view.to_undirected(), topic)
            shells = [
                [node for node, distance in distances.items() if distance == level]
                for level in range(max(distances.values(), default=0) + 1)
            ]
            pos = nx.shell_layout(view, nlist=[shell for shell in shells if shell])
        else:
            pos = nx.kamada_kawai_layout(view, weight="weight")
    else:
        pos = nx.kamada_kawai_layout(view, weight="weight")
    fig, axis = plt.subplots(figsize=figsize)

    colors = [ENT_TYPE_COLORS.get(view.nodes[n].get("ent_type"), ENT_TYPE_COLORS[None])
              for n in view.nodes]
    degrees = dict(view.degree())
    sizes = [900 + min(1600, 110 * degrees.get(node, 0)) for node in view.nodes]
    nx.draw_networkx_nodes(
        view, pos, ax=axis, node_color=colors, node_size=sizes, alpha=0.92,
        edgecolors="#333333", linewidths=0.8,
    )
    nx.draw_networkx_labels(
        view, pos, ax=axis,
        labels={node: display_name(view, node) for node in view.nodes},
        font_size=7, font_weight="bold",
    )

    real_edges, weak_edges, context_edges = [], [], []
    for source, target, key, data in view.edges(keys=True, data=True):
        item = (source, target, key)
        if data.get("edge_type") == "document_context" or data.get("relation") in CONTEXT_RELATIONS:
            context_edges.append(item)
        elif data.get("relation") == CO_OCCUR_RELATION:
            weak_edges.append(item)
        else:
            real_edges.append(item)

    if real_edges:
        widths = [0.8 + 2.0 * float(view[u][v][k].get("confidence", 0.5))
                  for u, v, k in real_edges]
        nx.draw_networkx_edges(
            view, pos, ax=axis, edgelist=real_edges, width=widths,
            edge_color="#4A6FA5", alpha=0.68, arrows=True,
            arrowstyle="-|>", arrowsize=13, connectionstyle="arc3,rad=0.07",
            min_source_margin=14, min_target_margin=14,
        )
    if weak_edges:
        nx.draw_networkx_edges(
            view, pos, ax=axis, edgelist=weak_edges, width=0.7,
            edge_color="#AAAAAA", alpha=0.45, arrows=False, style="dashed",
        )
    if context_edges:
        nx.draw_networkx_edges(
            view, pos, ax=axis, edgelist=context_edges, width=1.0,
            edge_color="#6B7280", alpha=0.55, arrows=True,
            arrowstyle="-|>", arrowsize=10, style="dotted",
            connectionstyle="arc3,rad=0.03",
            min_source_margin=14, min_target_margin=14,
        )

    if show_edge_labels:
        relation_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for source, target, key in real_edges:
            relation = view[source][target][key].get("relation", "related_to")
            if relation not in relation_groups[(source, target)]:
                relation_groups[(source, target)].append(relation)
        labels = {
            pair: "\n".join(relation.replace("_", " ") for relation in relations[:3])
            for pair, relations in relation_groups.items()
        }
        nx.draw_networkx_edge_labels(
            view, pos, ax=axis, edge_labels=labels, font_size=6,
            font_color="#7A1F1F", rotate=False, label_pos=0.55,
        )

    present_types = sorted(
        {view.nodes[node].get("ent_type") for node in view.nodes},
        key=lambda value: value or "ZZZ",
    )
    handles = [
        mpatches.Patch(color=ENT_TYPE_COLORS.get(kind, ENT_TYPE_COLORS[None]),
                       label=kind or "concept / unlabeled entity")
        for kind in present_types
    ]
    if weak_edges:
        handles.append(mpatches.Patch(facecolor="white", edgecolor="#AAAAAA",
                                      label="dashed: weak co-occurrence"))
    if context_edges:
        handles.append(mpatches.Patch(facecolor="white", edgecolor="#6B7280",
                                      label="dotted: inferred document context"))
    axis.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)

    components = list(nx.weakly_connected_components(view))
    axis.set_title(
        f"{title}\n{view.number_of_nodes()} nodes • {len(real_edges)} factual claims • "
        f"{len(context_edges)} context bridges • {len(components)} component(s)",
        fontsize=14, fontweight="bold",
    )
    axis.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved graph plot to {output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="UTF-8 text document; otherwise use HotpotQA")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--all-context", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--max-chunk-chars", type=int, default=1200)
    parser.add_argument("--layout", choices=["shell", "hierarchical", "spring", "kamada"], default="shell")
    parser.add_argument("--title", default="ECAP-RAG Knowledge Graph")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--full-graph-plot", action="store_true")
    parser.add_argument("--hide-edge-labels", action="store_true")
    parser.add_argument("--fixture-checks", action="store_true", help="Also run the quantum-physics alias/retrieval checks")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when an evaluation check fails")
    args = parser.parse_args(argv)
    if args.max_chunk_chars < 100:
        parser.error("--max-chunk-chars must be at least 100")
    if args.strict and args.skip_evaluation:
        parser.error("--strict requires evaluation")
    if args.input and (args.all_context or args.sample_idx != 0):
        parser.error("HotpotQA options cannot be combined with a text input")

    if args.input:
        state = build_graph(
            args.input.read_text(encoding="utf-8"), args.input.name,
            args.model, args.max_chunk_chars,
        )
    else:
        documents, sample = import_hotpotqa_dataset(
            args.sample_idx, not args.all_context,
            str(SCRIPT_DIR / "data/raw/hotpotqa_sample.json"),
        )
        print(f"HotpotQA: {sample['question']}")
        state = build_graph_from_documents(
            documents, output_path=None, spacy_model=args.model,
            max_chunk_chars=args.max_chunk_chars,
        ).state

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    graph_path = output / "knowledge_graph.pkl"
    with graph_path.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    export_json(state, output / "processed_graph/graph_output.json")
    print(json.dumps(state["diagnostics"], indent=2))

    report = None
    if not args.skip_evaluation:
        report = evaluate(state, fixture_checks=args.fixture_checks)
        (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        for result in report["results"]:
            print(f"[{'PASS' if result['passed'] else 'FAIL'}] {result['name']}: {result['detail']}")
        print(f"Passed {report['passed']}/{report['total']} checks")

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
    return int(args.strict and report is not None and not report["all_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
