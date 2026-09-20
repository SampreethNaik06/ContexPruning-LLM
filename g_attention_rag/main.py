import os
import json
import argparse
from typing import Dict, Tuple
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Query Parser (Commented out for now as requested)
# ------------------------------------------------------------------------------
# from src.query_parser import process_query_input

from src.graph_builder import build_graph_from_documents, KnowledgeGraphBuilder


def import_hotpotqa_dataset(
    sample_idx: int = 0,
    supporting_only: bool = True,
    local_path: str = "data/raw/hotpotqa_sample.json",
) -> Tuple[Dict[str, str], Dict]:
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


def visualize_knowledge_graph(
    kg: KnowledgeGraphBuilder,
    title_text: str = "Knowledge Graph",
    output_img_path: str = "data/processed_graph/knowledge_graph.png",
):
    """
    Renders and saves a visual plot of the Knowledge Graph using NetworkX and Matplotlib.
    - Solid dark edges represent extracted SVO triples with relation labels.
    - Dashed lighter edges represent co-occurrence fallbacks.
    - Node colors reflect entity types.
    """
    os.makedirs(os.path.dirname(output_img_path) or ".", exist_ok=True)
    G = kg.graph

    if G.number_of_nodes() == 0:
        print("[Warning] Graph is empty. Skipping image visualization.")
        return

    plt.figure(figsize=(15, 11), dpi=300)
    pos = nx.spring_layout(G, k=1.8, iterations=60, seed=42)

    # Distinguish SVO and co-occurrence edges
    svo_edges = []
    cooccur_edges = []
    svo_labels = {}

    for u, v, data in G.edges(data=True):
        rels = list(data.get("relations", []))
        if "co_occurs" in rels and len(rels) == 1:
            cooccur_edges.append((u, v))
        else:
            svo_edges.append((u, v))
            non_cooccur = [r for r in rels if r != "co_occurs"]
            svo_labels[(u, v)] = ", ".join(non_cooccur) if non_cooccur else "rel"

    # Color palette for entity types
    type_color_map = {
        "ORG": "#a8dadc",       # Soft cyan
        "PERSON": "#f4a261",    # Soft peach/orange
        "GPE": "#ffd166",       # Pastel yellow
        "DATE": "#e76f51",      # Coral red
        "EVENT": "#b7b7a4",     # Muted grey
        "WORK_OF_ART": "#bde0fe", # Light sky blue
        "LOC": "#90be6d",       # Muted green
    }
    node_colors = [
        type_color_map.get(G.nodes[n].get("ent_type"), "#e9ecef")
        for n in G.nodes()
    ]

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=2800,
        edgecolors="#264653",
        linewidths=1.6
    )

    # Draw node labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_weight="bold",
        font_family="sans-serif"
    )

    # Draw co-occurrence edges (secondary fallback - dashed light blue)
    if cooccur_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=cooccur_edges,
            edge_color="#a8dadc",
            style="dashed",
            width=1.0,
            arrows=True,
            arrowsize=10,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.06",
            alpha=0.7
        )

    # Draw SVO edges (primary relations - solid dark blue)
    if svo_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=svo_edges,
            edge_color="#1d3557",
            width=2.2,
            arrows=True,
            arrowsize=16,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.06"
        )
        # SVO relation text labels
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=svo_labels,
            font_size=8,
            font_color="#9d0208",
            label_pos=0.45,
            rotate=False
        )

    plt.title(f"HotpotQA Knowledge Graph Visualization\n{title_text}", fontsize=13, fontweight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_img_path, bbox_inches="tight", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Construction Pipeline with HotpotQA")
    parser.add_argument("--sample-idx", type=int, default=0, help="HotpotQA sample index (default: 0)")
    parser.add_argument("--all-context", action="store_true", help="Use all 10 context paragraphs instead of supporting only")
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Query Parser CLI (Commented out for now as requested)
    # --------------------------------------------------------------------------
    # query_input = args.query.strip() if args.query else input("Enter Query: ").strip()
    # if not query_input:
    #     print(json.dumps({"error": "No query provided."}, indent=4))
    #     sys.exit(1)
    #
    # anchors = process_query_input(
    #     query=query_input, 
    #     save_log=not args.no_log, 
    #     filepath="data/parsed_queries.json"
    # )
    # print("\n[Pipeline In-Memory Anchors]:")
    # print(json.dumps(anchors, indent=4, ensure_ascii=False))
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # 1. Import HotpotQA Dataset
    # --------------------------------------------------------------------------
    output_pkl_path = "data/knowledge_graph.pkl"
    output_json_path = "data/processed_graph/graph_output.json"
    output_img_path = "data/processed_graph/knowledge_graph.png"

    print("=" * 70)
    print("Knowledge Graph Pipeline: Importing HotpotQA Dataset")
    print("=" * 70)

    supporting_only = not args.all_context
    docs, sample_meta = import_hotpotqa_dataset(
        sample_idx=args.sample_idx,
        supporting_only=supporting_only,
    )

    print(f"Dataset      : HotpotQA (distractor)")
    print(f"Sample Index : {args.sample_idx}")
    print(f"Question     : {sample_meta['question']}")
    print(f"Answer       : {sample_meta['answer']}")
    print(f"Type         : {sample_meta['type']}")
    print(f"Documents ({len(docs)}): {list(docs.keys())}\n")

    # --------------------------------------------------------------------------
    # 2. Build Knowledge Graph from Documents
    # --------------------------------------------------------------------------
    print("Building Knowledge Graph from document chunks...")
    kg = build_graph_from_documents(docs, output_path=output_pkl_path)

    # Export graph dictionary to JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    graph_dict = kg.to_dict()
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(graph_dict, f, indent=2)

    # --------------------------------------------------------------------------
    # 3. Generate Visual Image of the Graph
    # --------------------------------------------------------------------------
    title_text = f"Q: {sample_meta['question'][:90]}...\nAnswer: {sample_meta['answer']}"
    visualize_knowledge_graph(kg, title_text=title_text, output_img_path=output_img_path)

    # --------------------------------------------------------------------------
    # 4. Display Graph Statistics & Preview
    # --------------------------------------------------------------------------
    stats = kg.stats()
    print("\n[Graph Statistics]:")
    print(f"  - Nodes: {stats['num_nodes']}")
    print(f"  - Edges: {stats['num_edges']}\n")

    print(f"[Graph Nodes ({stats['num_nodes']})]:")
    for node in graph_dict["nodes"]:
        ent_info = f" (type: {node['ent_type']})" if node["ent_type"] else ""
        print(f"  - {node['id']}{ent_info} | Chunks: {node['chunks']}")

    print(f"\n[Graph Edges ({stats['num_edges']})]:")
    for edge in graph_dict["edges"][:25]:  # Preview first 25 edges
        relations_str = ", ".join(edge["relations"])
        print(
            f"  - {edge['source']} --[{relations_str}]--> {edge['target']} "
            f"(weight: {edge['weight']}, chunks: {edge['chunks']})"
        )
    if stats['num_edges'] > 25:
        print(f"  ... and {stats['num_edges'] - 25} more edges (see full JSON file).")

    print("\n[Saved Artifacts]:")
    print(f"  - NetworkX Binary : {output_pkl_path}")
    print(f"  - JSON Graph Data : {output_json_path}")
    print(f"  - Graph Image     : {output_img_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()