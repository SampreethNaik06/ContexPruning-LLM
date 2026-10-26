"""Graph layout and rendering helpers used by the command-line entry point."""

from __future__ import annotations

import pickle
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

if __package__:
    from .src.graph_builder import CO_OCCUR_RELATION, CONTEXT_RELATIONS
else:
    from src.graph_builder import CO_OCCUR_RELATION, CONTEXT_RELATIONS


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
