"""JSON-safe graph serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx


def json_safe(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(json_safe(item) for item in value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value

def graph_to_dict(state: dict[str, Any]) -> dict[str, Any]:
    graph: nx.MultiDiGraph = state["graph"]
    traversal_graph: nx.MultiDiGraph = state["traversal_graph"]
    payload = {
        key: value for key, value in state.items()
        if key not in {"graph", "traversal_graph"}
    }
    payload["nodes"] = [
        {"id": node_id, **json_safe(attributes)}
        for node_id, attributes in graph.nodes(data=True)
    ]
    payload["edges"] = [
        {"source": source, "target": target, "key": key, **json_safe(attributes)}
        for source, target, key, attributes in graph.edges(keys=True, data=True)
    ]
    payload["traversal_nodes"] = [
        {"id": node_id, **json_safe(attributes)}
        for node_id, attributes in traversal_graph.nodes(data=True)
    ]
    payload["traversal_edges"] = [
        {"source": source, "target": target, "key": key, **json_safe(attributes)}
        for source, target, key, attributes in traversal_graph.edges(keys=True, data=True)
    ]
    return json_safe(payload)

def export_json(state: dict[str, Any], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(graph_to_dict(state), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
