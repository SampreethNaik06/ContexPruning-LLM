# Graph module map

The CLI lives in `g_attention_rag/main.py`. It coordinates build, evaluation,
retrieval, and plotting by calling the stable APIs in `src/graph_builder.py`,
`src/graph_traversal.py`, and `graph_visualization.py`.

## Build flow

```text
main.py
  -> graph_builder.py
       -> corpus_builder.py        corpus aliases, merge, save/load
            -> document_graph.py   build one document into nodes and claims
                 -> graph_text.py
                 -> entity_aliases.py
                 -> claim_extraction.py
                 -> graph_structure.py
       -> graph_serialization.py   JSON export
```

| File | Responsibility |
| --- | --- |
| `graph_constants.py` | Shared schema version and extraction vocabulary |
| `graph_text.py` | Text cleanup, canonical forms, stable IDs, paragraph ranges |
| `entity_aliases.py` | Mention phrase spans and local/corpus alias resolution |
| `claim_extraction.py` | Dependency-based SVO extraction and reference validation |
| `graph_structure.py` | Chunks, nodes, context bridges, traversal backbone, invariants |
| `document_graph.py` | Orchestrates construction for one source document |
| `corpus_builder.py` | Resolves aliases over all sources and merges their graphs |
| `graph_serialization.py` | Converts graph state to JSON and writes exports |
| `graph_builder.py` | Public builder API imported by the CLI and other callers |
| `graph_visualization.py` | Loads a graph view and renders it with the selected layout |
| `main.py` | Parses CLI options and calls the build, retrieval, and display APIs |

## Retrieval flow

```text
main.py
  -> graph_traversal.py
       -> traversal_types.py    anchor matching, polarity key, path records
       -> path_traversal.py     resource flow and bounded path search
       -> claim_retrieval.py    claim deduplication and source evidence
```

`graph_traversal.py` is the retrieval entry point. Path search returns claim IDs;
claim retrieval resolves those IDs against the claim store and attaches the
original document chunks from graph state.
