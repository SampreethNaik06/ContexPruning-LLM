# ContexPruning-LLM

The main project is `g_attention_rag/`. Module 2's SVO graph builder is implemented
directly in `g_attention_rag/src/graph_builder.py`; the separate
`ecap_svo_knowledge_graph_v2/` folder is not needed to run it.

## Run Module 2

```powershell
python -m pip install -r g_attention_rag/requirements.txt
python -m spacy download en_core_web_sm
python g_attention_rag/main.py --sample-idx 0 --strict
python g_attention_rag/main.py --sample-idx 0 --all-context --strict
python g_attention_rag/main.py g_attention_rag/data/quantum_physics.txt --fixture-checks --strict
```

Pass any UTF-8 text file as the positional input. Without a file, the runner uses
the cached HotpotQA sample. `--fixture-checks` adds the five quantum-specific
alias/retrieval checks to the four general structural/provenance checks; use it
only with the included quantum fixture. `--layout` accepts `shell` (default),
`hierarchical`, `spring`, or `kamada`. `--full-graph-plot` also renders every edge.

Outputs default to `g_attention_rag/data/`, regardless of the working directory:

- `knowledge_graph.pkl`: full `MultiDiGraph`, traversal backbone, claims, chunks,
  normalized source documents, aliases, rejected triples, and diagnostics.
- `processed_graph/graph_output.json`: readable export of the complete state.
- `processed_graph/knowledge_graph.png`: sparse traversal visualization.
- `evaluation.json`: validation results.

Use `--output-dir PATH` to keep experimental outputs separate. Existing graph
caches use the older schema and must be rebuilt. Source spans index the stored,
normalized document text; HotpotQA sources retain independent document offsets.
Inferred context bridges connect components but are not factual claims.

## Development checks

```powershell
python -B -m unittest discover -s g_attention_rag/tests
```

The tests check multi-document claim preservation, passive normalization,
provenance, retrieval, pickle/JSON round trips, query anchors, and query logging. The quantum command above
runs the original nine Module 2 acceptance checks. Module 1 remains in
`src/query_parser.py`; Modules 3 onward remain future work. Retrieval helpers in
`eval/evaluate.py` serve Module 2 validation, not the completed Module 3 pipeline.
