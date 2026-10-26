"""Main graph-building API used by the CLI and downstream modules.

The implementation is split by responsibility:
- ``document_graph`` builds one document.
- ``corpus_builder`` resolves aliases across documents and merges the corpus.
- ``graph_structure`` manages graph connectivity and invariants.
- ``graph_serialization`` exports persisted graph state.
"""

from .claim_extraction import (
    expand_conjuncts,
    extract_sentence_triples,
    meaningful_phrase,
    resolve_reference,
    subjects_for,
    verb_relation,
)
from .corpus_builder import KnowledgeGraphBuilder, build_graph_from_documents
from .document_graph import build_graph
from .entity_aliases import (
    canonical_display,
    canonical_key,
    discover_aliases,
    discover_corpus_aliases,
    entity_covering_token,
    phrase_from_token,
    preferred_display,
)
from .graph_constants import (
    ALLOWED_PREPOSITIONS,
    CO_OCCUR_RELATION,
    CONTEXT_RELATIONS,
    DEICTIC_WORDS,
    DESCRIPTOR_HEADS,
    DIRECT_OBJECT_DEPS,
    GENERIC_EXACT,
    GENERIC_HEADS,
    PRONOUN_WORDS,
    SCHEMA_VERSION,
    SUBJECT_DEPS,
)
from .graph_serialization import export_json, graph_to_dict, json_safe
from .graph_structure import (
    add_entity_node,
    build_traversal_backbone,
    chunk_for_offset,
    ensure_connected,
    graph_diagnostics,
    make_chunks,
    validate_graph_invariants,
)
from .graph_text import (
    canonical_form,
    clean_source_text,
    load_nlp,
    normalized_words,
    paragraph_index_for,
    paragraph_ranges,
    stable_id,
)
