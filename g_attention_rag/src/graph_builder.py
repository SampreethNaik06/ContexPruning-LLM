import os
import re
import pickle
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import spacy


class KnowledgeGraphBuilder:
    """
    Week 2: Knowledge Graph Indexer (src/graph_builder.py).

    Builds a CPU-side directed graph from text chunks using SVO-triple
    extraction (typed, directed edges — e.g. "Microsoft -acquired-> Nuance"),
    with passive-voice normalization so "Nuance was acquired by Microsoft"
    produces the same triple as the active form.

    Falls back to lower-weighted co-occurrence edges for entity pairs that
    share a chunk but have no direct SVO relation, so multi-hop traversal
    (Week 3) still has *something* to walk between loosely related entities,
    without treating co-mention as equally reliable as a real relation.
    """

    TARGET_ENT_TYPES = {
        "PERSON", "ORG", "WORK_OF_ART", "DATE",
        "GPE", "LOC", "NORP", "EVENT", "FAC",
    }

    SUBJ_DEPS = {"nsubj", "nsubjpass"}
    OBJ_DEPS = {"dobj", "attr", "oprd"}
    PASSIVE_SUBJ_DEP = "nsubjpass"

    SVO_EDGE_WEIGHT = 1.0
    COOCCUR_EDGE_WEIGHT = 0.25  # deliberately lower than a real SVO edge

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)
        self.graph = nx.DiGraph()
        # canonical_key -> chosen display node_id, so "Microsoft" and
        # "microsoft's" and "the Microsoft" all resolve to one graph node.
        self._alias_index: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Canonicalization — keeps node identity stable across mentions.
    # NOTE: this normalization should stay in sync with however Week 3
    # matches query anchors onto these node ids (same lowercasing /
    # article-stripping / possessive-stripping rules on both sides),
    # or anchor-to-node lookup will silently miss matches.
    # ------------------------------------------------------------------
    def _canonical_key(self, text: str) -> str:
        cleaned = text.strip().lower()
        cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)
        cleaned = re.sub(r"['\u2019]s\b", "", cleaned)   # drop possessive 's
        cleaned = re.sub(r"[^\w\s.\-]", "", cleaned)      # strip stray punctuation
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _resolve_node_id(self, display_text: str) -> str:
        key = self._canonical_key(display_text)
        if not key:
            key = display_text.strip().lower()
        if key not in self._alias_index:
            self._alias_index[key] = display_text.strip()
        return self._alias_index[key]

    # ------------------------------------------------------------------
    # Tier 1: Entity spans — reused for node typing and co-occurrence edges
    # ------------------------------------------------------------------
    def _extract_entities(self, doc) -> List[Tuple[str, str, int, int]]:
        return [
            (ent.text, ent.label_, ent.start_char, ent.end_char)
            for ent in doc.ents
            if ent.label_ in self.TARGET_ENT_TYPES
        ]

    # ------------------------------------------------------------------
    # Tier 2: SVO triple extraction with passive-voice normalization
    # and basic coordination ("Microsoft and Google acquired X")
    # ------------------------------------------------------------------
    def _span_for_token(self, token) -> str:
        """Expands a subject/object token to its enclosing noun chunk."""
        for chunk in token.doc.noun_chunks:
            if chunk.start <= token.i < chunk.end:
                return chunk.text
        span_tokens = sorted(
            (t for t in token.subtree if t.dep_ in {"compound", "amod", "poss"} or t.i == token.i),
            key=lambda t: t.i,
        )
        return " ".join(t.text for t in span_tokens) if span_tokens else token.text

    def _expand_conjuncts(self, token):
        """Returns [token] plus coordinated siblings joined via 'conj' of the same POS."""
        result = [token]
        for child in token.children:
            if child.dep_ == "conj" and child.pos_ == token.pos_:
                result.append(child)
        return result

    def _find_agent(self, verb_token) -> Optional[str]:
        """For passives, finds the 'by <agent>' phrase attached to the verb."""
        for child in verb_token.children:
            if child.dep_ == "agent":
                pobj = next((gc for gc in child.children if gc.dep_ == "pobj"), None)
                if pobj is not None:
                    return self._span_for_token(pobj)
        return None

    def _extract_svo_triples(self, doc) -> List[Tuple[str, str, str]]:
        """
        Returns (subject_text, relation_lemma, object_text) triples.
        Passive constructions are normalized into active-voice order:
            "Nuance was acquired by Microsoft" -> (Microsoft, acquire, Nuance)
        Passives with no explicit "by <agent>" are skipped rather than
        guessed, since there's no reliable way to recover direction.
        """
        triples = []
        for sent in doc.sents:
            for token in sent:
                if token.pos_ != "VERB":
                    continue

                is_passive = any(c.dep_ == self.PASSIVE_SUBJ_DEP for c in token.children)
                subj_token = next((c for c in token.children if c.dep_ in self.SUBJ_DEPS), None)
                obj_token = next((c for c in token.children if c.dep_ in self.OBJ_DEPS), None)

                # fall back to a prepositional object if there's no direct object
                # e.g. "the report was submitted to the board"
                if obj_token is None:
                    for child in token.children:
                        if child.dep_ == "prep":
                            pobj = next((gc for gc in child.children if gc.dep_ == "pobj"), None)
                            if pobj is not None:
                                obj_token = pobj
                                break

                if subj_token is None or obj_token is None:
                    continue

                relation = token.lemma_.lower()
                subj_variants = self._expand_conjuncts(subj_token)
                obj_variants = self._expand_conjuncts(obj_token)

                if is_passive:
                    agent_text = self._find_agent(token)
                    if agent_text is None:
                        continue
                    for obj_tok in subj_variants:  # nsubjpass tokens become objects
                        triples.append((agent_text, relation, self._span_for_token(obj_tok)))
                else:
                    for s_tok in subj_variants:
                        for o_tok in obj_variants:
                            triples.append(
                                (self._span_for_token(s_tok), relation, self._span_for_token(o_tok))
                            )

        return triples

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------
    def add_chunk(self, chunk_id: str, text: str) -> None:
        doc = self.nlp(text)
        entities = self._extract_entities(doc)

        # --- typed SVO edges (primary signal) ---
        for subj_text, relation, obj_text in self._extract_svo_triples(doc):
            subj_id = self._resolve_node_id(subj_text)
            obj_id = self._resolve_node_id(obj_text)
            if subj_id == obj_id:
                continue
            self._add_or_update_node(subj_id, chunk_id)
            self._add_or_update_node(obj_id, chunk_id)
            self._add_or_update_edge(subj_id, obj_id, relation, chunk_id, weight=self.SVO_EDGE_WEIGHT)

        # --- register every named entity as a node, even if no SVO triple
        #     involved it, so it's still reachable during traversal ---
        for text_span, ent_type, _, _ in entities:
            node_id = self._resolve_node_id(text_span)
            self._add_or_update_node(node_id, chunk_id, ent_type=ent_type)

        # --- co-occurrence fallback edges (lower weight), only added
        #     between entity pairs with no existing direct edge ---
        entity_node_ids = sorted({self._resolve_node_id(t) for t, *_ in entities})
        for a, b in combinations(entity_node_ids, 2):
            if self.graph.has_edge(a, b) or self.graph.has_edge(b, a):
                continue
            self._add_or_update_edge(a, b, relation="co_occurs", chunk_id=chunk_id,
                                      weight=self.COOCCUR_EDGE_WEIGHT)

    def _add_or_update_node(self, node_id: str, chunk_id: str, ent_type: Optional[str] = None) -> None:
        if node_id not in self.graph:
            self.graph.add_node(node_id, chunks=set(), ent_type=ent_type)
        self.graph.nodes[node_id]["chunks"].add(chunk_id)
        if ent_type and not self.graph.nodes[node_id].get("ent_type"):
            self.graph.nodes[node_id]["ent_type"] = ent_type

    def _add_or_update_edge(self, u: str, v: str, relation: str, chunk_id: str, weight: float) -> None:
        if self.graph.has_edge(u, v):
            data = self.graph[u][v]
            data["relations"].add(relation)
            data["chunks"].add(chunk_id)
            data["weight"] = data.get("weight", 0.0) + weight  # frequency-accumulated
        else:
            self.graph.add_edge(u, v, relations={relation}, chunks={chunk_id}, weight=weight)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, filepath: str = "data/knowledge_graph.pkl") -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"graph": self.graph, "alias_index": self._alias_index}, f)

    @classmethod
    def load(cls, filepath: str = "data/knowledge_graph.pkl", spacy_model: str = "en_core_web_sm") -> "KnowledgeGraphBuilder":
        builder = cls(spacy_model=spacy_model)
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        builder.graph = state["graph"]
        builder._alias_index = state["alias_index"]
        return builder

    def stats(self) -> Dict:
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
        }

    def to_dict(self) -> Dict:
        nodes_data = [
            {
                "id": node_id,
                "ent_type": data.get("ent_type"),
                "chunks": sorted(list(data.get("chunks", set()))),
            }
            for node_id, data in self.graph.nodes(data=True)
        ]
        edges_data = [
            {
                "source": u,
                "target": v,
                "relations": sorted(list(data.get("relations", set()))),
                "weight": round(data.get("weight", 0.0), 3),
                "chunks": sorted(list(data.get("chunks", set()))),
            }
            for u, v, data in self.graph.edges(data=True)
        ]
        return {
            "stats": self.stats(),
            "nodes": nodes_data,
            "edges": edges_data,
        }


def chunk_document(text: str, doc_id: str, chunk_size_words: int = 250) -> List[Tuple[str, str]]:
    """Splits raw text into ~chunk_size_words chunks with stable ids."""
    words = text.split()
    chunks = []
    for i, start in enumerate(range(0, len(words), chunk_size_words)):
        chunk_words = words[start:start + chunk_size_words]
        chunk_id = f"{doc_id}_chunk_{i}"
        chunks.append((chunk_id, " ".join(chunk_words)))
    return chunks


def build_graph_from_documents(
    documents: Dict[str, str],
    output_path: str = "data/knowledge_graph.pkl",
    chunk_size_words: int = 250,
) -> KnowledgeGraphBuilder:
    """documents: {doc_id: raw_text}. Chunks each doc and indexes it into one graph."""
    builder = KnowledgeGraphBuilder()
    for doc_id, text in documents.items():
        for chunk_id, chunk_text in chunk_document(text, doc_id, chunk_size_words):
            builder.add_chunk(chunk_id, chunk_text)
    builder.save(output_path)
    return builder
