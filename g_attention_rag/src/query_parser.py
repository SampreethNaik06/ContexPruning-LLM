import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List
import spacy


class StandaloneQueryParser:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Two-Tier Query Parser.
        Extracts Entity Nodes (NER) and Relation Nodes (Dependency Walk).
        """
        self.nlp = spacy.load(spacy_model)
        
        # Target Named Entity types for Tier 1 Entity Nodes (Expanded)
        self.target_ent_types = {
            "PERSON", "ORG", "WORK_OF_ART", "DATE", 
            "GPE", "LOC", "NORP", "EVENT", "FAC",
            "LAW", "PRODUCT", "LANGUAGE", "MONEY", 
            "QUANTITY", "ORDINAL", "CARDINAL"
        }
        
        # Subtoken fragments, prefixes, and noise terms to exclude from relation anchors
        self.banned_subtokens = {
            # Hyphenated prefixes & suffixes
            "non", "co", "post", "pre", "anti", "pro", "sub", "super", 
            "multi", "inter", "intra", "trans", "over", "under", "re",
            "based", "related", "driven", "centric", "free", "proof", "like",
            
            # Common meta/query filler words
            "type", "kind", "sort", "thing", "item", "etc", "e.g", "i.e"
        }

    def _sanitize_input(self, text: str) -> str:
        """Strips genuine CLI flags and safely splits mid-word hyphens."""
        # 1. Strip real CLI flags (--no-log)
        cleaned = re.sub(r'(?<!\S)--[a-zA-Z][a-zA-Z0-9_-]*', '', text)
        # 2. Replace hyphens between alphanumeric chars with spaces
        cleaned = re.sub(r'(?<=[A-Za-z0-9])-(?=[A-Za-z0-9])', ' ', cleaned)
        return re.sub(r'\s+', ' ', cleaned).strip()

    def parse_query(self, query: str) -> Dict:
        """Parses input query string into Entity Nodes and Relation Nodes."""
        start_time = time.perf_counter()
        
        sanitized_query = self._sanitize_input(query)
        doc = self.nlp(sanitized_query)

        entity_nodes = []
        extracted_ent_tokens = set()

        # --- Tier 1: Extract Entity Nodes (spaCy NER) ---
        for ent in doc.ents:
            if ent.label_ in self.target_ent_types:
                entity_nodes.append({
                    "raw_mention": ent.text,
                    "canonical_node": ent.text.strip(),
                    "type": ent.label_
                })
                for token in ent:
                    extracted_ent_tokens.add(token.i)

        # --- Tier 2: Extract Relation Nodes (Dependency Tree Walk) ---
        relation_nodes = []
        for token in doc:
            lemma = token.lemma_.lower().strip()
            
            if (
                token.i in extracted_ent_tokens 
                or token.is_stop 
                or token.is_punct 
                or token.pos_ == "SYM" 
                or not token.is_alpha
                or len(lemma) <= 2
                or lemma in self.banned_subtokens
            ):
                continue
            
            if token.dep_ in {"nsubj", "dobj", "pobj", "prep", "attr", "ROOT"} or token.pos_ in {"NOUN", "VERB"}:
                relation_nodes.append({
                    "lemma": lemma,
                    "pos": token.pos_,
                    "dep": token.dep_
                })

        execution_latency_ms = (time.perf_counter() - start_time) * 1000

        entity_anchors = list(set([e["canonical_node"] for e in entity_nodes]))
        relation_anchors = list(set([r["lemma"] for r in relation_nodes]))

        anchors_dict = {
            "entity_anchors": entity_anchors,
            "relation_anchors": relation_anchors
        }

        detailed_log = {
            "query": query,
            "latency_ms": round(execution_latency_ms, 3),
            "anchors": anchors_dict,
            "detailed_entities": entity_nodes,
            "detailed_relations": relation_nodes
        }

        return {
            "anchors": anchors_dict,
            "detailed_log": detailed_log
        }

    def save_log_to_disk(self, detailed_log: Dict, output_filepath: str = "data/parsed_queries.json") -> None:
        """Saves evaluation query logs to a JSON file on disk."""
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        if os.path.exists(output_filepath):
            try:
                with open(output_filepath, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = {"metadata": {}, "logs": []}
        else:
            log_data = {"metadata": {}, "logs": []}

        log_data["metadata"] = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_queries_logged": len(log_data["logs"]) + 1
        }
        log_data["logs"].append(detailed_log)

        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)


def process_query_input(query: str, save_log: bool = True, filepath: str = "data/parsed_queries.json") -> Dict[str, List[str]]:
    """Helper function to execute parsing and optional disk logging."""
    parser = StandaloneQueryParser()
    result = parser.parse_query(query)
    
    if save_log:
        parser.save_log_to_disk(result["detailed_log"], output_filepath=filepath)
        
    return result["anchors"]