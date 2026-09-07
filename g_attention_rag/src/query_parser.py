import json
import os
import time
from datetime import datetime
from typing import Dict, List, Set
import spacy


class StandaloneQueryParser:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Two-Tier Query Parser (Fix 1).
        Extracts Entity Nodes (NER) and Relation Nodes (Dependency Walk).
        """
        self.nlp = spacy.load(spacy_model)
        
        # Target Named Entity types for Tier 1 Entity Nodes
        self.target_ent_types = {
            "PERSON", "ORG", "WORK_OF_ART", "DATE", 
            "GPE", "LOC", "NORP", "EVENT", "FAC"
        }

    def parse_query(self, query: str) -> Dict:
        """
        Parses an input query string.
        
        Returns:
            Dict containing:
                - anchors: In-memory nested dictionary {"entity_anchors": [...], "relation_anchors": [...]}
                - detailed_log: Metadata & detailed token info for JSON disk logging
        """
        start_time = time.perf_counter()
        doc = self.nlp(query)
        
        entity_nodes = []
        extracted_ent_tokens = set()

        # --- Tier 1: Extract Entity Nodes (spaCy NER) ---
        for ent in doc.ents:
            if ent.label_ in self.target_ent_types:
                entity_nodes.append({
                    "raw_mention": ent.text,
                    "canonical_node": ent.text,
                    "type": ent.label_
                })
                # Record token indices part of named entities to avoid duplicate relation extraction
                for token in ent:
                    extracted_ent_tokens.add(token.i)

        # --- Tier 2: Extract Relation Nodes (Dependency Tree Walk) ---
        relation_nodes = []
        for token in doc:
            # Skip punctuation, stop words, or tokens already inside extracted named entities
            if token.i in extracted_ent_tokens or token.is_stop or token.is_punct:
                continue
            
            # Identify core relation head terms (nouns, verbs, prepositions)
            if token.dep_ in {"nsubj", "dobj", "pobj", "prep", "attr", "ROOT"} or token.pos_ in {"NOUN", "VERB"}:
                relation_nodes.append({
                    "lemma": token.lemma_.lower(),
                    "pos": token.pos_,
                    "dep": token.dep_
                })

        execution_latency_ms = (time.perf_counter() - start_time) * 1000

        # Deduplicated lists for in-memory pipeline execution
        entity_anchors = list(set([e["canonical_node"] for e in entity_nodes]))
        relation_anchors = list(set([r["lemma"] for r in relation_nodes]))

        # 1. In-Memory Data Structure (For Week 3 & Week 5 pipeline consumption)
        anchors_dict = {
            "entity_anchors": entity_anchors,
            "relation_anchors": relation_anchors
        }

        # 2. Detailed Logging Structure (For thesis evaluation & disk dump)
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
        """Saves or appends evaluation query logs to a JSON file on disk."""
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        # Load existing log data if file exists
        if os.path.exists(output_filepath):
            try:
                with open(output_filepath, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = {"metadata": {}, "logs": []}
        else:
            log_data = {"metadata": {}, "logs": []}

        # Update metadata and append log entry
        log_data["metadata"] = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_queries_logged": len(log_data["logs"]) + 1
        }
        log_data["logs"].append(detailed_log)

        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)


def process_query_input(query: str, save_log: bool = True, filepath: str = "data/parsed_queries.json") -> Dict[str, List[str]]:
    """
    High-level function to parse a query, optionally dump logs to disk,
    and return the clean nested dictionary for in-memory pipeline execution.
    """
    parser = StandaloneQueryParser()
    result = parser.parse_query(query)
    
    if save_log:
        parser.save_log_to_disk(result["detailed_log"], output_filepath=filepath)
        
    return result["anchors"]