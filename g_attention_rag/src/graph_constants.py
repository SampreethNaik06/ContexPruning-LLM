"""Shared vocabulary and schema settings for graph construction."""


SCHEMA_VERSION = "3.3"
SUBJECT_DEPS = {"nsubj", "csubj", "nsubjpass", "csubjpass"}
DIRECT_OBJECT_DEPS = {"dobj", "obj", "attr", "oprd", "dative"}
DEICTIC_WORDS = {"this", "that", "these", "those"}
PRONOUN_WORDS = {"it", "they", "he", "she", "we", "which", "who", "whom"}
GENERIC_EXACT = {
    "act", "another", "branch", "conclusion", "core", "other", "one",
    "something", "state", "thing", "way", "which", "world",
}
GENERIC_HEADS = {"act", "conclusion", "core", "other", "one", "thing", "way", "which"}
ALLOWED_PREPOSITIONS = {
    "at", "between", "beyond", "by", "during", "for", "from", "in", "into",
    "of", "on", "through", "to", "until", "upon", "with",
}
DESCRIPTOR_HEADS = {"application", "cornerstone", "feature", "field", "form", "type"}
CO_OCCUR_RELATION = "co_occurs"
CONTEXT_RELATIONS = {"covers", "contextualizes", "next_topic"}
