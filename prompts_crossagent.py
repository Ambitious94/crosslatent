import json


CONLL04_ENTITY_TYPES = ["PER", "LOC", "ORG", "OTHER"]
CONLL04_RELATION_TYPES = ["Work_For", "Located_In", "OrgBased_In", "Live_In", "Kill"]

ENTITY_DEFINITIONS = {
    "PER": "A person, named individual, family member, or group of named people.",
    "LOC": "A physical or geopolitical place, including cities, countries, regions, facilities, and locations.",
    "ORG": "An organization, company, agency, institution, government body, team, or named group.",
    "OTHER": "A named entity that is relevant but is not clearly a person, location, or organization.",
}

RELATION_DEFINITIONS = {
    "Work_For": {
        "definition": "A person works for, is employed by, leads, represents, or is affiliated with an organization.",
        "head": "PER",
        "tail": "ORG",
    },
    "Located_In": {
        "definition": "An entity or place is geographically located in another location.",
        "head": "LOC/ORG",
        "tail": "LOC",
    },
    "OrgBased_In": {
        "definition": "An organization is based in, headquartered in, or primarily located in a location.",
        "head": "ORG",
        "tail": "LOC",
    },
    "Live_In": {
        "definition": "A person lives in, resides in, or is from a location.",
        "head": "PER",
        "tail": "LOC",
    },
    "Kill": {
        "definition": "A person or organization kills, murders, assassinates, or causes the death of a person.",
        "head": "PER/ORG",
        "tail": "PER",
    },
}


def _json_block(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def build_conll04_ner_type_prompt(entity_type: str, sentence: str):
    definition = ENTITY_DEFINITIONS[entity_type]
    user_prompt = f"""You are the {entity_type} entity type agent for CoNLL04 information extraction.

Entity type definition:
{entity_type}: {definition}

Sentence:
{sentence}

Extract only entities of type {entity_type}. Include a confidence score from 0.0 to 1.0.
Return JSON only. If no entity exists, return {{"entities": []}}.

Output schema:
{{"entities": [{{"text": "entity text", "type": "{entity_type}", "confidence": 0.0}}]}}
/no_think"""
    return [
        {"role": "system", "content": "You are a precise CoNLL04 entity extraction type agent. Output valid JSON only."},
        {"role": "user", "content": user_prompt},
    ]


def build_conll04_re_type_prompt(relation_type: str, sentence: str, entities):
    spec = RELATION_DEFINITIONS[relation_type]
    user_prompt = f"""You are the {relation_type} relation type agent for CoNLL04 relation extraction.

Relation definition:
{relation_type}: {spec["definition"]}

Head type constraint: {spec["head"]}
Tail type constraint: {spec["tail"]}

Sentence:
{sentence}

Candidate entities:
{_json_block(entities)}

Extract only {relation_type} relations. Include a confidence score from 0.0 to 1.0.
Return JSON only. If no relation exists, return {{"relations": []}}.

Output schema:
{{"relations": [{{"head": "head entity text", "relation": "{relation_type}", "tail": "tail entity text", "confidence": 0.0}}]}}
/no_think"""
    return [
        {"role": "system", "content": "You are a precise CoNLL04 relation extraction type agent. Output valid JSON only."},
        {"role": "user", "content": user_prompt},
    ]


def build_conll04_ner_debate_prompt(sentence: str, candidate_entities):
    user_prompt = f"""Resolve CoNLL04 NER type conflicts.

Sentence:
{sentence}

Entity type definitions:
{_json_block(ENTITY_DEFINITIONS)}

Candidate entities from type agents:
{_json_block(candidate_entities)}

Use the sentence context, type definitions, and confidence scores.
For duplicate or overlapping spans, choose one best type or reject the span.
Return final deduplicated entities as JSON only.

Output schema:
{{"entities": [{{"text": "entity text", "type": "PER"}}]}}
/no_think"""
    return [
        {"role": "system", "content": "You are a CoNLL04 NER debate agent. Output valid JSON only."},
        {"role": "user", "content": user_prompt},
    ]


def build_conll04_re_debate_prompt(sentence: str, candidate_relations):
    user_prompt = f"""Resolve CoNLL04 relation type conflicts.

Sentence:
{sentence}

Relation definitions and schema:
{_json_block(RELATION_DEFINITIONS)}

Candidate relations from relation type agents:
{_json_block(candidate_relations)}

Use the sentence context, relation definitions, and confidence scores.
For the same head-tail pair with conflicting relation types, choose the best relation or reject the pair.
Return final deduplicated relations as JSON only.

Output schema:
{{"relations": [{{"head": "head entity text", "relation": "Work_For", "tail": "tail entity text"}}]}}
/no_think"""
    return [
        {"role": "system", "content": "You are a CoNLL04 relation debate agent. Output valid JSON only."},
        {"role": "user", "content": user_prompt},
    ]


def build_conll04_cross_task_prompt(sentence: str, entities, relations):
    user_prompt = f"""Perform CoNLL04 cross-task verification between NER and RE.

Sentence:
{sentence}

Relation schema:
{_json_block(RELATION_DEFINITIONS)}

NER result:
{_json_block(entities)}

RE result:
{_json_block(relations)}

Revise entities and relations so that entity types and relation schemas agree.
You may repair an entity type when a relation strongly supports it, or drop a relation when it is incompatible.
Return final JSON only.

Output schema:
{{
  "entities": [{{"text": "entity text", "type": "PER"}}],
  "relations": [{{"head": "head entity text", "relation": "Work_For", "tail": "tail entity text"}}]
}}
/no_think"""
    return [
        {"role": "system", "content": "You are a CoNLL04 cross-task verifier. Output valid JSON only."},
        {"role": "user", "content": user_prompt},
    ]
