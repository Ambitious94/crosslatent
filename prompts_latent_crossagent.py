import json

from prompts_crossagent import (
    CONLL04_ENTITY_TYPES,
    CONLL04_RELATION_TYPES,
    ENTITY_DEFINITIONS,
    RELATION_DEFINITIONS,
)


def _json_block(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def build_conll04_latent_ner_type_prompt(entity_type: str, sentence: str):
    definition = ENTITY_DEFINITIONS[entity_type]
    user_prompt = f"""You are the {entity_type} latent entity type agent for CoNLL04.

Entity type definition:
{entity_type}: {definition}

Sentence:
{sentence}

Read the sentence and internally identify evidence for entities of type {entity_type}.
Do not produce a text answer. Pass concise latent evidence to the following debate agent."""
    return [
        {"role": "system", "content": "You are a CoNLL04 latent NER type agent. Think silently in latent space."},
        {"role": "user", "content": user_prompt},
    ]


def build_conll04_text_anchor_prompt(kind: str, type_name: str, sentence: str, candidates):
    user_prompt = f"""You are reading explicit text candidates from the independent {type_name} {kind} agent.

Sentence:
{sentence}

Candidate JSON:
{_json_block(candidates)}

Treat these candidates as optional, possibly noisy evidence rather than final truth.
Encode the candidate entity names, relation triples, labels, and confidence cues for the latent debate reader.
Do not produce a text answer."""
    return [
        {"role": "system", "content": "You are a CoNLL04 text-cache anchor encoder. Think silently in cache space."},
        {"role": "user", "content": user_prompt},
    ]


def build_conll04_latent_ner_debate_prompt(sentence: str):
    user_prompt = f"""You are the CoNLL04 NER debate agent.

You have latent evidence from PER, LOC, ORG, and OTHER type agents.

Sentence:
{sentence}

Entity type definitions:
{_json_block(ENTITY_DEFINITIONS)}

Resolve type conflicts and output final deduplicated entities.
Return JSON only. If no entity exists, return {{"entities": []}}.

Output schema:
{{"entities": [{{"text": "entity text", "type": "PER"}}]}}
/no_think"""
    return [
        {"role": "system", "content": "You are a CoNLL04 NER debate agent. Output valid JSON only."},
        {"role": "user", "content": user_prompt},
    ]


def build_conll04_latent_ner_read_prompt(entity_type: str, sentence: str):
    user_prompt = f"""You are the CoNLL04 NER debate agent.

You are reading latent evidence from the independent {entity_type} entity type agent.

Sentence:
{sentence}

Entity type definitions:
{_json_block(ENTITY_DEFINITIONS)}

Internally absorb useful {entity_type} evidence for the final NER decision.
Do not produce a text answer. Pass updated debate evidence forward in latent space."""
    return [
        {"role": "system", "content": "You are a CoNLL04 latent NER debate reader. Think silently in latent space."},
        {"role": "user", "content": user_prompt},
    ]


def build_conll04_latent_ner_decode_prompt(sentence: str):
    user_prompt = f"""Output the final CoNLL04 NER JSON.

You have latent debate evidence gathered from independent PER, LOC, ORG, and OTHER type agents.

Sentence:
{sentence}

Entity type definitions:
{_json_block(ENTITY_DEFINITIONS)}

Resolve type conflicts and output final deduplicated entities.
Return JSON only. If no entity exists, return {{"entities": []}}.

Output schema:
{{"entities": [{{"text": "entity text", "type": "PER"}}]}}
/no_think"""
    return [
        {"role": "system", "content": "You are a CoNLL04 NER decoder. Output valid JSON only."},
        {"role": "user", "content": user_prompt},
    ]


def build_conll04_latent_re_type_prompt(relation_type: str, sentence: str, entities):
    spec = RELATION_DEFINITIONS[relation_type]
    user_prompt = f"""You are the {relation_type} latent relation type agent for CoNLL04.

Relation definition:
{relation_type}: {spec["definition"]}

Head type constraint: {spec["head"]}
Tail type constraint: {spec["tail"]}

Sentence:
{sentence}

NER debate result:
{_json_block(entities)}

Read the sentence and internally identify evidence for {relation_type} only.
Do not produce a text answer. Pass concise latent evidence to the following relation debate agent."""
    return [
        {"role": "system", "content": "You are a CoNLL04 latent RE type agent. Think silently in latent space."},
        {"role": "user", "content": user_prompt},
    ]


def build_conll04_latent_re_read_prompt(relation_type: str, sentence: str, entities):
    user_prompt = f"""You are the CoNLL04 relation debate agent.

You are reading latent evidence from the independent {relation_type} relation type agent.

Sentence:
{sentence}

NER debate result:
{_json_block(entities)}

Relation definitions and schema:
{_json_block(RELATION_DEFINITIONS)}

Internally absorb useful {relation_type} evidence for the final RE decision.
Do not produce a text answer. Pass updated debate evidence forward in latent space."""
    return [
        {"role": "system", "content": "You are a CoNLL04 latent RE debate reader. Think silently in latent space."},
        {"role": "user", "content": user_prompt},
    ]


def build_conll04_latent_re_decode_prompt(sentence: str, entities):
    user_prompt = f"""Output the final CoNLL04 relation JSON.

You have latent debate evidence gathered from independent relation type agents.

Sentence:
{sentence}

NER debate result:
{_json_block(entities)}

Relation definitions and schema:
{_json_block(RELATION_DEFINITIONS)}

Resolve same-pair relation conflicts and output final deduplicated relations.
Return JSON only. If no relation exists, return {{"relations": []}}.

Output schema:
{{"relations": [{{"head": "head entity text", "relation": "Work_For", "tail": "tail entity text"}}]}}
/no_think"""
    return [
        {"role": "system", "content": "You are a CoNLL04 relation decoder. Output valid JSON only."},
        {"role": "user", "content": user_prompt},
    ]


def build_conll04_latent_re_debate_prompt(sentence: str, entities):
    user_prompt = f"""You are the CoNLL04 relation debate agent.

You have latent evidence from Work_For, Located_In, OrgBased_In, Live_In, and Kill relation agents.

Sentence:
{sentence}

NER debate result:
{_json_block(entities)}

Relation definitions and schema:
{_json_block(RELATION_DEFINITIONS)}

Resolve same-pair relation conflicts and output final deduplicated relations.
Return JSON only. If no relation exists, return {{"relations": []}}.

Output schema:
{{"relations": [{{"head": "head entity text", "relation": "Work_For", "tail": "tail entity text"}}]}}
/no_think"""
    return [
        {"role": "system", "content": "You are a CoNLL04 relation debate agent. Output valid JSON only."},
        {"role": "user", "content": user_prompt},
    ]


def build_conll04_latent_cross_task_seed_prompt(sentence: str, entities, relations):
    user_prompt = f"""You are the CoNLL04 latent cross-task verifier.

Sentence:
{sentence}

NER debate result:
{_json_block(entities)}

RE debate result:
{_json_block(relations)}

Relation schema:
{_json_block(RELATION_DEFINITIONS)}

Internally compare NER and RE results. Encode latent evidence for repairs:
- repair entity types only when relation evidence is strong
- drop incompatible relations
- keep only the CoNLL04 entity and relation schema

Do not produce a text answer. Pass latent verification evidence to the final JSON decoder."""
    return [
        {"role": "system", "content": "You are a CoNLL04 latent cross-task verifier. Think silently in latent space."},
        {"role": "user", "content": user_prompt},
    ]


def build_conll04_latent_cross_task_decode_prompt(sentence: str, entities, relations):
    user_prompt = f"""Output the final CoNLL04 extraction JSON.

You have latent verification evidence from the previous cross-task verifier.

Sentence:
{sentence}

NER debate result:
{_json_block(entities)}

RE debate result:
{_json_block(relations)}

Relation schema:
{_json_block(RELATION_DEFINITIONS)}

Return final JSON only. Do not include explanations or <think> tags.

Output schema:
{{
  "entities": [{{"text": "entity text", "type": "PER"}}],
  "relations": [{{"head": "head entity text", "relation": "Work_For", "tail": "tail entity text"}}]
}}
/no_think"""
    return [
        {"role": "system", "content": "You are an expert CoNLL04 extraction decoder. Output valid JSON only."},
        {"role": "user", "content": user_prompt},
    ]
