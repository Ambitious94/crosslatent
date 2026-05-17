import json

from prompts_crossagent import (
    CONLL04_ENTITY_TYPES,
    CONLL04_RELATION_TYPES,
    ENTITY_DEFINITIONS,
    RELATION_DEFINITIONS,
)


CHEMPROT_RELATION_TYPES = ["UPREGULATOR", "DOWNREGULATOR", "AGONIST", "ANTAGONIST", "SUBSTRATE"]

CHEMPROT_RELATION_DEFINITIONS = {
    "UPREGULATOR": "A chemical increases, activates, induces, stimulates, or up-regulates a gene/protein.",
    "DOWNREGULATOR": "A chemical decreases, inhibits, suppresses, blocks, or down-regulates a gene/protein.",
    "AGONIST": "A chemical acts as an agonist of a receptor or protein target.",
    "ANTAGONIST": "A chemical acts as an antagonist or blocker of a receptor or protein target.",
    "SUBSTRATE": "A chemical is a substrate of an enzyme or protein.",
}


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


def build_conll04_latent_re_c2c_decode_prompt(sentence: str, entities):
    user_prompt = f"""Output the final CoNLL04 relation JSON.

You have cache evidence from:
1. selected latent relation type agents
2. a text RE debate agent near the end of the cache

Sentence:
{sentence}

NER debate result:
{_json_block(entities)}

Relation definitions and schema:
{_json_block(RELATION_DEFINITIONS)}

Prioritize schema-compatible relation triples encoded by the text RE debate cache.
Use latent relation type cache to refine or reject contradictions.
Return JSON only. If no relation exists, return {{"relations": []}}.

Output schema:
{{"relations": [{{"head": "head entity text", "relation": "Work_For", "tail": "tail entity text"}}]}}
/no_think"""
    return [
        {"role": "system", "content": "You are a CoNLL04 C2C relation decoder. Output valid JSON only."},
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


def build_chemprot_text_re_type_prompt(relation_type: str, text: str, entity_list: str):
    user_prompt = f"""You are the {relation_type} relation type agent for ChemProt.

Relation definition:
{relation_type}: {CHEMPROT_RELATION_DEFINITIONS[relation_type]}

Annotated entities (ONLY use these, copy text exactly):
{entity_list}

Document:
{text}

Extract only {relation_type} relations.
Rules:
- head MUST be a CHEMICAL entity from the list.
- tail MUST be a gene/protein entity from the list, preferably GENE-Y rather than GENE-N.
- The document must explicitly describe this chemical-protein interaction.
- Do NOT infer a relation from mere co-occurrence, assay context, comparison, speculation, or background knowledge.
- If evidence is weak, ambiguous, indirect, or could fit multiple relation labels, output {{"relations": []}}.
- Use confidence >= 0.80 only for relations with strong explicit textual support; otherwise omit the relation.
- Include confidence from 0.0 to 1.0.
- Return JSON only. If no relation exists, return {{"relations": []}}.

Output schema:
{{"relations": [{{"head": "chemical_name", "relation": "{relation_type}", "tail": "protein_name", "confidence": 0.0}}]}}
/no_think"""
    return [
        {"role": "system", "content": "You are a precise ChemProt relation extraction type agent. Output valid JSON only."},
        {"role": "user", "content": user_prompt},
    ]


def build_chemprot_text_re_debate_prompt(text: str, entity_list: str, candidates):
    user_prompt = f"""Resolve ChemProt relation type conflicts.

Valid relations:
{_json_block(CHEMPROT_RELATION_DEFINITIONS)}

Annotated entities:
{entity_list}

Document:
{text}

Candidate relations from relation type agents:
{_json_block(candidates)}

Use the document, entity list, relation definitions, and confidence scores.
Prioritize precision over recall.
Keep only candidates with strong explicit textual support and confidence >= 0.80.
Remove unsupported, weakly implied, co-occurrence-only, duplicated, wrong-direction, wrong-schema, or ambiguous-label relations.
When candidates conflict or evidence is borderline, reject the relation instead of guessing.
Return final deduplicated relations as JSON only.

Output schema:
{{"relations": [{{"head": "chemical_name", "relation": "UPREGULATOR", "tail": "protein_name"}}]}}
/no_think"""
    return [
        {"role": "system", "content": "You are a ChemProt relation debate agent. Output valid JSON only."},
        {"role": "user", "content": user_prompt},
    ]


def build_chemprot_latent_re_type_prompt(relation_type: str, text: str, entity_list: str):
    user_prompt = f"""You are the {relation_type} latent relation type agent for ChemProt.

Relation definition:
{relation_type}: {CHEMPROT_RELATION_DEFINITIONS[relation_type]}

Annotated entities:
{entity_list}

Document:
{text}

Read the document and internally identify only strong explicit evidence for {relation_type}.
Ignore co-occurrence-only, weak, indirect, speculative, or ambiguous evidence.
Do not produce a text answer. Pass concise latent evidence to the following ChemProt relation debate agent."""
    return [
        {"role": "system", "content": "You are a ChemProt latent RE type agent. Think silently in latent space."},
        {"role": "user", "content": user_prompt},
    ]


def build_chemprot_latent_re_read_prompt(relation_type: str, text: str, entity_list: str):
    user_prompt = f"""You are the ChemProt latent relation debate agent.

You are reading latent evidence from the independent {relation_type} relation type agent.

Valid relations:
{_json_block(CHEMPROT_RELATION_DEFINITIONS)}

Annotated entities:
{entity_list}

Document:
{text}

Internally absorb only strong, explicit {relation_type} evidence for the final ChemProt decision.
Reject weak, co-occurrence-only, wrong-direction, or ambiguous evidence.
Do not produce a text answer. Pass updated debate evidence forward in latent space."""
    return [
        {"role": "system", "content": "You are a ChemProt latent RE debate reader. Think silently in latent space."},
        {"role": "user", "content": user_prompt},
    ]


def build_chemprot_latent_re_decode_prompt(text: str, entity_list: str):
    user_prompt = f"""Output the final ChemProt relation JSON.

You have latent debate evidence gathered from independent ChemProt relation type agents.

Valid relations:
{_json_block(CHEMPROT_RELATION_DEFINITIONS)}

Annotated entities:
{entity_list}

Document:
{text}

Rules:
- Prioritize precision over recall.
- Keep only strong explicit chemical-protein interactions.
- Do not output co-occurrence-only, weakly implied, wrong-direction, or ambiguous-label relations.
- If uncertain, output {{"relations": []}}.

Return JSON only. If no relation exists, return {{"relations": []}}.

Output schema:
{{"relations": [{{"head": "chemical_name", "relation": "UPREGULATOR", "tail": "protein_name"}}]}}
/no_think"""
    return [
        {"role": "system", "content": "You are a ChemProt relation decoder. Output valid JSON only."},
        {"role": "user", "content": user_prompt},
    ]


def build_chemprot_latent_re_c2c_decode_prompt(text: str, entity_list: str):
    user_prompt = f"""Output the final ChemProt relation JSON.

You have cache evidence from:
1. selected latent ChemProt relation type agents
2. a text ChemProt relation debate agent near the end of the cache

Valid relations:
{_json_block(CHEMPROT_RELATION_DEFINITIONS)}

Annotated entities:
{entity_list}

Document:
{text}

Prioritize precision over recall.
Use schema-compatible relation triples encoded by the text debate cache only when they have strong explicit support.
Use latent relation type cache to refine or reject contradictions.
Reject co-occurrence-only, weakly implied, wrong-direction, or ambiguous-label relations.
Return JSON only. If no relation exists, return {{"relations": []}}.

Output schema:
{{"relations": [{{"head": "chemical_name", "relation": "UPREGULATOR", "tail": "protein_name"}}]}}
/no_think"""
    return [
        {"role": "system", "content": "You are a ChemProt C2C relation decoder. Output valid JSON only."},
        {"role": "user", "content": user_prompt},
    ]


def build_chemprot_latent_verifier_seed_prompt(text: str, entity_list: str, relations):
    user_prompt = f"""You are the ChemProt latent relation verifier.

Valid relations:
{_json_block(CHEMPROT_RELATION_DEFINITIONS)}

Annotated entities:
{entity_list}

Document:
{text}

RE debate result:
{_json_block(relations)}

Internally verify support, direction, and schema.
Prefer rejecting weak or ambiguous relations over keeping false positives.
Do not produce text."""
    return [
        {"role": "system", "content": "You are a ChemProt latent verifier. Think silently in latent space."},
        {"role": "user", "content": user_prompt},
    ]


def build_chemprot_latent_final_decode_prompt(text: str, entity_list: str, relations):
    user_prompt = f"""Output the final ChemProt extraction JSON.

You have latent verification evidence from the previous verifier.

Valid relations:
{_json_block(CHEMPROT_RELATION_DEFINITIONS)}

Annotated entities:
{entity_list}

Document:
{text}

RE debate result:
{_json_block(relations)}

Rules:
- Keep only relations with strong explicit textual support.
- Remove co-occurrence-only, weakly implied, wrong-direction, wrong-schema, duplicated, or ambiguous-label relations.
- If uncertain, output {{"relations": []}}.

Return final JSON only. Do not include explanations or <think> tags.

Output schema:
{{"relations": [{{"head": "chemical_name", "relation": "UPREGULATOR", "tail": "protein_name"}}]}}
/no_think"""
    return [
        {"role": "system", "content": "You are an expert ChemProt extraction decoder. Output valid JSON only."},
        {"role": "user", "content": user_prompt},
    ]
