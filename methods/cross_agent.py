from typing import Dict, List, Optional
import json
import re

from models import ModelWrapper
from utils import evaluate_prediction
from prompts_crossagent import (
    CONLL04_ENTITY_TYPES,
    CONLL04_RELATION_TYPES,
    build_conll04_cross_task_prompt,
    build_conll04_ner_debate_prompt,
    build_conll04_ner_type_prompt,
    build_conll04_re_debate_prompt,
    build_conll04_re_type_prompt,
)
from prompts_latent_crossagent import (
    CHEMPROT_RELATION_TYPES,
    build_chemprot_latent_final_decode_prompt,
    build_chemprot_text_re_debate_prompt,
    build_chemprot_text_re_type_prompt,
)


def _extract_json(text: str) -> Dict:
    stripped = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"\{.*\}",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, stripped, flags=re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except Exception:
                continue
    try:
        return json.loads(stripped)
    except Exception:
        return {}


def _json_items(data, key: str) -> List[Dict]:
    if isinstance(data, dict):
        value = data.get(key, [])
        return value if isinstance(value, list) else []
    if isinstance(data, list):
        return data
    return []


def _norm_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _norm_entity_type(value) -> str:
    raw = str(value or "").strip().upper().replace("PERSON", "PER").replace("LOCATION", "LOC").replace("ORGANIZATION", "ORG")
    if raw in {"PEOP", "PEOPLE"}:
        raw = "PER"
    return raw if raw in set(CONLL04_ENTITY_TYPES) else "OTHER"


def _norm_relation(value) -> str:
    aliases = {
        "WORK_FOR": "Work_For",
        "WORK-FOR": "Work_For",
        "WORK FOR": "Work_For",
        "LOCATED_IN": "Located_In",
        "LOCATED-IN": "Located_In",
        "LOCATED IN": "Located_In",
        "ORGBASED_IN": "OrgBased_In",
        "ORGBASED-IN": "OrgBased_In",
        "ORGBASED IN": "OrgBased_In",
        "ORG_BASED_IN": "OrgBased_In",
        "ORG-BASED-IN": "OrgBased_In",
        "LIVE_IN": "Live_In",
        "LIVE-IN": "Live_In",
        "LIVE IN": "Live_In",
        "KILL": "Kill",
        "KILLED": "Kill",
    }
    raw = str(value or "").strip()
    if raw in CONLL04_RELATION_TYPES:
        return raw
    return aliases.get(raw.upper(), raw)


def _clean_entities(entities: List[Dict], keep_confidence: bool = False) -> List[Dict]:
    cleaned = []
    seen = set()
    for ent in entities or []:
        if not isinstance(ent, dict):
            continue
        text = _norm_text(ent.get("text") or ent.get("entity") or ent.get("name"))
        if not text:
            continue
        etype = _norm_entity_type(ent.get("type") or ent.get("label"))
        key = (text.lower(), etype)
        if key in seen:
            continue
        seen.add(key)
        item = {"text": text, "type": etype}
        if keep_confidence and "confidence" in ent:
            try:
                item["confidence"] = float(ent.get("confidence"))
            except Exception:
                pass
        cleaned.append(item)
    return cleaned


def _clean_relations(relations: List[Dict], keep_confidence: bool = False) -> List[Dict]:
    cleaned = []
    seen = set()
    for rel in relations or []:
        if not isinstance(rel, dict):
            continue
        head = _norm_text(rel.get("head") or rel.get("subject") or rel.get("h"))
        tail = _norm_text(rel.get("tail") or rel.get("object") or rel.get("t"))
        relation = _norm_relation(rel.get("relation") or rel.get("type") or rel.get("label"))
        if not head or not tail or relation not in CONLL04_RELATION_TYPES:
            continue
        key = (head.lower(), relation, tail.lower())
        if key in seen:
            continue
        seen.add(key)
        item = {"head": head, "relation": relation, "tail": tail}
        if keep_confidence and "confidence" in rel:
            try:
                item["confidence"] = float(rel.get("confidence"))
            except Exception:
                pass
        cleaned.append(item)
    return cleaned


def _clean_chemprot_relations(relations: List[Dict], entities_meta=None, keep_confidence: bool = False) -> List[Dict]:
    valid = set(CHEMPROT_RELATION_TYPES)
    canonical = {
        str(e.get("text", "")).strip().lower(): str(e.get("text", "")).strip()
        for e in (entities_meta or [])
        if isinstance(e, dict) and str(e.get("text", "")).strip()
    }
    cleaned = []
    seen = set()
    for rel in relations or []:
        if not isinstance(rel, dict):
            continue
        head = _norm_text(rel.get("head") or rel.get("subject") or rel.get("h"))
        tail = _norm_text(rel.get("tail") or rel.get("object") or rel.get("t"))
        relation = str(rel.get("relation") or rel.get("type") or rel.get("label") or "").strip().upper()
        if keep_confidence and "confidence" in rel:
            try:
                if float(rel.get("confidence")) < 0.8:
                    continue
            except Exception:
                continue
        if relation not in valid or not head or not tail:
            continue
        head = canonical.get(head.lower(), head)
        tail = canonical.get(tail.lower(), tail)
        key = (head.lower(), relation, tail.lower())
        if key in seen:
            continue
        seen.add(key)
        item = {"head": head, "relation": relation, "tail": tail}
        if keep_confidence and "confidence" in rel:
            try:
                item["confidence"] = float(rel.get("confidence"))
            except Exception:
                pass
        cleaned.append(item)
    return cleaned


class CrossAgentMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        generate_bs: int = 1,
        use_vllm: bool = False,
        args=None,
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.use_vllm = use_vllm
        self.args = args
        self.task = args.task
        self.method_name = "cross_agent"
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _encode_output_tokens(self, text: str):
        encoded = self.model.tokenizer(text, add_special_tokens=False, return_tensors=None)
        token_ids = encoded["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return token_ids, self.model.tokenizer.convert_ids_to_tokens(token_ids)

    def _generate_one(self, messages: List[Dict], name: str, role: str) -> tuple[str, Dict]:
        prompts, input_ids, attention_mask, tokens_batch, _ = self.model.prepare_chat_batch(
            [messages], add_generation_prompt=True
        )
        if self.model.use_vllm:
            generated = self.model.vllm_generate_text_batch(
                prompts,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=1.1,
            )[0]
        else:
            generated_batch, _ = self.model.generate_text_batch(
                input_ids,
                attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=1.1,
            )
            generated = generated_batch[0]
        mask = attention_mask[0].bool()
        trimmed_ids = input_ids[0][mask].to("cpu").tolist()
        _, output_tokens = self._encode_output_tokens(generated)
        self.total_input_tokens += len(tokens_batch[0])
        self.total_output_tokens += len(output_tokens)
        trace = {
            "name": name,
            "role": role,
            "input": prompts[0],
            "input_ids": trimmed_ids,
            "input_tokens": tokens_batch[0],
            "output": generated,
        }
        return generated, trace

    def _run_item_conll04(self, item: Dict) -> Dict:
        sentence = item["question"]
        traces = []

        ner_candidates = []
        for etype in CONLL04_ENTITY_TYPES:
            text, trace = self._generate_one(
                build_conll04_ner_type_prompt(etype, sentence),
                name=f"{etype}_Agent",
                role="ner_type_agent",
            )
            traces.append(trace)
            ner_candidates.extend(_clean_entities(_json_items(_extract_json(text), "entities"), keep_confidence=True))

        text, trace = self._generate_one(
            build_conll04_ner_debate_prompt(sentence, ner_candidates),
            name="NER_Debate_Agent",
            role="ner_debate",
        )
        traces.append(trace)
        entities = _clean_entities(_json_items(_extract_json(text), "entities"))
        if not entities:
            entities = _clean_entities(ner_candidates)

        re_candidates = []
        for rtype in CONLL04_RELATION_TYPES:
            text, trace = self._generate_one(
                build_conll04_re_type_prompt(rtype, sentence, entities),
                name=f"{rtype}_Agent",
                role="re_type_agent",
            )
            traces.append(trace)
            re_candidates.extend(_clean_relations(_json_items(_extract_json(text), "relations"), keep_confidence=True))

        text, trace = self._generate_one(
            build_conll04_re_debate_prompt(sentence, re_candidates),
            name="RE_Debate_Agent",
            role="re_debate",
        )
        traces.append(trace)
        relations = _clean_relations(_json_items(_extract_json(text), "relations"))
        if not relations:
            relations = _clean_relations(re_candidates)

        text, trace = self._generate_one(
            build_conll04_cross_task_prompt(sentence, entities, relations),
            name="Cross_Task_Verifier",
            role="cross_task_verifier",
        )
        traces.append(trace)
        final_data = _extract_json(text)
        final_entities = _clean_entities(_json_items(final_data, "entities"))
        final_relations = _clean_relations(_json_items(final_data, "relations"))
        if not final_entities:
            final_entities = entities
        if not final_relations:
            final_relations = relations

        cleaned_text = json.dumps(
            {"entities": final_entities, "relations": final_relations},
            ensure_ascii=False,
        )
        eval_result = evaluate_prediction(self.task, cleaned_text, item, 0)
        return {
            "question": sentence,
            "gold": eval_result["gold"],
            "solution": item["solution"],
            "prediction": eval_result["prediction"],
            "raw_prediction": text,
            "agents": traces,
            "correct": eval_result["correct"],
        }

    def _run_item_chemprot(self, item: Dict) -> Dict:
        text = item["question"]
        entity_list = item.get("entity_list", "")
        entities_meta = item.get("entities_meta", [])
        traces = []

        re_candidates = []
        for relation_type in CHEMPROT_RELATION_TYPES:
            output, trace = self._generate_one(
                build_chemprot_text_re_type_prompt(relation_type, text, entity_list),
                name=f"{relation_type}_Agent",
                role="re_type_agent",
            )
            traces.append(trace)
            re_candidates.extend(
                _clean_chemprot_relations(
                    _json_items(_extract_json(output), "relations"),
                    entities_meta=entities_meta,
                    keep_confidence=True,
                )
            )

        output, trace = self._generate_one(
            build_chemprot_text_re_debate_prompt(text, entity_list, re_candidates),
            name="ChemProt_RE_Debate_Agent",
            role="re_debate",
        )
        traces.append(trace)
        relations = _clean_chemprot_relations(
            _json_items(_extract_json(output), "relations"),
            entities_meta=entities_meta,
        )
        if not relations:
            relations = _clean_chemprot_relations(re_candidates, entities_meta=entities_meta)

        output, trace = self._generate_one(
            build_chemprot_latent_final_decode_prompt(text, entity_list, relations),
            name="ChemProt_Final_Verifier",
            role="relation_verifier",
        )
        traces.append(trace)
        final_relations = _clean_chemprot_relations(
            _json_items(_extract_json(output), "relations"),
            entities_meta=entities_meta,
        )
        if not final_relations:
            final_relations = relations

        cleaned_text = json.dumps({"relations": final_relations}, ensure_ascii=False)
        eval_result = evaluate_prediction(self.task, cleaned_text, item, 0)
        return {
            "question": text,
            "gold": eval_result["gold"],
            "solution": item["solution"],
            "prediction": eval_result["prediction"],
            "raw_prediction": output,
            "agents": traces,
            "correct": eval_result["correct"],
            "entities_meta": entities_meta,
        }

    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")
        if self.task not in {"conll04", "chemprot"}:
            raise ValueError("cross_agent currently supports only --task conll04 or chemprot")
        if self.task == "chemprot":
            return [self._run_item_chemprot(item) for item in items]
        return [self._run_item_conll04(item) for item in items]

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
