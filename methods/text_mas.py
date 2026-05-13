from typing import Dict, List, Optional

from . import default_agents, verifier_agent
from models import ModelWrapper
# from prompts import build_agent_messages, build_agent_messages_v6, build_agent_messages_v6_text_mas
from prompts import build_agent_messages_hierarchical_text_mas, build_agent_messages_sequential_text_mas
from utils import evaluate_prediction
import argparse
import json
import re

_VERIFIER_TEMPERATURE = 0.0
_VERIFIER_TOP_P = 1.0
_CHEMPROT_RELATIONS = {"UPREGULATOR", "DOWNREGULATOR", "AGONIST", "ANTAGONIST", "SUBSTRATE"}


def _extract_valid_json_or_none(text: str) -> Optional[str]:
    stripped = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    start_idx = stripped.find("{")
    end_idx = stripped.rfind("}")
    if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
        return None
    candidate = stripped[start_idx:end_idx + 1]
    try:
        json.loads(candidate)
        return candidate
    except Exception:
        return None


def _stabilize_funsd_verifier_json(judger_text: str, verifier_text: str) -> str:
    judger_json = _extract_valid_json_or_none(judger_text)
    verifier_json = _extract_valid_json_or_none(verifier_text)
    if judger_json is None or verifier_json is None:
        return judger_text

    try:
        judger_data = json.loads(judger_json)
        verifier_data = json.loads(verifier_json)
    except Exception:
        return judger_text

    valid_labels = {"question", "answer", "header", "other"}
    judger_entities = judger_data.get("entities", [])
    verifier_entities = verifier_data.get("entities", [])

    if not isinstance(judger_entities, list) or not isinstance(verifier_entities, list):
        return judger_json
    if len(judger_entities) != len(verifier_entities):
        return judger_json

    stabilized_entities = []
    canonical_ids = []
    same_layout = True
    for j_ent, v_ent in zip(judger_entities, verifier_entities):
        if not isinstance(j_ent, dict) or not isinstance(v_ent, dict):
            return judger_json
        if j_ent.get("id") != v_ent.get("id") or j_ent.get("text") != v_ent.get("text"):
            same_layout = False
            break
        label = v_ent.get("label") if v_ent.get("label") in valid_labels else j_ent.get("label")
        if label not in valid_labels:
            label = "other"
        stabilized_entities.append(
            {
                "id": j_ent.get("id"),
                "text": j_ent.get("text", ""),
                "label": label,
            }
        )
        canonical_ids.append(j_ent.get("id"))

    if not same_layout:
        return judger_json

    valid_id_set = set(canonical_ids)

    def _valid_rel_set(relations):
        valid = []
        if not isinstance(relations, list):
            return valid
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            head = rel.get("head")
            tail = rel.get("tail")
            if head in valid_id_set and tail in valid_id_set and rel.get("type") == "linked":
                valid.append({"head": head, "tail": tail, "type": "linked"})
        return valid

    judger_relations = _valid_rel_set(judger_data.get("relations", []))
    verifier_relations = _valid_rel_set(verifier_data.get("relations", []))
    judger_rel_set = {(r["head"], r["tail"], r["type"]) for r in judger_relations}
    verifier_rel_set = {(r["head"], r["tail"], r["type"]) for r in verifier_relations}

    if verifier_rel_set.issubset(judger_rel_set):
        final_relations = verifier_relations
    else:
        final_relations = judger_relations

    return json.dumps(
        {"entities": stabilized_entities, "relations": final_relations},
        ensure_ascii=False,
    )


def _stabilize_cord_verifier_json(judger_text: str, verifier_text: str) -> str:
    judger_json = _extract_valid_json_or_none(judger_text)
    verifier_json = _extract_valid_json_or_none(verifier_text)
    if judger_json is None or verifier_json is None:
        return judger_text

    try:
        judger_data = json.loads(judger_json)
        verifier_data = json.loads(verifier_json)
    except Exception:
        return judger_text

    def _as_str_or_empty(value):
        return value if isinstance(value, str) else ""

    judger_menu = judger_data.get("menu", [])
    verifier_menu = verifier_data.get("menu", [])
    if not isinstance(judger_menu, list) or not isinstance(verifier_menu, list):
        return judger_json

    stabilized_menu = []
    if len(judger_menu) == len(verifier_menu):
        for j_item, v_item in zip(judger_menu, verifier_menu):
            if not isinstance(j_item, dict) or not isinstance(v_item, dict):
                return judger_json
            item = {}
            for key in ("nm", "cnt", "price"):
                j_val = _as_str_or_empty(j_item.get(key, ""))
                v_val = _as_str_or_empty(v_item.get(key, ""))
                item[key] = j_val if j_val != "" else v_val
            stabilized_menu.append(item)
    else:
        for j_item in judger_menu:
            if not isinstance(j_item, dict):
                return judger_json
            stabilized_menu.append(
                {key: _as_str_or_empty(j_item.get(key, "")) for key in ("nm", "cnt", "price")}
            )

    judger_total = judger_data.get("total", {})
    verifier_total = verifier_data.get("total", {})
    if not isinstance(judger_total, dict) or not isinstance(verifier_total, dict):
        return judger_json

    stabilized_total = {}
    for key in ("total_price", "cashprice", "changeprice", "subtotal_price", "tax_price"):
        j_val = _as_str_or_empty(judger_total.get(key, ""))
        v_val = _as_str_or_empty(verifier_total.get(key, ""))
        stabilized_total[key] = j_val if j_val != "" else v_val

    return json.dumps(
        {"menu": stabilized_menu, "total": stabilized_total},
        ensure_ascii=False,
    )


def _parse_chemprot_entities(entity_list_text: str):
    chemical_names = set()
    gene_y_names = set()
    gene_n_names = set()
    for raw_line in (entity_list_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^\[\d+\]\s+\(([^)]+)\)\s+(.+)$", line)
        if not match:
            continue
        label = match.group(1).strip()
        name = match.group(2).strip()
        if label == "CHEMICAL":
            chemical_names.add(name)
        elif label == "GENE-Y":
            gene_y_names.add(name)
        elif label == "GENE-N":
            gene_n_names.add(name)
    return chemical_names, gene_y_names, gene_n_names


def _sanitize_chemprot_context(text: str, entity_list_text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    chemical_names, gene_y_names, gene_n_names = _parse_chemprot_entities(entity_list_text)
    lines = []
    seen = set()
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-").strip()
        if not line or "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 2:
            continue
        head = parts[0]
        tail = parts[1]
        if chemical_names and head not in chemical_names:
            continue
        if gene_y_names and tail not in gene_y_names:
            continue
        if tail in gene_n_names:
            continue
        normalized = f"{head} | {tail}"
        if normalized in seen:
            continue
        seen.add(normalized)
        lines.append(normalized)
    return "\n".join(lines) if lines else "NONE"


def _sanitize_docred_context(text: str, entity_list_text: str) -> str:
    from prompts import DOCRED_REL_MAP

    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    valid_names = set(DOCRED_REL_MAP.values())
    valid_entity_ids = _parse_docred_entity_ids(entity_list_text)
    lines = []
    seen = set()
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-").strip()
        if not line or "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 3:
            continue
        head_id = _as_int_or_none(parts[0].strip("[]"))
        relation = parts[1]
        tail_id = _as_int_or_none(parts[2].strip("[]"))
        if relation in DOCRED_REL_MAP:
            relation = DOCRED_REL_MAP[relation]
        if head_id is None or tail_id is None or head_id == tail_id:
            continue
        if valid_entity_ids and (head_id not in valid_entity_ids or tail_id not in valid_entity_ids):
            continue
        key = (head_id, relation, tail_id)
        if relation not in valid_names or key in seen:
            continue
        seen.add(key)
        lines.append(f"{head_id} | {relation} | {tail_id}")
    return "\n".join(lines) if lines else "NONE"


def _hard_filter_chemprot_prediction(text: str, entity_list_text: str, document_text: str) -> str:
    candidate = _extract_valid_json_or_none(text)
    if candidate is None:
        return '{"relations": []}'

    try:
        data = json.loads(candidate)
    except Exception:
        return '{"relations": []}'

    chemical_names, gene_y_names, _ = _parse_chemprot_entities(entity_list_text)
    relations = data.get("relations", [])
    if not isinstance(relations, list):
        return '{"relations": []}'

    filtered = []
    seen = set()
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        head = str(rel.get("head", "")).strip()
        relation = str(rel.get("relation", "")).strip().upper()
        tail = str(rel.get("tail", "")).strip()
        if chemical_names and head not in chemical_names:
            continue
        if gene_y_names and tail not in gene_y_names:
            continue
        if relation not in _CHEMPROT_RELATIONS:
            continue
        key = (head.lower(), relation, tail.lower())
        if key in seen:
            continue
        seen.add(key)
        filtered.append({"head": head, "relation": relation, "tail": tail})

    return json.dumps({"relations": filtered}, ensure_ascii=False)

def _parse_docred_entity_ids(entity_list_text: str):
    ids = set()
    for raw_line in (entity_list_text or "").splitlines():
        match = re.match(r"^\s*\[(\d+)\]", raw_line.strip())
        if match:
            ids.add(int(match.group(1)))
    return ids


def _as_int_or_none(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _hard_filter_docred_prediction(text: str, entity_list_text: str = "") -> str:
    from prompts import DOCRED_REL_MAP
    
    candidate = _extract_valid_json_or_none(text)
    if candidate is None:
        return '{"relations": []}'

    try:
        data = json.loads(candidate)
    except Exception:
        return '{"relations": []}'

    valid_names = set(DOCRED_REL_MAP.values())
    valid_entity_ids = _parse_docred_entity_ids(entity_list_text)
    filtered = []
    seen = set()
    
    for rel in data.get("relations", []):
        if not isinstance(rel, dict):
            continue
            
        r_name = str(rel.get("relation", "")).strip()
        head_id = rel.get("head_id") if "head_id" in rel else rel.get("head")
        tail_id = rel.get("tail_id") if "tail_id" in rel else rel.get("tail")
        head_id = _as_int_or_none(head_id)
        tail_id = _as_int_or_none(tail_id)

        if head_id is None or tail_id is None or head_id == tail_id:
            continue
        if valid_entity_ids and (head_id not in valid_entity_ids or tail_id not in valid_entity_ids):
            continue
        
        # 1. 容错补救：如果模型吐出了 P1056 这种 P-ID，强制帮它转成自然语言
        if r_name in DOCRED_REL_MAP:
            r_name = DOCRED_REL_MAP[r_name]
            
        # 2. 强力击杀：如果转换后依然不在那96个合法关系里，直接丢弃（不计入成绩）
        key = (head_id, r_name, tail_id)
        if r_name in valid_names and key not in seen:
            seen.add(key)
            filtered.append({"head_id": head_id, "relation": r_name, "tail_id": tail_id})

    return json.dumps({"relations": filtered}, ensure_ascii=False)


class TextMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens_each: int = 256,
        max_new_tokens_judger: int = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        verifier_model: ModelWrapper = None,
        args: argparse.Namespace = None,
    ) -> None:
        self.model = model
        self.verifier_model = verifier_model or model
        self.max_new_tokens_each = max_new_tokens_each
        self.max_new_tokens_judger = max_new_tokens_judger if max_new_tokens_judger is not None else max_new_tokens_each
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.args = args
        self.method_name = "text_mas"
        self.task = args.task
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @staticmethod
    def _set_model_lora_enabled(model: ModelWrapper, enabled: bool) -> None:
        if model is None:
            return
        setter = getattr(model, "set_lora_enabled", None)
        if callable(setter):
            setter(enabled)

    @staticmethod
    def _should_enable_lora_for_role(task: str, role: str) -> bool:
        if task == "chemprot":
            return role in {"planner", "judger"}
        return role == "judger"

    def _encode_output_tokens(self, text: str):
        encoded = self.model.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors=None,
        )
        token_ids = encoded["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return token_ids, self.model.tokenizer.convert_ids_to_tokens(token_ids)
        
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        contexts = ["" for _ in range(batch_size)]
        history_contexts = ["" for _ in range(batch_size)]
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        for agent in self.agents:
            self._set_model_lora_enabled(
                self.model,
                self._should_enable_lora_for_role(self.args.task, agent.role),
            )

            # Route to extraction prompts for document extraction datasets
            if self.args.task in ['docred', 'cord', 'funsd', 'finer', 'chemprot']:
                from prompts import build_extraction_prompts_text_mas_sequential, build_extraction_prompts_text_mas_hierarchical
                if self.args.prompt == "hierarchical":
                    batch_messages = [
                        build_extraction_prompts_text_mas_hierarchical(
                            dataset=self.args.task,
                            role=agent.role,
                            question=item["question"],
                            context=contexts[idx],
                            item=item,
                            method=self.method_name,
                            args=self.args,
                        )
                        for idx, item in enumerate(items)
                    ]
                else:
                    batch_messages = [
                        build_extraction_prompts_text_mas_sequential(
                            dataset=self.args.task,
                            role=agent.role,
                            question=item["question"],
                            context=contexts[idx],
                            item=item,
                            method=self.method_name,
                            args=self.args,
                        )
                        for idx, item in enumerate(items)
                    ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_messages_hierarchical_text_mas(
                        role=agent.role,
                        question=item["question"],
                        context=contexts[idx],
                        method=self.method_name,
                        args=self.args,
                    )
                    for idx, item in enumerate(items)
                ]
            else:
                batch_messages = [
                    build_agent_messages_sequential_text_mas(
                        role=agent.role,
                        question=item["question"],
                        context=contexts[idx],
                        method=self.method_name,
                        args=self.args,
                    )
                    for idx, item in enumerate(items)
                ]

            prompts, input_ids, attention_mask, tokens_batch, _ = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            _max_tok = self.max_new_tokens_judger if agent.role == "judger" else self.max_new_tokens_each
            repetition_penalty = 1.15 if self.args.task == "chemprot" else 1.0
            no_repeat_ngram_size = 6 if self.args.task == "chemprot" and agent.role != "judger" else 0
            if self.model.use_vllm:
                generated_texts = self.model.vllm_generate_text_batch(
                    prompts,
                    max_new_tokens=_max_tok,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repetition_penalty=repetition_penalty,
                )
            else:
                generated_texts, _ = self.model.generate_text_batch(
                    input_ids,
                    attention_mask,
                    max_new_tokens=_max_tok,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )

            agent_name_map_for_prompt_hierarchical = {
                "Planner": "Math Agent",
                "Critic": "Science Agent",
                "Refiner": "Code Agent",
                "Judger": "Task Summrizer",
                "planner": "Math Agent",
                "critic": "Science Agent",
                "refiner": "Code Agent",
                "judger": "Task Summrizer",
            }

            for idx in range(batch_size):

                text_out = generated_texts[idx].strip()

                if self.args.prompt == "hierarchical":
                    formatted_output = f"[{agent_name_map_for_prompt_hierarchical[agent.name]}]:\n{text_out}\n\n"
                else:
                    formatted_output = f"[{agent.name}]:\n{text_out}\n\n"

                if agent.role != "judger":
                    if self.args.task == "chemprot":
                        clean_output = _sanitize_chemprot_context(text_out, items[idx].get("entity_list", ""))
                        formatted_clean = f"[{agent.name}]:\n{clean_output}\n\n" if clean_output else ""
                        contexts[idx] = formatted_clean
                        history_contexts[idx] = f"{history_contexts[idx]}{formatted_clean}"
                    elif self.args.task == "docred":
                        clean_output = _sanitize_docred_context(text_out, items[idx].get("entity_list", ""))
                        formatted_clean = f"[{agent.name}]:\n{clean_output}\n\n" if clean_output else ""
                        contexts[idx] = formatted_clean
                        history_contexts[idx] = f"{history_contexts[idx]}{formatted_clean}"
                    else:
                        contexts[idx] = f"{contexts[idx]}{formatted_output}"
                        history_contexts[idx] = f"{history_contexts[idx]}{formatted_output}"
                else:
                    final_texts[idx] = text_out
                mask = attention_mask[idx].bool()
                trimmed_ids = input_ids[idx][mask].to("cpu").tolist()
                output_ids, output_tokens = self._encode_output_tokens(text_out)
                self.total_input_tokens += len(tokens_batch[idx])
                self.total_output_tokens += len(output_tokens)
                agent_traces[idx].append(
                    {
                        "name": agent.name,
                        "role": agent.role,
                        "input": prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": tokens_batch[idx],
                        "output": text_out,
                    }
                )

        use_verifier = getattr(self.args, "use_verifier", False)
        verifier_tasks = ["chemprot", "docred", "cord", "funsd"]
        if use_verifier and self.args.task in verifier_tasks:
            from prompts import (
                build_extraction_prompts_text_mas_hierarchical,
                build_extraction_prompts_text_mas_sequential,
            )

            verifier = verifier_agent()
            verifier_model = self.verifier_model
            judger_texts = list(final_texts)
            for idx, item in enumerate(items):
                item["_judger_output"] = final_texts[idx]

            if self.args.prompt == "hierarchical":
                verifier_messages = [
                    build_extraction_prompts_text_mas_hierarchical(
                        dataset=self.args.task,
                        role="verifier",
                        question=item["question"],
                        context="",
                        item=item,
                        method=self.method_name,
                        args=self.args,
                    )
                    for item in items
                ]
            else:
                verifier_messages = [
                    build_extraction_prompts_text_mas_sequential(
                        dataset=self.args.task,
                        role="verifier",
                        question=item["question"],
                        context="",
                        item=item,
                        method=self.method_name,
                        args=self.args,
                    )
                    for item in items
                ]

            try:
                self._set_model_lora_enabled(verifier_model, False)
                v_prompts, v_ids, v_mask, v_tokens_batch, v_extra_inputs = verifier_model.prepare_chat_batch(
                    verifier_messages, add_generation_prompt=True
                )
                if verifier_model.use_vllm:
                    verifier_generated = verifier_model.vllm_generate_text_batch(
                        v_prompts,
                        max_new_tokens=self.max_new_tokens_judger,
                        temperature=_VERIFIER_TEMPERATURE,
                        top_p=_VERIFIER_TOP_P,
                        repetition_penalty=1.1,
                    )
                else:
                    verifier_generated, _ = verifier_model.generate_text_batch(
                        v_ids,
                        v_mask,
                        max_new_tokens=self.max_new_tokens_judger,
                        temperature=_VERIFIER_TEMPERATURE,
                        top_p=_VERIFIER_TOP_P,
                        past_key_values=None,
                        repetition_penalty=1.1,
                        pixel_values=v_extra_inputs.get("pixel_values") if v_extra_inputs else None,
                        image_grid_thw=v_extra_inputs.get("image_grid_thw") if v_extra_inputs else None,
                    )

                for idx in range(batch_size):
                    verifier_text = verifier_generated[idx].strip()
                    valid_json = _extract_valid_json_or_none(verifier_text)
                    if valid_json is not None and self.args.task == "funsd":
                        final_texts[idx] = _stabilize_funsd_verifier_json(judger_texts[idx], verifier_text)
                    elif valid_json is not None and self.args.task == "cord":
                        final_texts[idx] = _stabilize_cord_verifier_json(judger_texts[idx], verifier_text)
                    else:
                        final_texts[idx] = valid_json if valid_json is not None else judger_texts[idx]
                    mask = v_mask[idx].bool()
                    trimmed_ids = v_ids[idx][mask].to("cpu").tolist()
                    output_ids, output_tokens = self._encode_output_tokens(verifier_text)
                    self.total_input_tokens += len(v_tokens_batch[idx])
                    self.total_output_tokens += len(output_tokens)
                    agent_traces[idx].append(
                        {
                            "name": verifier.name,
                            "role": verifier.role,
                            "input": v_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": v_tokens_batch[idx],
                            "output": verifier_text,
                        }
                    )
            finally:
                self._set_model_lora_enabled(self.model, True)
                for item in items:
                    item.pop("_judger_output", None)
        else:
            self._set_model_lora_enabled(self.model, True)

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]

            # ====== 新增：剥离 <think> 标签，精准提取 JSON ======
            import re as _re
            stripped = _re.sub(r"<think>.*?</think>", "", final_text, flags=_re.DOTALL).strip()
            start_idx = stripped.find('{')
            end_idx = stripped.rfind('}')
            if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
                cleaned_text = stripped[start_idx:end_idx+1]
            else:
                cleaned_text = stripped
            # =================================================

            # ====== ChemProt 硬过滤：只保留合法实体/关系，并兜底为合法 JSON ======
            if self.task == "chemprot":
                cleaned_text = _hard_filter_chemprot_prediction(
                    cleaned_text,
                    item.get("entity_list", ""),
                    item.get("question", ""),
                )
            # ====== 新增 DocRED 的硬过滤 ======
            elif self.task == "docred":
                cleaned_text = _hard_filter_docred_prediction(cleaned_text, item.get("entity_list", ""))
            # ======================================================

            # 注意这里传入的是 cleaned_text
            eval_result = evaluate_prediction(self.task, cleaned_text, item, idx)
            pred = eval_result["prediction"]
            gold = eval_result["gold"]
            ok = eval_result["correct"]

            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "context": history_contexts[idx],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                    "entities_meta": item.get("entities_meta", []),
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
