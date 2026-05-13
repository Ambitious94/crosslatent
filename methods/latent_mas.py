from typing import Dict, List, Optional, Tuple
import json
import re

from . import default_agents, verifier_agent
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import evaluate_prediction
import torch
import argparse

try:
    from vllm import SamplingParams
    _HAS_VLLM = True
except ImportError:
    SamplingParams = None
    _HAS_VLLM = False

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None

_VERIFIER_TEMPERATURE = 0.0
_VERIFIER_TOP_P = 1.0


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

def _parse_docred_entity_ids(entity_list_text: str):
    import re

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
    import json

    try:
        data = json.loads(text)
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

        if r_name in DOCRED_REL_MAP:
            r_name = DOCRED_REL_MAP[r_name]

        key = (head_id, r_name, tail_id)
        if r_name in valid_names and key not in seen:
            seen.add(key)
            filtered.append({"head_id": head_id, "relation": r_name, "tail_id": tail_id})

    return json.dumps({"relations": filtered}, ensure_ascii=False)

class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        verifier_model: ModelWrapper = None,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.verifier_model = verifier_model or model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.method_name = 'latent_mas'
        self.vllm_device = args.device 
        self.HF_device = args.device2
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False

        if self.latent_only:
            self.sequential_info_only = True

        if _HAS_VLLM and SamplingParams is not None:
            self.sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=args.max_new_tokens,
                repetition_penalty=1.1,
            )
        else:
            self.sampling_params = None
        self.task = args.task
        self.total_input_tokens = 0
        self.total_output_tokens = 0

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

    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            trimmed_legacy = tuple(
                tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                for layer in legacy
            )
            return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        # 检查是否使用LoRA模型
        use_lora = hasattr(self.args, 'lora_weights') and self.args.lora_weights
        
        # 注意: LoRA模型同样走完整的多Agent流程(Planner→Critic→Refiner→Judger)
        # 之前的直接推理旁路已移除,因为多Agent reasoning能显著提升LoRA模型的抽取质�?
        for agent in self.agents:

            # Route to extraction prompts for document extraction datasets
            if self.args.task in ['docred', 'cord', 'funsd', 'finer', 'chemprot']:
                # 原始模型使用详细prompts
                from prompts import build_extraction_prompts_sequential, build_extraction_prompts_hierarchical
                if self.args.prompt == "sequential":
                    batch_messages = [
                        build_extraction_prompts_sequential(dataset=self.args.task, role=agent.role, question=item["question"], item=item, method=self.method_name, args=self.args)
                        for item in items
                    ]
                elif self.args.prompt == "hierarchical":
                    batch_messages = [
                        build_extraction_prompts_hierarchical(dataset=self.args.task, role=agent.role, question=item["question"], item=item, method=self.method_name, args=self.args)
                        for item in items
                    ]
            else:
                # Original prompts for existing tasks
                if self.args.prompt == "sequential":
                    batch_messages = [
                        build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                        for item in items
                    ]
                elif self.args.prompt == "hierarchical":
                    batch_messages = [
                        build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                        for item in items
                    ]

            prompts, input_ids, attention_mask, tokens_batch, extra_inputs = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                use_think_for_judger = self.args.think and self.args.task not in {"docred", "cord", "funsd", "finer", "chemprot"}

                if use_think_for_judger:
                    # For vision models, append <think> token to input_ids instead of re-tokenizing
                    if self.model.is_vision_model and extra_inputs:
                        think_token_id = self.model.tokenizer.encode("<think>", add_special_tokens=False)[0]
                        # Append think token to each sequence
                        think_ids = torch.full((input_ids.shape[0], 1), think_token_id, dtype=input_ids.dtype, device=input_ids.device)
                        wrapped_ids = torch.cat([input_ids, think_ids], dim=1)
                        wrapped_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
                    else:
                        # Text-only model: re-tokenize with <think>
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                        wrapped_encoded = self.model.tokenizer(
                            wrapped_prompts,
                            return_tensors="pt",
                            padding=True,
                            add_special_tokens=False,
                        )
                        wrapped_ids = wrapped_encoded["input_ids"].to(self.model.device)
                        wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.device)
                else:
                    wrapped_ids = input_ids
                    wrapped_mask = attention_mask

                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                # Prepare vision inputs if available (only for first agent to encode images)
                vision_kwargs = {}
                if extra_inputs and agent.role == "planner":
                    # Only pass pixel_values for the first agent (planner)
                    # Subsequent agents will use the KV-cache which already contains vision information
                    vision_kwargs = {
                        "pixel_values": extra_inputs.get("pixel_values"),
                        "image_grid_thw": extra_inputs.get("image_grid_thw"),
                    }
                
                past_kv = self.model.generate_latent_batch(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                    **vision_kwargs,
                )
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    self.total_input_tokens += len(wrapped_tokens_batch[idx])
                    self.total_output_tokens += int(self.latent_steps)
                    # For vision models, use text representation of input_ids instead of wrapped_prompts
                    input_repr = prompts[idx] if not (self.model.is_vision_model and extra_inputs) else f"[Vision input with {len(trimmed_ids)} tokens]"
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": input_repr,
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "output": "",
                        }
                    )
            else:

                past_for_decoding = past_kv if self.latent_steps > 0 else None

                use_think_for_judger = self.args.think and self.args.task not in {"docred", "cord", "funsd", "finer", "chemprot"}

                if use_think_for_judger:
                    # For vision models, append <think> token to input_ids
                    if self.model.is_vision_model and extra_inputs:
                        think_token_id = self.model.tokenizer.encode("<think>", add_special_tokens=False)[0]
                        think_ids = torch.full((input_ids.shape[0], 1), think_token_id, dtype=input_ids.dtype, device=input_ids.device)
                        judger_ids = torch.cat([input_ids, think_ids], dim=1)
                        judger_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
                        judger_prompts = prompts  # Use original prompts for logging
                    else:
                        # Text-only model: re-tokenize with <think>
                        judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                        judger_encoded = self.model.tokenizer(
                            judger_prompts,
                            return_tensors="pt",
                            padding=True,
                            add_special_tokens=False,
                        )
                        judger_ids = judger_encoded["input_ids"].to(self.model.device)
                        judger_mask = judger_encoded["attention_mask"].to(self.model.device)
                else:
                    judger_ids = input_ids
                    judger_mask = attention_mask
                    judger_prompts = prompts  # Use original prompts for logging
                
                judger_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(judger_ids, judger_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    judger_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))
                generated_batch, _ = self.model.generate_text_batch(
                    judger_ids,
                    judger_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    past_key_values=past_for_decoding,
                    repetition_penalty=1.1,
                )
                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    mask = judger_mask[idx].bool()
                    trimmed_ids = judger_ids[idx][mask].to("cpu").tolist()
                    output_ids, output_tokens = self._encode_output_tokens(final_text)
                    self.total_input_tokens += len(judger_tokens_batch[idx])
                    self.total_output_tokens += len(output_tokens)
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": judger_tokens_batch[idx],
                            "output": final_text,
                        }
                    )

        use_verifier = getattr(self.args, "use_verifier", False)
        verifier_tasks = ["chemprot", "docred", "cord", "funsd"]
        if use_verifier and self.args.task in verifier_tasks:
            from prompts import build_extraction_prompts_hierarchical, build_extraction_prompts_sequential

            verifier = verifier_agent()
            verifier_model = self.verifier_model
            judger_texts = list(final_texts)
            for idx, item in enumerate(items):
                item["_judger_output"] = final_texts[idx]

            if self.args.prompt == "hierarchical":
                verifier_messages = [
                    build_extraction_prompts_hierarchical(
                        dataset=self.args.task,
                        role="verifier",
                        question=item["question"],
                        item=item,
                        method=self.method_name,
                        args=self.args,
                    )
                    for item in items
                ]
            else:
                verifier_messages = [
                    build_extraction_prompts_sequential(
                        dataset=self.args.task,
                        role="verifier",
                        question=item["question"],
                        item=item,
                        method=self.method_name,
                        args=self.args,
                    )
                    for item in items
                ]

            try:
                v_prompts, v_ids, v_mask, v_tokens_batch, v_extra_inputs = verifier_model.prepare_chat_batch(
                    verifier_messages, add_generation_prompt=True
                )
                verifier_generated, _ = verifier_model.generate_text_batch(
                    v_ids,
                    v_mask,
                    max_new_tokens=self.judger_max_new_tokens,
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
                for item in items:
                    item.pop("_judger_output", None)

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]

            # ====== 新增：剥�?<think> 标签，精准提�?JSON ======
            import re as _re
            stripped = _re.sub(r"<think>.*?</think>", "", final_text, flags=_re.DOTALL).strip()
            start_idx = stripped.find('{')
            end_idx = stripped.rfind('}')
            if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
                cleaned_text = stripped[start_idx:end_idx+1]
            else:
                cleaned_text = stripped

            # ====== ChemProt 去重：模型复读时同一三元组会重复出现 ======
            if self.task == "chemprot":
                import json as _json
                try:
                    _data = _json.loads(cleaned_text)
                    _rels = _data.get("relations", [])
                    if isinstance(_rels, list):
                        _seen = set()
                        _deduped = []
                        for _r in _rels:
                            if isinstance(_r, dict):
                                _key = (
                                    str(_r.get("head", "")).strip().lower(),
                                    str(_r.get("relation", "")).strip().lower(),
                                    str(_r.get("tail", "")).strip().lower(),
                                )
                                if _key not in _seen:
                                    _seen.add(_key)
                                    _deduped.append(_r)
                        _data["relations"] = _deduped
                        cleaned_text = _json.dumps(_data, ensure_ascii=False)
                except Exception:
                    pass
            # ======================================================

            # ====== 终极修复：坐标对齐吸�?(Coordinate Snapping) ======
            if self.task == "finer":
                import json
                try:
                    data = json.loads(cleaned_text)
                    doc_text = item.get("question", "")
                    for ent in data.get("entities", []):
                        ent_text = ent.get("text", "")
                        pred_start = ent.get("start", 0)

                        # Snap extracted entity offsets back to the source text when possible.
                        if ent_text and ent_text in doc_text:
                            starts = []
                            idx_search = doc_text.find(ent_text)
                            while idx_search != -1:
                                starts.append(idx_search)
                                idx_search = doc_text.find(ent_text, idx_search + 1)

                            if starts:
                                # 寻找距离模型预测坐标（带偏移的坐标）最近的真实坐标
                                real_start = min(starts, key=lambda x: abs(x - pred_start))
                                ent["start"] = real_start
                                ent["end"] = real_start + len(ent_text)

                    cleaned_text = json.dumps(data)
                except Exception:
                    pass
            # ====== 新增 DocRED 的硬过滤 ======
            elif self.task == "docred":
                cleaned_text = _hard_filter_docred_prediction(cleaned_text, item.get("entity_list", ""))
            # =================================================

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
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                    "entities_meta": item.get("entities_meta", []),
                }
            )
        return results
    
    def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        embedding_record = []
        for agent in self.agents:
            
            # Route to extraction prompts for document extraction datasets
            if self.args.task in ['docred', 'cord', 'funsd', 'finer', 'chemprot']:
                from prompts import build_extraction_prompts_sequential, build_extraction_prompts_hierarchical
                if self.args.prompt == "sequential":
                    batch_messages = [
                        build_extraction_prompts_sequential(dataset=self.args.task, role=agent.role, question=item["question"], item=item, method=self.method_name, args=self.args)
                        for item in items
                    ]
                elif self.args.prompt == "hierarchical":
                    batch_messages = [
                        build_extraction_prompts_hierarchical(dataset=self.args.task, role=agent.role, question=item["question"], item=item, method=self.method_name, args=self.args)
                        for item in items
                    ]
            else:
                # Original prompts for existing tasks
                if self.args.prompt == "sequential":
                    batch_messages = [
                        build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                        for item in items
                    ]
                elif self.args.prompt == "hierarchical":
                    batch_messages = [
                        build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                        for item in items
                    ]
                
            prompts, input_ids, attention_mask, tokens_batch, extra_inputs = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                # to wrap all latent thoughts from previous agents
                if self.args.think:
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.HF_device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                past_kv, previous_hidden_embedding = self.model.generate_latent_batch_hidden_state(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                if self.latent_only:
                    if self.latent_steps > 0:
                        previous_hidden_embedding = previous_hidden_embedding[:, -self.latent_steps:, :]
                    else:
                        previous_hidden_embedding = previous_hidden_embedding[:, 0:0, :]

                embedding_record.append(previous_hidden_embedding)

                if self.sequential_info_only or self.latent_only:
                    embedding_record = embedding_record[-1:]
                
                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    self.total_input_tokens += len(wrapped_tokens_batch[idx])
                    self.total_output_tokens += int(self.latent_steps)
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "output": "",
                        }
                    )
            else:
                
                # A stack of [B, L_i, H]
                past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)
                
                use_think_for_judger = self.args.think and self.args.task not in {"docred", "cord", "funsd", "finer", "chemprot"}
                
                if use_think_for_judger:
                    judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    judger_prompts = prompts
                
                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ) 
                judger_encoded = judger_encoded["input_ids"].to(self.model.HF_device)
                # Get current prompt embedding �?keep the batch dimension intact.
                # squeeze(0) would collapse [1, L, H] �?[L, H] when batch_size=1,
                # causing curr_prompt_emb[i] to index a single token vector instead
                # of the i-th sample sequence, corrupting all downstream logic.
                curr_prompt_emb = self.model.embedding_layer(judger_encoded).to(self.vllm_device)
                
                # assert Qwen model
                assert "Qwen" in self.args.model_name or "qwen" in self.args.model_name, "latent_embedding_position is only supported for Qwen models currently."

                # handle latent embedding insertion position    
                len_of_left = []
                for p in judger_prompts:
                    idx = p.find("<|im_start|>user\n")
                    # Get the text up to and including "<|im_start|>user\n"
                    left = p[: idx + len("<|im_start|>user\n")]
                    len_of_left.append(len(self.model.tokenizer(left)['input_ids']))
                    
                B, L, H = curr_prompt_emb.shape
                _, Lp, H = past_embedding.shape  # assume shape consistency
                    
                whole_prompt_emb_list = []
                for i in range(B):
                    insert_idx = len_of_left[i]
                    left_emb = curr_prompt_emb[i, :insert_idx, :]
                    right_emb = curr_prompt_emb[i, insert_idx:, :]
                    combined = torch.cat([left_emb, past_embedding[i], right_emb], dim=0)
                    whole_prompt_emb_list.append(combined)

                # Pass variable-length embeddings directly to vLLM;
                # vLLM natively handles variable-length inputs without padding.
                # Zero-padding would pollute short sequences with meaningless vectors.
                prompt_embeds_list = [
                    {
                        "prompt_embeds": embeds
                    } for embeds in whole_prompt_emb_list
                ]
                
                
                outputs = self.model.vllm_engine.generate(
                    prompt_embeds_list,
                    self.sampling_params,
                )

                generated_texts = [out.outputs[0].text.strip() for out in outputs]
                    
                for idx in range(batch_size):
                    text_out = generated_texts[idx].strip()
                    final_texts[idx] = text_out
                    output_ids, output_tokens = self._encode_output_tokens(text_out)
                    self.total_input_tokens += len(self.model.tokenizer(judger_prompts[idx], add_special_tokens=False)["input_ids"])
                    self.total_output_tokens += len(output_tokens)
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "output": text_out,
                        }
                    )

        use_verifier = getattr(self.args, "use_verifier", False)
        verifier_tasks = ["chemprot", "docred", "cord", "funsd"]
        if use_verifier and self.args.task in verifier_tasks:
            from prompts import build_extraction_prompts_hierarchical, build_extraction_prompts_sequential

            verifier = verifier_agent()
            verifier_model = self.verifier_model
            judger_texts = list(final_texts)
            for idx, item in enumerate(items):
                item["_judger_output"] = final_texts[idx]

            if self.args.prompt == "hierarchical":
                verifier_messages = [
                    build_extraction_prompts_hierarchical(
                        dataset=self.args.task,
                        role="verifier",
                        question=item["question"],
                        item=item,
                        method=self.method_name,
                        args=self.args,
                    )
                    for item in items
                ]
            else:
                verifier_messages = [
                    build_extraction_prompts_sequential(
                        dataset=self.args.task,
                        role="verifier",
                        question=item["question"],
                        item=item,
                        method=self.method_name,
                        args=self.args,
                    )
                    for item in items
                ]

            try:
                v_prompts, v_ids, v_mask, v_tokens_batch, _ = verifier_model.prepare_chat_batch(
                    verifier_messages, add_generation_prompt=True
                )
                verifier_generated = verifier_model.vllm_generate_text_batch(
                    v_prompts,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=_VERIFIER_TEMPERATURE,
                    top_p=_VERIFIER_TOP_P,
                    repetition_penalty=1.1,
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
                for item in items:
                    item.pop("_judger_output", None)


        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]

            # ====== 新增：剥�?<think> 标签，精准提�?JSON ======
            import re as _re
            stripped = _re.sub(r"<think>.*?</think>", "", final_text, flags=_re.DOTALL).strip()
            start_idx = stripped.find('{')
            end_idx = stripped.rfind('}')
            if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
                cleaned_text = stripped[start_idx:end_idx+1]
            else:
                cleaned_text = stripped

            # ====== ChemProt 去重：模型复读时同一三元组会重复出现 ======
            if self.task == "chemprot":
                import json as _json
                try:
                    _data = _json.loads(cleaned_text)
                    _rels = _data.get("relations", [])
                    if isinstance(_rels, list):
                        _seen = set()
                        _deduped = []
                        for _r in _rels:
                            if isinstance(_r, dict):
                                _key = (
                                    str(_r.get("head", "")).strip().lower(),
                                    str(_r.get("relation", "")).strip().lower(),
                                    str(_r.get("tail", "")).strip().lower(),
                                )
                                if _key not in _seen:
                                    _seen.add(_key)
                                    _deduped.append(_r)
                        _data["relations"] = _deduped
                        cleaned_text = _json.dumps(_data, ensure_ascii=False)
                except Exception:
                    pass
            # ======================================================

            # ====== 终极修复：坐标对齐吸�?(Coordinate Snapping) ======
            if self.task == "finer":
                import json
                try:
                    data = json.loads(cleaned_text)
                    doc_text = item.get("question", "")
                    for ent in data.get("entities", []):
                        ent_text = ent.get("text", "")
                        pred_start = ent.get("start", 0)

                        # Snap extracted entity offsets back to the source text when possible.
                        if ent_text and ent_text in doc_text:
                            starts = []
                            idx_search = doc_text.find(ent_text)
                            while idx_search != -1:
                                starts.append(idx_search)
                                idx_search = doc_text.find(ent_text, idx_search + 1)

                            if starts:
                                # 寻找距离模型预测坐标（带偏移的坐标）最近的真实坐标
                                real_start = min(starts, key=lambda x: abs(x - pred_start))
                                ent["start"] = real_start
                                ent["end"] = real_start + len(ent_text)

                    cleaned_text = json.dumps(data)
                except Exception:
                    pass
            # ====== 新增 DocRED 的硬过滤 ======
            elif self.task == "docred":
                cleaned_text = _hard_filter_docred_prediction(cleaned_text, item.get("entity_list", ""))
            # =================================================

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
