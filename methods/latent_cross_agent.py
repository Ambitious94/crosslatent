from typing import Dict, List, Optional, Tuple
import json

import torch

from models import ModelWrapper, _past_length
from utils import evaluate_prediction
from prompts_crossagent import (
    CONLL04_ENTITY_TYPES,
    CONLL04_RELATION_TYPES,
    build_conll04_ner_type_prompt,
    build_conll04_re_debate_prompt,
    build_conll04_re_type_prompt,
)
from prompts_latent_crossagent import (
    build_conll04_latent_cross_task_decode_prompt,
    build_conll04_latent_cross_task_seed_prompt,
    build_conll04_latent_ner_decode_prompt,
    build_conll04_latent_ner_read_prompt,
    build_conll04_latent_ner_type_prompt,
    build_conll04_latent_re_decode_prompt,
    build_conll04_latent_re_c2c_decode_prompt,
    build_conll04_latent_re_read_prompt,
    build_conll04_latent_re_type_prompt,
    build_conll04_text_anchor_prompt,
)
from .cross_agent import _clean_entities, _clean_relations, _extract_json, _json_items


class LatentCrossAgentMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 5,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        generate_bs: int = 1,
        args=None,
    ) -> None:
        self.model = model
        self.latent_steps = max(0, int(latent_steps))
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = 1
        self.args = args
        self.task = args.task
        self.method_name = "latent_cross_agent"
        self.fusion_mode = getattr(args, "latent_cross_fusion", "pure") if args else "pure"
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        if generate_bs != 1:
            print("[INFO] latent_cross_agent v1 forces generate_bs=1")
        if self.fusion_mode not in {"pure", "re_text_cache", "text_cache", "re_c2c"}:
            raise ValueError("--latent_cross_fusion must be one of: pure, re_text_cache, text_cache, re_c2c")

    def _encode_output_tokens(self, text: str):
        encoded = self.model.tokenizer(text, add_special_tokens=False, return_tensors=None)
        token_ids = encoded["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return token_ids, self.model.tokenizer.convert_ids_to_tokens(token_ids)

    def _prepare(self, messages: List[Dict]):
        return self.model.prepare_chat_batch([messages], add_generation_prompt=True)

    def _concat_past(self, left: Optional[Tuple], right: Optional[Tuple]):
        if left is None:
            return right
        if right is None:
            return left

        left_is_cache = hasattr(left, "to_legacy_cache")
        right_is_cache = hasattr(right, "to_legacy_cache")
        left_legacy = left.to_legacy_cache() if left_is_cache else left
        right_legacy = right.to_legacy_cache() if right_is_cache else right

        merged_layers = []
        for left_layer, right_layer in zip(left_legacy, right_legacy):
            if isinstance(left_layer, tuple) and isinstance(right_layer, tuple):
                merged_layers.append(
                    tuple(torch.cat([l_tensor, r_tensor], dim=-2) for l_tensor, r_tensor in zip(left_layer, right_layer))
                )
            elif torch.is_tensor(left_layer) and torch.is_tensor(right_layer):
                merged_layers.append(torch.cat([left_layer, right_layer], dim=-2))
            else:
                merged_layers.append(left_layer)

        if left_is_cache:
            return left.__class__.from_legacy_cache(tuple(merged_layers))
        if right_is_cache:
            return right.__class__.from_legacy_cache(tuple(merged_layers))
        return tuple(merged_layers)

    @torch.no_grad()
    def _generate_text_candidate(
        self,
        messages: List[Dict],
        *,
        name: str,
        role: str,
        max_new_tokens: int,
    ):
        prompts, input_ids, attention_mask, tokens_batch, _ = self._prepare(messages)
        generated_batch, text_past = self.model.generate_text_batch(
            input_ids,
            attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=1.1,
        )
        generated = generated_batch[0].strip()
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
        return generated, text_past, trace

    @torch.no_grad()
    def _build_anchor_cache(
        self,
        *,
        kind: str,
        type_name: str,
        sentence: str,
        candidates,
    ):
        messages = build_conll04_text_anchor_prompt(kind, type_name, sentence, candidates)
        prompts, input_ids, attention_mask, tokens_batch, _ = self._prepare(messages)
        past = self.model.generate_latent_batch(
            input_ids,
            attention_mask=attention_mask,
            latent_steps=0,
            past_key_values=None,
        )
        mask = attention_mask[0].bool()
        trimmed_ids = input_ids[0][mask].to("cpu").tolist()
        self.total_input_tokens += len(tokens_batch[0])
        trace = {
            "name": f"{type_name}_Text_Anchor_Cache",
            "role": f"text_cache_anchor_{kind}",
            "input": prompts[0],
            "input_ids": trimmed_ids,
            "input_tokens": tokens_batch[0],
            "latent_steps": 0,
            "latent_tokens_added": _past_length(past),
            "output": "",
        }
        return past, trace

    def _fuse_type_context(
        self,
        debate_past: Optional[Tuple],
        anchor_past: Optional[Tuple],
        type_past: Optional[Tuple],
    ):
        fused = self._concat_past(debate_past, anchor_past)
        return self._concat_past(fused, type_past)

    @torch.no_grad()
    def _latent_one(
        self,
        messages: List[Dict],
        *,
        name: str,
        role: str,
        past_key_values: Optional[Tuple] = None,
    ):
        prompts, input_ids, attention_mask, tokens_batch, _ = self._prepare(messages)
        prev_past_len = _past_length(past_key_values)
        past = self.model.generate_latent_batch(
            input_ids,
            attention_mask=attention_mask,
            latent_steps=self.latent_steps,
            past_key_values=past_key_values,
        )
        new_past_len = _past_length(past)
        mask = attention_mask[0].bool()
        trimmed_ids = input_ids[0][mask].to("cpu").tolist()
        self.total_input_tokens += len(tokens_batch[0])
        self.total_output_tokens += self.latent_steps
        trace = {
            "name": name,
            "role": role,
            "input": prompts[0],
            "input_ids": trimmed_ids,
            "input_tokens": tokens_batch[0],
            "latent_steps": self.latent_steps,
            "latent_tokens_added": new_past_len - prev_past_len,
            "output": "",
        }
        return past, trace

    @torch.no_grad()
    def _decode_one(
        self,
        messages: List[Dict],
        *,
        name: str,
        role: str,
        past_key_values: Optional[Tuple] = None,
    ):
        prompts, input_ids, attention_mask, tokens_batch, _ = self._prepare(messages)
        generated_batch, _ = self.model.generate_text_batch(
            input_ids,
            attention_mask,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            past_key_values=past_key_values if self.latent_steps > 0 else None,
            repetition_penalty=1.1,
        )
        generated = generated_batch[0].strip()
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

        ner_type_pasts = []
        for entity_type in CONLL04_ENTITY_TYPES:
            anchor_past = None
            if self.fusion_mode == "text_cache":
                candidate_text, _, trace = self._generate_text_candidate(
                    build_conll04_ner_type_prompt(entity_type, sentence),
                    name=f"{entity_type}_Text_Candidate_Agent",
                    role="text_ner_type_agent",
                    max_new_tokens=min(self.max_new_tokens, 256),
                )
                traces.append(trace)
                entities_candidate = _clean_entities(
                    _json_items(_extract_json(candidate_text), "entities"),
                    keep_confidence=True,
                )
                if entities_candidate:
                    anchor_past, trace = self._build_anchor_cache(
                        kind="NER",
                        type_name=entity_type,
                        sentence=sentence,
                        candidates={"entities": entities_candidate},
                    )
                    traces.append(trace)

            type_past, trace = self._latent_one(
                build_conll04_latent_ner_type_prompt(entity_type, sentence),
                name=f"{entity_type}_Latent_Agent",
                role="latent_ner_type_agent",
                past_key_values=None,
            )
            ner_type_pasts.append((entity_type, type_past, anchor_past))
            traces.append(trace)

        ner_debate_past = None
        for entity_type, type_past, anchor_past in ner_type_pasts:
            read_context = self._fuse_type_context(ner_debate_past, anchor_past, type_past)
            ner_debate_past, trace = self._latent_one(
                build_conll04_latent_ner_read_prompt(entity_type, sentence),
                name=f"NER_Debate_Read_{entity_type}",
                role="latent_ner_debate_reader",
                past_key_values=read_context,
            )
            traces.append(trace)

        ner_text, trace = self._decode_one(
            build_conll04_latent_ner_decode_prompt(sentence),
            name="NER_Latent_Debate_Agent",
            role="ner_debate",
            past_key_values=ner_debate_past,
        )
        traces.append(trace)
        entities = _clean_entities(_json_items(_extract_json(ner_text), "entities"))

        re_type_pasts = []
        re_text_candidates = []
        for relation_type in CONLL04_RELATION_TYPES:
            anchor_past = None
            text_type_past = None
            relations_candidate = []
            if self.fusion_mode in {"re_text_cache", "text_cache", "re_c2c"}:
                candidate_text, text_type_past, trace = self._generate_text_candidate(
                    build_conll04_re_type_prompt(relation_type, sentence, entities),
                    name=f"{relation_type}_Text_Candidate_Agent",
                    role="text_re_type_agent",
                    max_new_tokens=min(self.max_new_tokens, 384),
                )
                traces.append(trace)
                relations_candidate = _clean_relations(
                    _json_items(_extract_json(candidate_text), "relations"),
                    keep_confidence=True,
                )
                re_text_candidates.extend(relations_candidate)
                if relations_candidate and self.fusion_mode in {"re_text_cache", "text_cache"}:
                    anchor_past, trace = self._build_anchor_cache(
                        kind="RE",
                        type_name=relation_type,
                        sentence=sentence,
                        candidates={"relations": relations_candidate},
                    )
                    traces.append(trace)

            type_past, trace = self._latent_one(
                build_conll04_latent_re_type_prompt(relation_type, sentence, entities),
                name=f"{relation_type}_Latent_Agent",
                role="latent_re_type_agent",
                past_key_values=text_type_past if self.fusion_mode == "re_c2c" else None,
            )
            re_type_pasts.append((relation_type, type_past, anchor_past, bool(relations_candidate) if self.fusion_mode == "re_c2c" else True))
            traces.append(trace)

        text_re_debate_past = None
        if self.fusion_mode == "re_c2c":
            text_re_debate_input = _clean_relations(re_text_candidates, keep_confidence=True)
            _, text_re_debate_past, trace = self._generate_text_candidate(
                build_conll04_re_debate_prompt(sentence, text_re_debate_input),
                name="RE_Text_Debate_Agent",
                role="text_re_debate",
                max_new_tokens=min(self.max_new_tokens, 512),
            )
            traces.append(trace)

        if self.fusion_mode == "re_c2c":
            selected_type_past = None
            for _relation_type, type_past, _anchor_past, has_text_candidate in re_type_pasts:
                if has_text_candidate:
                    selected_type_past = self._concat_past(selected_type_past, type_past)
            re_decode_context = self._concat_past(selected_type_past, text_re_debate_past)
            re_text, trace = self._decode_one(
                build_conll04_latent_re_c2c_decode_prompt(sentence, entities),
                name="RE_Latent_C2C_Debate_Agent",
                role="re_debate",
                past_key_values=re_decode_context,
            )
            traces.append(trace)
        else:
            re_debate_past = None
            for relation_type, type_past, anchor_past, _has_text_candidate in re_type_pasts:
                read_context = self._fuse_type_context(re_debate_past, anchor_past, type_past)
                re_debate_past, trace = self._latent_one(
                    build_conll04_latent_re_read_prompt(relation_type, sentence, entities),
                    name=f"RE_Debate_Read_{relation_type}",
                    role="latent_re_debate_reader",
                    past_key_values=read_context,
                )
                traces.append(trace)

            re_text, trace = self._decode_one(
                build_conll04_latent_re_decode_prompt(sentence, entities),
                name="RE_Latent_Debate_Agent",
                role="re_debate",
                past_key_values=re_debate_past,
            )
            traces.append(trace)
        relations = _clean_relations(_json_items(_extract_json(re_text), "relations"))

        verifier_past, trace = self._latent_one(
            build_conll04_latent_cross_task_seed_prompt(sentence, entities, relations),
            name="Cross_Task_Latent_Verifier",
            role="latent_cross_task_verifier",
            past_key_values=None,
        )
        traces.append(trace)
        final_text, trace = self._decode_one(
            build_conll04_latent_cross_task_decode_prompt(sentence, entities, relations),
            name="Cross_Task_Final_Decoder",
            role="cross_task_verifier",
            past_key_values=verifier_past,
        )
        traces.append(trace)

        final_data = _extract_json(final_text)
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
            "raw_prediction": final_text,
            "agents": traces,
            "correct": eval_result["correct"],
        }

    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if self.task != "conll04":
            raise ValueError("latent_cross_agent currently supports only --task conll04")
        if getattr(self.args, "use_vllm", False):
            raise ValueError("latent_cross_agent v1 supports HF backend only")
        if len(items) != 1:
            raise ValueError("latent_cross_agent v1 requires generate_bs=1")
        return [self._run_item_conll04(items[0])]

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
