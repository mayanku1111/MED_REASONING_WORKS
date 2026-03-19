from __future__ import annotations

import csv
import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_lens import SAE
from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer
from sae_lens.util import extract_stop_at_layer_from_tlens_hook_name

ANSWER_PATTERN = re.compile(r"\b([A-I])\b")
OPTION_PATTERN = re.compile(r"\s*([A-I])\s*:\s*(.+?)\s*(?=(?:[A-I]\s*:)|$)")


@dataclass
class QAExample:
	question_id: int
	question: str
	options: dict[str, str]
	answer_key: str


@dataclass
class DecodeConfig:
	max_new_tokens: int = 256
	temperature: float = 0.0
	top_p: float = 1.0
	do_sample: bool = False


@dataclass
class ModelSpec:
	name: str
	hf_model_name: str
	sae_release: str
	sae_id: str
	tlens_model_name: str | None = None
	device: str = "cpu"
	dtype: str = "float32"
	hf_model_kwargs: dict[str, Any] = field(default_factory=dict)
	tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationOutput:
	text: str
	first_generated_token_id: int | None
	token_ids: list[int]


@dataclass
class ExampleRunRecord:
	question_id: int
	model_name: str
	condition: str
	answer_text: str
	parsed_answer: str | None
	is_correct: bool | None
	first_token_id: int | None
	reasoning_text: str | None = None


@dataclass
class DiscoveryArtifacts:
	direct_features: list[torch.Tensor]
	cot_features: list[torch.Tensor]
	direct_records: list[ExampleRunRecord]
	cot_records: list[ExampleRunRecord]


@dataclass
class ModelRuntime:
	spec: ModelSpec
	generation_model: Any
	generation_tokenizer: Any
	cache_model: Any | None
	sae: SAE[Any]
	hf_hook_module_path: str | None = None


def parse_options(options_text: str) -> dict[str, str]:
	matches = OPTION_PATTERN.findall(options_text)
	if not matches:
		raise ValueError(f"Could not parse options: {options_text[:120]}")
	return {letter.strip(): text.strip() for letter, text in matches}


def load_confidential_dataset(csv_path: str | Path) -> list[QAExample]:
	path = Path(csv_path)
	examples: list[QAExample] = []
	with path.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.reader(handle)
		for idx, row in enumerate(reader):
			if len(row) < 3:
				continue
			question = row[0].strip()
			options = parse_options(row[1].strip())
			answer_key = row[2].strip().upper()
			if answer_key and answer_key[0] in options:
				answer_key = answer_key[0]
			examples.append(
				QAExample(
					question_id=idx,
					question=question,
					options=options,
					answer_key=answer_key,
				)
			)
	return examples


def format_mcq(example: QAExample) -> str:
	options = "\n".join(f"{key}: {value}" for key, value in example.options.items())
	return f"Question:\n{example.question}\n\nOptions:\n{options}"


def direct_prompt(example: QAExample) -> str:
	return (
		f"{format_mcq(example)}\n\n"
		"Answer with exactly one option letter (A-I) and nothing else."
	)


def cot_prompt(example: QAExample) -> str:
	return (
		f"{format_mcq(example)}\n\n"
		"Think step by step briefly, then end with: Final answer: <LETTER>."
	)


def self_revision_prompt(example: QAExample, prior_reasoning: str, prior_answer: str) -> str:
	return (
		f"{format_mcq(example)}\n\n"
		"You previously answered this question. Re-evaluate your reasoning carefully.\n"
		f"Previous reasoning:\n{prior_reasoning}\n\n"
		f"Previous answer: {prior_answer}\n\n"
		"If needed, revise. End with: Final answer: <LETTER>."
	)


def scaffold_prompt(example: QAExample, external_reasoning: str) -> str:
	return (
		f"{format_mcq(example)}\n\n"
		"Another model produced this reasoning:\n"
		f"{external_reasoning}\n\n"
		"Use it if helpful, then give your own final answer as exactly one option letter (A-I)."
	)


def parse_answer_letter(text: str) -> str | None:
	candidates = ANSWER_PATTERN.findall(text.upper())
	if not candidates:
		return None
	return candidates[-1]


def _resolve_torch_dtype(dtype: str) -> torch.dtype:
	if dtype == "float16":
		return torch.float16
	if dtype == "bfloat16":
		return torch.bfloat16
	return torch.float32


def _default_hf_module_path_from_hook_name(hook_name: str) -> str | None:
	match = re.search(r"blocks\.(\d+)\.", hook_name)
	if match is None:
		return None
	layer_idx = int(match.group(1))
	return f"model.layers.{layer_idx}"


def _resolve_hf_hook_module_path(sae: SAE[Any]) -> str | None:
	metadata = sae.cfg.metadata
	for attr_name in ("hf_hook_name", "hook_name_hf"):
		attr_value = getattr(metadata, attr_name, None)
		if isinstance(attr_value, str) and attr_value:
			if attr_value.endswith(".output"):
				return attr_value[: -len(".output")]
			return attr_value

	hook_name = metadata.hook_name
	if isinstance(hook_name, str):
		return _default_hf_module_path_from_hook_name(hook_name)
	return None


def _get_module_by_path(root_module: Any, module_path: str) -> Any | None:
	module: Any = root_module
	for piece in module_path.split("."):
		if piece.isdigit():
			if not hasattr(module, "__getitem__"):
				return None
			module = module[int(piece)]
		else:
			if not hasattr(module, piece):
				return None
			module = getattr(module, piece)
	return module


def _candidate_hf_module_paths(runtime: ModelRuntime) -> list[str]:
	paths: list[str] = []
	if runtime.hf_hook_module_path:
		paths.append(runtime.hf_hook_module_path)
		if runtime.hf_hook_module_path.endswith(".output"):
			paths.append(runtime.hf_hook_module_path[: -len(".output")])

	hook_name = runtime.sae.cfg.metadata.hook_name
	match = re.search(r"blocks\.(\d+)\.", hook_name)
	if match is not None:
		layer_idx = int(match.group(1))
		paths.extend(
			[
				f"model.layers.{layer_idx}",
				f"model.model.layers.{layer_idx}",
				f"base_model.model.layers.{layer_idx}",
			]
		)

	# dedupe while preserving order
	seen: set[str] = set()
	ordered: list[str] = []
	for path in paths:
		if path and path not in seen:
			ordered.append(path)
			seen.add(path)
	return ordered


def load_model_runtime(spec: ModelSpec) -> ModelRuntime:
	tokenizer = AutoTokenizer.from_pretrained(spec.hf_model_name, **spec.tokenizer_kwargs)
	if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
		tokenizer.pad_token = tokenizer.eos_token

	generation_model = AutoModelForCausalLM.from_pretrained(
		spec.hf_model_name,
		dtype=_resolve_torch_dtype(spec.dtype),
		**spec.hf_model_kwargs,
	).to(spec.device)
	generation_model.eval()

	cache_model: Any | None
	tlens_name = spec.tlens_model_name or spec.hf_model_name
	try:
		cache_model = HookedSAETransformer.from_pretrained_no_processing(
			tlens_name,
			hf_model=generation_model,
			device=spec.device,
		)
	except ValueError as hooked_error:
		# Fallback for environments where HookedTransformer path is unavailable
		# but TransformerLens v3 TransformerBridge is installed.
		if "gemma-3" in spec.hf_model_name.lower():
			try:
				from sae_lens.analysis.sae_transformer_bridge import SAETransformerBridge

				cache_model = SAETransformerBridge.boot_transformers(
					spec.hf_model_name,
					device=spec.device,
				)
			except Exception:
				cache_model = None
		else:
			raise ValueError(
				f"TransformerLens could not load cache model '{tlens_name}'. "
				"Set ModelSpec.tlens_model_name to a supported backbone and keep hf_model_name as your generation model."
			) from hooked_error

	sae = SAE.from_pretrained(
		release=spec.sae_release,
		sae_id=spec.sae_id,
		device=spec.device,
		dtype=spec.dtype,
	)
	sae.eval()
	hf_hook_module_path = _resolve_hf_hook_module_path(sae)
	if cache_model is None and hf_hook_module_path is None:
		raise ValueError(
			"Gemma-3 fallback requires a resolvable HF hook module path from SAE metadata. "
			"Try a different SAE id/release or update SAELens converters."
		)

	return ModelRuntime(
		spec=spec,
		generation_model=generation_model,
		generation_tokenizer=tokenizer,
		cache_model=cache_model,
		sae=sae,
		hf_hook_module_path=hf_hook_module_path,
	)


@torch.inference_mode()
def generate_text(
	runtime: ModelRuntime,
	prompt: str,
	decode_cfg: DecodeConfig,
) -> GenerationOutput:
	tokenizer = runtime.generation_tokenizer
	model = runtime.generation_model
	encoded = tokenizer(prompt, return_tensors="pt")
	input_ids = encoded["input_ids"].to(runtime.spec.device)
	attn_mask = encoded["attention_mask"].to(runtime.spec.device)

	generate_kwargs: dict[str, Any] = {
		"input_ids": input_ids,
		"attention_mask": attn_mask,
		"max_new_tokens": decode_cfg.max_new_tokens,
		"do_sample": decode_cfg.do_sample,
		"pad_token_id": tokenizer.pad_token_id,
		"eos_token_id": tokenizer.eos_token_id,
	}
	if decode_cfg.do_sample:
		generate_kwargs["temperature"] = decode_cfg.temperature
		generate_kwargs["top_p"] = decode_cfg.top_p

	output_ids = model.generate(**generate_kwargs)
	generated_ids = output_ids[0, input_ids.shape[1] :]
	first_token = int(generated_ids[0].item()) if generated_ids.numel() > 0 else None
	text = tokenizer.decode(generated_ids, skip_special_tokens=True)
	return GenerationOutput(
		text=text,
		first_generated_token_id=first_token,
		token_ids=generated_ids.detach().cpu().tolist(),
	)


@torch.inference_mode()
def extract_first_token_features(
	runtime: ModelRuntime,
	prompt: str,
	first_token_id: int,
) -> torch.Tensor:
	tokenizer = runtime.generation_tokenizer
	prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(runtime.spec.device)
	append_token = torch.tensor([[first_token_id]], device=runtime.spec.device, dtype=prompt_ids.dtype)
	full_ids = torch.cat([prompt_ids, append_token], dim=1)

	if runtime.cache_model is not None:
		hook_name = runtime.sae.cfg.metadata.hook_name
		stop_at_layer = extract_stop_at_layer_from_tlens_hook_name(hook_name)
		_, cache = runtime.cache_model.run_with_cache(
			full_ids,
			names_filter=[hook_name],
			stop_at_layer=stop_at_layer,
		)
		resid = cache[hook_name][0, prompt_ids.shape[1], :]
		sae_dtype = next(runtime.sae.parameters()).dtype
		encode_in = resid.unsqueeze(0).to(dtype=sae_dtype)
		features = runtime.sae.encode(encode_in).squeeze(0)
		features = torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)
		return features.detach().float().cpu()

	if runtime.hf_hook_module_path is None:
		raise ValueError("No HF hook module path available for fallback extraction.")

	target_module = None
	resolved_path = None
	for candidate_path in _candidate_hf_module_paths(runtime):
		target_module = _get_module_by_path(runtime.generation_model, candidate_path)
		if target_module is not None:
			resolved_path = candidate_path
			break

	if target_module is None:
		raise ValueError(
			"Could not resolve an HF hook module path for fallback extraction. "
			f"Tried: {_candidate_hf_module_paths(runtime)}"
		)

	captured: dict[str, torch.Tensor] = {}

	def _capture_hook(_module: Any, _inputs: Any, output: Any) -> Any:
		tensor_out = output[0] if isinstance(output, tuple) else output
		if isinstance(tensor_out, torch.Tensor):
			captured["activations"] = tensor_out
		return output

	handle = target_module.register_forward_hook(_capture_hook)
	try:
		runtime.generation_model(full_ids, use_cache=False)
	finally:
		handle.remove()

	if "activations" not in captured:
		raise RuntimeError(
			"Failed to capture activations from HF hook fallback path "
			f"'{resolved_path}'."
		)

	resid = captured["activations"][0, prompt_ids.shape[1], :]
	sae_dtype = next(runtime.sae.parameters()).dtype
	encode_in = resid.unsqueeze(0).to(dtype=sae_dtype)
	features = runtime.sae.encode(encode_in).squeeze(0)
	features = torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)
	return features.detach().float().cpu()


def run_single_condition(
	runtime: ModelRuntime,
	example: QAExample,
	prompt: str,
	decode_cfg: DecodeConfig,
	condition: str,
	reasoning_text: str | None = None,
) -> tuple[ExampleRunRecord, torch.Tensor | None]:
	generation = generate_text(runtime, prompt, decode_cfg)
	parsed = parse_answer_letter(generation.text)
	is_correct = None if parsed is None else parsed == example.answer_key

	feature_tensor: torch.Tensor | None = None
	if generation.first_generated_token_id is not None:
		feature_tensor = extract_first_token_features(
			runtime,
			prompt,
			generation.first_generated_token_id,
		)

	record = ExampleRunRecord(
		question_id=example.question_id,
		model_name=runtime.spec.name,
		condition=condition,
		answer_text=generation.text,
		parsed_answer=parsed,
		is_correct=is_correct,
		first_token_id=generation.first_generated_token_id,
		reasoning_text=reasoning_text,
	)
	return record, feature_tensor


def run_discovery(
	runtime: ModelRuntime,
	examples: list[QAExample],
	decode_cfg: DecodeConfig,
) -> DiscoveryArtifacts:
	direct_features: list[torch.Tensor] = []
	cot_features: list[torch.Tensor] = []
	direct_records: list[ExampleRunRecord] = []
	cot_records: list[ExampleRunRecord] = []

	for example in examples:
		direct_record, direct_feature = run_single_condition(
			runtime,
			example,
			direct_prompt(example),
			decode_cfg,
			condition="direct",
		)
		cot_record, cot_feature = run_single_condition(
			runtime,
			example,
			cot_prompt(example),
			decode_cfg,
			condition="cot",
		)

		direct_records.append(direct_record)
		cot_records.append(cot_record)

		if direct_feature is not None and cot_feature is not None:
			direct_features.append(direct_feature)
			cot_features.append(cot_feature)

	return DiscoveryArtifacts(
		direct_features=direct_features,
		cot_features=cot_features,
		direct_records=direct_records,
		cot_records=cot_records,
	)


def compute_delta(cot_features: list[torch.Tensor], direct_features: list[torch.Tensor]) -> torch.Tensor:
	if not cot_features or not direct_features:
		raise ValueError("Empty feature lists. Check generation and hook extraction.")
	cot_tensor = torch.stack(cot_features, dim=0)
	direct_tensor = torch.stack(direct_features, dim=0)
	return cot_tensor.mean(dim=0) - direct_tensor.mean(dim=0)


def topk_reasoning_features(delta: torch.Tensor, k: int = 10) -> list[dict[str, float | int]]:
	values, indices = torch.topk(delta.abs(), k=min(k, delta.shape[0]))
	output: list[dict[str, float | int]] = []
	for idx, value in zip(indices.tolist(), values.tolist(), strict=False):
		output.append(
			{
				"feature_index": int(idx),
				"abs_delta": float(value),
				"signed_delta": float(delta[idx].item()),
			}
		)
	return output


def paired_permutation_test(
	x: list[float],
	y: list[float],
	n_permutations: int = 5000,
	seed: int = 0,
) -> dict[str, float]:
	if len(x) != len(y):
		raise ValueError("x and y must be same length")
	if len(x) == 0:
		raise ValueError("x and y must be non-empty")

	diffs = [a - b for a, b in zip(x, y, strict=False)]
	observed = abs(sum(diffs) / len(diffs))

	random.seed(seed)
	hits = 0
	for _ in range(n_permutations):
		permuted = [d if random.random() < 0.5 else -d for d in diffs]
		if abs(sum(permuted) / len(permuted)) >= observed:
			hits += 1

	p_value = (hits + 1) / (n_permutations + 1)
	return {
		"mean_delta": float(sum(diffs) / len(diffs)),
		"p_value": float(p_value),
		"n": float(len(diffs)),
	}


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
	out = Path(path)
	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_jsonl(path: str | Path, records: list[ExampleRunRecord]) -> None:
	out = Path(path)
	out.parent.mkdir(parents=True, exist_ok=True)
	with out.open("w", encoding="utf-8") as handle:
		for record in records:
			handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def build_default_specs(device: str = "cpu") -> tuple[ModelSpec, ModelSpec]:
	deepseek = ModelSpec(
		name="deepseek_r1_distill_llama8b",
		hf_model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
		tlens_model_name="meta-llama/Llama-3.1-8B",
		sae_release="deepseek-r1-distill-llama-8b-qresearch",
		sae_id="blocks.19.hook_resid_post",
		device=device,
		dtype="float16" if device != "cpu" else "float32",
	)
	gemma = ModelSpec(
		name="gemma_3_4b_it",
		hf_model_name="google/gemma-3-4b-it",
		sae_release="gemma-scope-2-4b-it-res",
		sae_id="layer_17_width_16k_l0_medium",
		device=device,
		dtype="float16" if device != "cpu" else "float32",
	)
	return deepseek, gemma


def summarize_records(records: list[ExampleRunRecord]) -> dict[str, Any]:
	total = len(records)
	valid = [r for r in records if r.is_correct is not None]
	correct = [r for r in valid if r.is_correct]
	return {
		"total": total,
		"valid_answer_count": len(valid),
		"accuracy": float(len(correct) / len(valid)) if valid else math.nan,
	}

