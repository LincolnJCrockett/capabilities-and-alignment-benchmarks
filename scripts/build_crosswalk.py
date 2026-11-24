"""Build the curated model crosswalk: safety-source model names -> Epoch ECI model.

Inputs : data/raw/{irt,safetywashing,epoch}, data/interim/crosswalk_candidates.csv
Output : data/interim/model_crosswalk.csv

match_type
  exact    - same model and (as far as we can tell) same checkpoint/snapshot
  snapshot - same named model, but the safety source used an alias/preview/dated
             snapshot that may differ from the one(s) Epoch scored
  family   - different post-training variant of the same base (e.g. chat vs base,
             VL vs text, v0.2 vs v0.1); Epoch's ECI for that entry may pool variants
  none     - no ECI entry for this model

Edit MANUAL below to change a mapping, then re-run.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

# source_model -> (epoch_model or None, match_type, confidence, note)
E, S, F, N = "exact", "snapshot", "family", "none"
MANUAL = {
    # ---------------- IRT for AI Safety (OpenRouter slugs, collected ~mid-2026) -------------
    "amazon/nova-2-lite-v1": (None, N, "high", "Nova not in ECI; Nova Lite/Micro/Pro appear in some Epoch external benchmark files"),
    "amazon/nova-lite-v1": (None, N, "high", "see above"),
    "amazon/nova-micro-v1": (None, N, "high", "see above"),
    "anthropic/claude-3-haiku": ("Claude 3 Haiku", E, "high", ""),
    "anthropic/claude-3.5-haiku": ("Claude 3.5 Haiku", E, "high", ""),
    "anthropic/claude-haiku-4.5": ("Claude Haiku 4.5", E, "high", ""),
    "anthropic/claude-opus-4": ("Claude Opus 4", E, "high", ""),
    "anthropic/claude-sonnet-4": ("Claude Sonnet 4", E, "high", ""),
    "anthropic/claude-sonnet-4.5": ("Claude Sonnet 4.5", E, "high", ""),
    "arcee-ai/trinity-mini": (None, N, "high", ""),
    "bytedance-seed/seed-1.6": (None, N, "high", ""),
    "bytedance-seed/seed-1.6-flash": (None, N, "high", ""),
    "bytedance-seed/seed-2.0-lite": (None, N, "high", ""),
    "bytedance-seed/seed-2.0-mini": (None, N, "high", ""),
    "bytedance/ui-tars-1.5-7b": (None, N, "high", ""),
    "cohere/command-a": (None, N, "high", ""),
    "cohere/command-r-08-2024": (None, N, "high", ""),
    "cohere/command-r7b-12-2024": (None, N, "high", ""),
    "deepseek/deepseek-chat": ("DeepSeek-V3", S, "medium", "OpenRouter alias; pointed at DeepSeek-V3 (Dec 2024) - verify it wasn't routed to V3-0324"),
    "deepseek/deepseek-chat-v3": ("DeepSeek-V3", S, "medium", "Likely duplicate of deepseek/deepseek-chat; Epoch pools V3 and V3-Base"),
    "deepseek/deepseek-chat-v3-0324": ("DeepSeek-V3 (Mar 2025)", E, "high", ""),
    "deepseek/deepseek-chat-v3.1": ("DeepSeek-V3.1", E, "high", ""),
    "deepseek/deepseek-r1-distill-llama-70b": (None, N, "high", "Distill, not R1"),
    "deepseek/deepseek-r1-distill-qwen-32b": (None, N, "high", "Distill, not R1"),
    "deepseek/deepseek-v3.1-terminus": (None, N, "medium", "Update of V3.1; could proxy with DeepSeek-V3.1 (family)"),
    "deepseek/deepseek-v3.2": ("DeepSeek-V3.2", E, "high", ""),
    "deepseek/deepseek-v3.2-exp": ("DeepSeek-V3.2-Exp", E, "high", ""),
    "deepseek/deepseek-v4-flash": ("DeepSeek-V4-Flash", E, "high", "IRT data built 2026-07-27, before the 0731 Flash snapshot"),
    "deepseek/deepseek-v4-pro": ("DeepSeek-V4-Pro", E, "high", "IRT data built before the 0813 Pro snapshot"),
    "essentialai/rnj-1-instruct": (None, N, "high", ""),
    "google/gemini-2.0-flash-001": ("Gemini 2.0 Flash (Feb 2025)", E, "high", ""),
    "google/gemini-2.0-flash-lite-001": (None, N, "high", ""),
    "google/gemini-2.5-flash": ("Gemini 2.5 Flash (Jun 2025)", E, "high", "OpenRouter slug = GA release"),
    "google/gemini-2.5-flash-lite": ("Gemini 2.5 Flash-Lite (Jun 2025)", S, "medium", "OpenRouter = GA (Jul 2025); Epoch scored preview-06-17"),
    "google/gemini-2.5-flash-lite-preview-09-2025": (None, N, "medium", "Sep 2025 Flash-Lite preview not in ECI"),
    "google/gemini-2.5-pro": ("Gemini 2.5 Pro (Jun 2025)", E, "high", "OpenRouter slug = GA release"),
    "google/gemini-3-flash-preview": ("Gemini 3 Flash", E, "high", "Epoch also scored gemini-3-flash-preview"),
    "google/gemini-3.1-flash-lite": ("Gemini 3.1 Flash-Lite", E, "high", ""),
    "google/gemini-3.1-flash-lite-preview": ("Gemini 3.1 Flash-Lite", S, "medium", "Preview vs GA; duplicate of the GA row"),
    "google/gemma-2-27b-it": ("Gemma 2 27B", E, "high", "Epoch pools base + it"),
    "google/gemma-3-12b-it": (None, N, "high", ""),
    "google/gemma-3-27b-it": ("Gemma 3 27B", E, "high", ""),
    "google/gemma-3-4b-it": (None, N, "high", ""),
    "google/gemma-3n-e4b-it": (None, N, "high", ""),
    "google/gemma-4-26b-a4b-it": ("Gemma 4 26B A4B", E, "high", ""),
    "google/gemma-4-31b-it": ("Gemma 4 31B IT", E, "high", ""),
    "gryphe/mythomax-l2-13b": (None, N, "high", "Community Llama-2 merge"),
    "ibm-granite/granite-4.0-h-micro": (None, N, "high", ""),
    "ibm-granite/granite-4.1-8b": (None, N, "high", ""),
    "inclusionai/ling-2.6-1t": (None, N, "high", ""),
    "inclusionai/ling-2.6-flash": (None, N, "high", ""),
    "inflection/inflection-3-pi": (None, N, "high", ""),
    "inflection/inflection-3-productivity": (None, N, "high", ""),
    "kwaipilot/kat-coder-pro-v2": (None, N, "high", ""),
    "liquid/lfm-2-24b-a2b": (None, N, "high", ""),
    "liquid/lfm-2-24b-a2b-20260224": (None, N, "high", ""),
    "meta-llama/llama-3-70b-instruct": ("Llama 3-70B", E, "high", "Epoch pools base + instruct"),
    "meta-llama/llama-3-8b-instruct": ("Llama 3-8B", E, "high", "Epoch pools base + instruct"),
    "meta-llama/llama-3.1-70b-instruct": ("Llama 3.1-70B", E, "high", ""),
    "meta-llama/llama-3.1-8b-instruct": ("Llama 3.1-8B", E, "high", ""),
    "meta-llama/llama-3.2-11b-vision-instruct": (None, N, "high", "Only Llama 3.2 90B in ECI"),
    "meta-llama/llama-3.2-1b-instruct": (None, N, "high", ""),
    "meta-llama/llama-3.2-3b-instruct": (None, N, "high", ""),
    "meta-llama/llama-3.3-70b-instruct": ("Llama 3.3 70B", E, "high", ""),
    "meta-llama/llama-4-maverick": ("Llama 4 Maverick", E, "high", ""),
    "meta-llama/llama-4-scout": ("Llama 4 Scout", E, "high", ""),
    "meta-llama/llama-guard-4-12b": (None, N, "high", "Safety classifier, not a chat model - consider excluding from analysis"),
    "microsoft/phi-4": ("Phi-4", E, "high", ""),
    "microsoft/phi-4-mini-instruct": (None, N, "high", ""),
    "microsoft/wizardlm-2-8x22b": (None, N, "high", "Mixtral fine-tune"),
    "minimax/minimax-01": (None, N, "high", "Only MiniMax-M3 in ECI"),
    "minimax/minimax-m2": (None, N, "high", ""),
    "minimax/minimax-m2-her": (None, N, "high", ""),
    "minimax/minimax-m2.1": (None, N, "high", ""),
    "minimax/minimax-m2.5": (None, N, "high", ""),
    "minimax/minimax-m2.7": (None, N, "high", ""),
    "mistralai/codestral-2508": (None, N, "high", ""),
    "mistralai/devstral-2512": (None, N, "high", ""),
    "mistralai/devstral-small": (None, N, "high", ""),
    "mistralai/ministral-14b-2512": (None, N, "high", ""),
    "mistralai/ministral-3b-2512": (None, N, "high", ""),
    "mistralai/ministral-8b-2512": (None, N, "high", ""),
    "mistralai/mistral-7b-instruct-v0.1": ("Mistral 7B v0.1", F, "medium", "Epoch scored base Mistral-7B-v0.1 only"),
    "mistralai/mistral-large": ("Mistral Large 2 (Jul 2024)", S, "low", "OpenRouter alias has pointed at 2407/2411; ECI 'Mistral Large' is 2402. Verify"),
    "mistralai/mistral-large-2512": (None, N, "high", "Mistral Large 3 not in ECI (only Large 2, Large 4)"),
    "mistralai/mistral-medium-3": ("Mistral Medium 3", E, "high", ""),
    "mistralai/mistral-medium-3.1": (None, N, "medium", "Could proxy with Mistral Medium 3 (family)"),
    "mistralai/mistral-nemo": ("Mistral NeMo", E, "high", ""),
    "mistralai/mistral-saba": (None, N, "high", ""),
    "mistralai/mistral-small-24b-instruct-2501": (None, N, "high", ""),
    "mistralai/mistral-small-2603": (None, N, "high", ""),
    "mistralai/mistral-small-3.1-24b-instruct": (None, N, "high", ""),
    "mistralai/mistral-small-3.2-24b-instruct": (None, N, "high", ""),
    "mistralai/mixtral-8x22b-instruct": ("Mixtral 8x22B", E, "high", "Epoch pools base + instruct"),
    "mistralai/voxtral-small-24b-2507": (None, N, "high", ""),
    "moonshotai/kimi-k2": ("Kimi K2 (Jul 2025)", E, "high", ""),
    "moonshotai/kimi-k2-0905": (None, N, "medium", "Could proxy with Kimi K2 (Jul 2025) (snapshot)"),
    "moonshotai/kimi-k2.5": ("Kimi K2.5", E, "high", ""),
    "moonshotai/kimi-k2.6": ("Kimi K2.6", E, "high", ""),
    "nex-agi/deepseek-v3.1-nex-n1": (None, N, "high", "Third-party fine-tune"),
    "nousresearch/hermes-2-pro-llama-3-8b": (None, N, "high", "Fine-tune"),
    "nousresearch/hermes-3-llama-3.1-405b": (None, N, "high", "Fine-tune"),
    "nousresearch/hermes-3-llama-3.1-70b": (None, N, "high", "Fine-tune"),
    "nousresearch/hermes-4-405b": (None, N, "high", "Fine-tune"),
    "nousresearch/hermes-4-70b": (None, N, "high", "Fine-tune"),
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": (None, N, "high", ""),
    "nvidia/nemotron-3-nano-30b-a3b": (None, N, "high", "Only Nemotron 3 Ultra in ECI"),
    "nvidia/nemotron-3-super-120b-a12b": (None, N, "high", ""),
    "nvidia/nemotron-nano-9b-v2": (None, N, "high", ""),
    "openai/gpt-3.5-turbo": ("GPT-3.5 Turbo (Jan 2024)", S, "medium", "Alias -> gpt-3.5-turbo-0125"),
    "openai/gpt-3.5-turbo-0613": ("GPT-3.5 Turbo (Jun 2023)", E, "high", ""),
    "openai/gpt-3.5-turbo-16k": (None, N, "medium", "16k context variant; snapshot ambiguous"),
    "openai/gpt-4-turbo": ("GPT-4 Turbo (Apr 2024)", S, "medium", "Alias -> gpt-4-turbo-2024-04-09 (fuzzy match wrongly suggested Nov 2023)"),
    "openai/gpt-4.1": ("GPT-4.1", E, "high", ""),
    "openai/gpt-4.1-mini": ("GPT-4.1 mini", E, "high", ""),
    "openai/gpt-4.1-nano": ("GPT-4.1 nano", E, "high", ""),
    "openai/gpt-4o": ("GPT-4o (Aug 2024)", S, "low", "Alias; ECI has May/Aug/Nov 2024 snapshots. Verify which one OpenRouter served"),
    "openai/gpt-4o-mini": ("GPT-4o mini", E, "high", ""),
    "openai/gpt-4o-mini-2024-07-18": ("GPT-4o mini", E, "high", "Duplicate of openai/gpt-4o-mini"),
    "openai/gpt-5-mini": ("GPT-5 mini", E, "high", ""),
    "openai/gpt-5-nano": ("GPT-5 nano", E, "high", ""),
    "openai/gpt-5.4-nano": ("GPT-5.4 Nano", E, "high", ""),
    "openai/gpt-oss-120b": ("gpt-oss-120b", E, "high", ""),
    "openai/gpt-oss-20b": ("gpt-oss-20b", E, "high", ""),
    "openai/o3-mini": ("o3-mini", E, "high", "Reasoning effort may differ (OpenRouter default vs Epoch runs)"),
    "openai/o4-mini": ("o4-mini", E, "high", "Reasoning effort may differ"),
    "perceptron/perceptron-mk1": (None, N, "high", ""),
    "perplexity/sonar": (None, N, "high", ""),
    "prime-intellect/intellect-3": (None, N, "high", "ECI has INTELLECT-1 only"),
    "qwen/qwen-2.5-72b-instruct": ("Qwen2.5-72B", E, "high", "Epoch pools base, instruct, and VL-instruct"),
    "qwen/qwen-2.5-7b-instruct": (None, N, "high", ""),
    "qwen/qwen-2.5-coder-32b-instruct": ("Qwen2.5-Coder-32B", F, "medium", "Epoch scored the base Coder model"),
    "qwen/qwen-plus": (None, N, "medium", "Hosted alias; underlying model unclear"),
    "qwen/qwen-plus-2025-07-28": (None, N, "high", ""),
    "qwen/qwen2.5-vl-72b-instruct": ("Qwen2.5-72B", F, "medium", "VL variant; Epoch's Qwen2.5-72B pools in some VL-72B results"),
    "qwen/qwen3-14b": ("Qwen3-14B", E, "high", "Hybrid-thinking model; thinking mode may differ"),
    "qwen/qwen3-235b-a22b": ("Qwen3-235B-A22B", E, "high", "Hybrid-thinking model; thinking mode may differ"),
    "qwen/qwen3-235b-a22b-2507": ("Qwen3-235B-A22B-Instruct (Jul 2025)", E, "high", ""),
    "qwen/qwen3-30b-a3b": ("Qwen3-30B-A3B", E, "high", "Hybrid-thinking model"),
    "qwen/qwen3-30b-a3b-instruct-2507": ("Qwen3-30B-A3B-Instruct (Jul 2025)", E, "high", ""),
    "qwen/qwen3-30b-a3b-thinking-2507": ("Qwen3-30B-A3B-Thinking (Jul 2025)", E, "high", ""),
    "qwen/qwen3-32b": ("Qwen3-32B", E, "high", "Hybrid-thinking model"),
    "qwen/qwen3-8b": ("Qwen3-8B", E, "high", "Hybrid-thinking model"),
    "qwen/qwen3-coder": (None, N, "high", ""),
    "qwen/qwen3-coder-30b-a3b-instruct": (None, N, "high", ""),
    "qwen/qwen3-coder-flash": (None, N, "high", ""),
    "qwen/qwen3-coder-next": (None, N, "high", ""),
    "qwen/qwen3-coder-plus": (None, N, "high", ""),
    "qwen/qwen3-next-80b-a3b-instruct": (None, N, "high", ""),
    "qwen/qwen3-vl-235b-a22b-instruct": (None, N, "high", "VL model; could proxy with Qwen3-235B-A22B (family)"),
    "qwen/qwen3-vl-30b-a3b-instruct": (None, N, "high", "VL model"),
    "qwen/qwen3-vl-32b-instruct": (None, N, "high", "VL model"),
    "qwen/qwen3-vl-8b-instruct": (None, N, "high", "VL model"),
    "qwen/qwen3.5-122b-a10b": (None, N, "high", ""),
    "qwen/qwen3.5-27b": (None, N, "high", "ECI has Qwen3.6 27B / 3.8 27B, not 3.5 27B"),
    "qwen/qwen3.5-35b-a3b": ("Qwen3.5-35B-A3B", E, "high", ""),
    "qwen/qwen3.5-397b-a17b": ("Qwen3.5 397B-A17B", E, "high", ""),
    "qwen/qwen3.5-9b": ("Qwen3.5-9B", E, "high", ""),
    "qwen/qwen3.5-flash-02-23": ("Qwen 3.5 Flash (hosted 35B-A3B)", S, "medium", "Dated hosted snapshot; Epoch scored undated qwen3.5-flash"),
    "qwen/qwen3.5-plus-02-15": ("Qwen 3.5 Plus (hosted 397B-A17B)", S, "medium", "Dated hosted snapshot; Epoch scored undated qwen3.5-plus"),
    "qwen/qwen3.5-plus-20260420": ("Qwen 3.5 Plus (hosted 397B-A17B)", S, "low", "Later hosted snapshot (Apr 2026); may differ from what Epoch scored"),
    "qwen/qwen3.6-27b": ("Qwen3.6 27B", E, "high", ""),
    "qwen/qwen3.6-35b-a3b": ("Qwen 3.6 35B-A3B", E, "high", ""),
    "qwen/qwen3.6-flash": ("Qwen 3.6 Flash", E, "high", ""),
    "qwen/qwen3.6-plus": ("Qwen 3.6 Plus", E, "high", ""),
    "rekaai/reka-edge": (None, N, "high", ""),
    "rekaai/reka-flash-3": (None, N, "high", ""),
    "sao10k/l3-lunaris-8b": (None, N, "high", "Community fine-tune"),
    "stepfun/step-3.5-flash": (None, N, "high", ""),
    "stepfun/step-3.7-flash": (None, N, "high", ""),
    "tencent/hunyuan-a13b-instruct": (None, N, "high", ""),
    "tencent/hy3-preview": (None, N, "high", ""),
    "thedrummer/rocinante-12b": (None, N, "high", "Community fine-tune"),
    "upstage/solar-pro-3": (None, N, "high", ""),
    "x-ai/grok-4.20": ("Grok 4.20", E, "high", ""),
    "x-ai/grok-4.3": ("Grok 4.3 Beta", S, "medium", "Epoch labels it Beta; same grok-4.3 id"),
    "x-ai/grok-build-0.1": (None, N, "high", ""),
    "xiaomi/mimo-v2-flash": (None, N, "high", ""),
    "xiaomi/mimo-v2.5": (None, N, "high", ""),
    "xiaomi/mimo-v2.5-pro": (None, N, "high", ""),
    "z-ai/glm-4-32b": (None, N, "high", ""),
    "z-ai/glm-4.5": (None, N, "high", ""),
    "z-ai/glm-4.5-air": (None, N, "high", ""),
    "z-ai/glm-4.5v": (None, N, "high", ""),
    "z-ai/glm-4.6": (None, N, "high", ""),
    "z-ai/glm-4.6v": (None, N, "high", ""),
    "z-ai/glm-4.7": ("GLM-4.7", E, "high", ""),
    "z-ai/glm-4.7-flash": (None, N, "high", ""),
    "z-ai/glm-5": ("GLM-5", E, "high", ""),
    "z-ai/glm-5-turbo": (None, N, "high", ""),
    "z-ai/glm-5.1": ("GLM-5.1", E, "high", ""),
    "z-ai/glm-5v-turbo": (None, N, "high", ""),
    # ---------------- Safetywashing (HF repo names, 2024) ----------------
    "Llama-2-13b-chat-hf": ("Llama 2-13B", E, "high", "Epoch pools base + chat"),
    "Llama-2-70b-chat-hf": ("Llama 2-70B", E, "high", "Epoch pools base + chat"),
    "Llama-2-7b-chat-hf": ("Llama 2-7B", F, "medium", "Epoch scored base only"),
    "Meta-Llama-3-70B-Instruct": ("Llama 3-70B", E, "high", "Epoch pools base + instruct"),
    "Meta-Llama-3-8B-Instruct": ("Llama 3-8B", E, "high", "Epoch pools base + instruct"),
    "Mistral-7B-Instruct-v0.2": ("Mistral 7B v0.1", F, "low", "Different version (v0.2 instruct vs v0.1 base)"),
    "Mixtral-8x22B-Instruct-v0.1": ("Mixtral 8x22B", E, "high", ""),
    "Mixtral-8x7B-Instruct-v0.1": ("Mixtral 8x7B", E, "high", ""),
    "Qwen1.5-0.5B-Chat": (None, N, "high", "No Qwen1.5 in ECI"),
    "Qwen1.5-1.8B-Chat": (None, N, "high", "No Qwen1.5 in ECI (Qwen-1_8B is Qwen 1)"),
    "Qwen1.5-110B-Chat": (None, N, "high", ""),
    "Qwen1.5-14B-Chat": (None, N, "high", "Qwen-14B is Qwen 1"),
    "Qwen1.5-32B-Chat": (None, N, "high", ""),
    "Qwen1.5-4B-Chat": (None, N, "high", ""),
    "Qwen1.5-72B-Chat": (None, N, "high", ""),
    "Qwen1.5-7B-Chat": (None, N, "high", "Qwen-7B is Qwen 1"),
    "Yi-34B-Chat": (None, N, "high", ""),
    "Yi-6B-Chat": ("Yi 6B", E, "high", "Epoch pools base + chat"),
    "dbrx-instruct": (None, N, "high", ""),
    "deepseek-llm-67b-chat": (None, N, "high", ""),
    "deepseek-llm-7b-chat": (None, N, "high", ""),
    "falcon-180B-chat": ("Falcon-180B", F, "medium", "Epoch scored base only"),
    "falcon-40b-instruct": ("Falcon-40B", E, "high", "Epoch pools base + instruct"),
    "falcon-7b-instruct": ("Falcon-7B", F, "medium", "Epoch scored base only"),
    "gemma-1.1-2b-it": ("Gemma 2B", F, "low", "Gemma 1.1 instruct vs Gemma 1.0 base"),
    "gemma-1.1-7b-it": ("Gemma 7B", F, "low", "Gemma 1.1 instruct vs Gemma 1.0 base"),
    "Mixtral-8x22B-v0.1": ("Mixtral 8x22B", E, "high", ""),
    "Mistral-7B-v0.1": ("Mistral 7B v0.1", E, "high", ""),
    "Llama-2-13b-hf": ("Llama 2-13B", E, "high", ""),
    "Qwen1.5-72B": (None, N, "high", ""),
    "Qwen1.5-14B": (None, N, "high", ""),
    "Yi-9B": ("Yi-9B", E, "high", ""),
    "falcon-40b": ("Falcon-40B", E, "high", ""),
    "Meta-Llama-3-70B": ("Llama 3-70B", E, "high", ""),
    "falcon-7b": ("Falcon-7B", E, "high", ""),
    "Meta-Llama-3-8B": ("Llama 3-8B", E, "high", ""),
    "Qwen1.5-0.5B": (None, N, "high", ""),
    "dbrx-base": (None, N, "high", ""),
    "deepseek-llm-67b-base": (None, N, "high", ""),
    "Qwen1.5-1.8B": (None, N, "high", ""),
    "Yi-6B": ("Yi 6B", E, "high", ""),
    "Yi-34B": (None, N, "high", ""),
    "gemma-2b": ("Gemma 2B", E, "high", ""),
    "gemma-7b": ("Gemma 7B", E, "high", ""),
    "Llama-2-70b-hf": ("Llama 2-70B", E, "high", ""),
    "deepseek-llm-7b-base": (None, N, "high", ""),
    "Llama-2-7b-hf": ("Llama 2-7B", E, "high", ""),
    "falcon-180B": ("Falcon-180B", E, "high", ""),
    "Qwen1.5-4B": (None, N, "high", ""),
    "Qwen1.5-110B": (None, N, "high", ""),
    "Mixtral-8x7B-v0.1": ("Mixtral 8x7B", E, "high", ""),
    "Qwen1.5-7B": (None, N, "high", ""),
    "Qwen1.5-32B": (None, N, "high", ""),
}


def main():
    cand = pd.read_csv(INTERIM / "crosswalk_candidates.csv")
    eci = pd.read_csv(RAW / "epoch/extracted/epoch_capabilities_index/eci_scores.csv")
    proc = pd.read_csv(RAW / "epoch/extracted/epoch_capabilities_index/processed_data_for_eci.csv")
    nbench = proc.groupby("Model")["benchmark"].nunique().rename("eci_n_benchmarks")
    eci = eci.set_index("Model").join(nbench)

    missing = set(cand["source_model"]) - set(MANUAL)
    if missing:
        raise SystemExit(f"No manual entry for: {sorted(missing)}")
    bad = {v[0] for v in MANUAL.values() if v[0]} - set(eci.index)
    if bad:
        raise SystemExit(f"Epoch names not in ECI: {sorted(bad)}")

    rows = []
    for r in cand.itertuples(index=False):
        em, mtype, conf, note = MANUAL[r.source_model]
        e = eci.loc[em] if em else None
        rows.append({
            "source": r.source,
            "source_model": r.source_model,
            "epoch_model": em,
            "match_type": mtype,
            "confidence": conf,
            "needs_review": mtype in (S, F) or conf != "high",
            "note": note,
            "eci": None if e is None else e["eci"],
            "eci_ci_low": None if e is None else e["eci_ci_low"],
            "eci_ci_high": None if e is None else e["eci_ci_high"],
            "eci_n_benchmarks": None if e is None else e["eci_n_benchmarks"],
            "release_date": None if e is None else e["date"],
            "organization": None if e is None else e["Organization"],
            "fuzzy_top1": r.cand1,
            "fuzzy_top2": r.cand2,
        })
    out = pd.DataFrame(rows)
    # Merge in the user's manual review (hand-maintained; never written by any script).
    review_path = ROOT / "data" / "manual" / "crosswalk_review.csv"
    if review_path.exists():
        rev = pd.read_csv(review_path)[["source", "source_model", "review_output"]]
        out = out.merge(rev, on=["source", "source_model"], how="left")
        cols = list(out.columns)
        cols.insert(cols.index("confidence") + 1, cols.pop(cols.index("review_output")))
        out = out[cols]
    out.to_csv(INTERIM / "model_crosswalk.csv", index=False)

    print(out.groupby(["source", "match_type"]).size().unstack(fill_value=0))
    print("\nneeds_review:", int(out["needs_review"].sum()))
    m = out[out.epoch_model.notna()]
    print("distinct ECI models matched per source:")
    print(m.groupby("source")["epoch_model"].nunique())


if __name__ == "__main__":
    main()
