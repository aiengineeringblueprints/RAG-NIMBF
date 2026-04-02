# Local Models for RAGAS LLM-as-Judge

## Tier 1 — Best Quality (needs 16-24GB VRAM, quantized)

| Model | Size | Why Good for RAGAS |
|-------|------|--------------------|
| **Qwen 2.5 32B** (Q4_K_M) | ~20GB VRAM | Top-tier reasoning, excellent structured output. One of the strongest open models for evaluation tasks. |
| **Qwen 3 32B** (Q4_K_M) | ~20GB VRAM | Successor with even better reasoning and hybrid thinking mode. |
| **Command R 35B** (Q4_K_M) | ~22GB VRAM | Purpose-built for RAG tasks — naturally strong at faithfulness/context assessment. |
| **Llama 3.1 70B** (Q3_K_M) | ~36GB VRAM | Near GPT-4 quality for judgment, but needs significant VRAM or offloading. |

## Tier 2 — Good Balance (8-16GB VRAM)

| Model | Size | Why Good for RAGAS |
|-------|------|--------------------|
| **Qwen 2.5 14B** | ~9GB VRAM | Punches well above its weight. Strong instruction following. |
| **Qwen 3 14B** | ~9GB VRAM | Newer, with thinking mode that helps nuanced judgments. |
| **Gemma 3 12B** | ~8GB VRAM | Decent but not the strongest for structured scoring tasks. |
| **Mistral Small 3.1 24B** (Q4) | ~14GB VRAM | Strong reasoning-per-parameter ratio. |
| **Phi-4 14B** | ~9GB VRAM | Excellent reasoning for its size, good at following evaluation rubrics. |

## Tier 3 — Budget/Speed (4-8GB VRAM)

| Model | Size | Notes |
|-------|------|-------|
| **Qwen 2.5 7B** | ~5GB | Surprisingly capable for small model. |
| **Gemma 3 4B** | ~3GB | OK but will have lower correlation with human judgments. |
| **Phi-4 Mini 3.8B** | ~3GB | Fast but limited nuanced judgment ability. |

## Recommendations

### Best pick for single-GPU (8-16GB VRAM)
`qwen3:14b` or `qwen2.5:14b` via Ollama — significantly stronger than `gemma3:12b` for structured evaluation, and fits in the same VRAM budget.

### Best pick with 24GB VRAM
`qwen3:32b` (Q4_K_M) is the best local judge available right now. It approaches GPT-4-level quality on RAGAS metrics.

### Why Qwen excels for RAGAS
RAGAS judge tasks require extracting claims from text, verifying them against context, and scoring on rubrics. Qwen models are particularly strong at this kind of structured analytical reasoning and instruction following, consistently outperforming similarly-sized Llama and Gemma variants on these tasks.

## Usage Examples

### Via Ollama (.env)
```
EVAL_CRITIC_LLM=qwen3:14b
```

### Via vLLM / OpenAI-compatible
```
EVAL_CRITIC_LLM=openai:Qwen/Qwen3-14B
```
