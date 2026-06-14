# metric.py

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime

METRICS_PATH = "results/metrics.jsonl"

INPUT_COST_PER_MILLION = 0.50
OUTPUT_COST_PER_MILLION = 1.50

os.makedirs("results", exist_ok=True)


def calculate_llm_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculates estimated LLM cost using provider pricing.

    Input price:  $0.50 per 1M tokens
    Output price: $1.50 per 1M tokens
    """
    input_cost = (prompt_tokens / 1_000_000) * INPUT_COST_PER_MILLION
    output_cost = (completion_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION

    return round(input_cost + output_cost, 8)


def init_metrics(report_id: str, patient_id: str, top_k: int) -> dict:
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "report_id": report_id,
        "patient_id": patient_id,
        "top_k": top_k,

        # Latency metrics
        "total_latency_sec": 0.0,
        "retrieval_latency_sec": 0.0,
        "summary_latency_sec": 0.0,
        "routing_latency_sec": 0.0,
        "specialist_agents_latency_sec": 0.0,
        "final_summary_latency_sec": 0.0,
        "upsert_latency_sec": 0.0,

        # Agent metrics
        "agents_activated_count": 0,
        "activated_agents": [],

        # Cost/token metrics
        "successful_llm_requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_cost_usd": 0.0,

        # Custom production metric
        "cost_latency_score": 0.0,

        # Status
        "status": "success",
        "error": "",
    }


@contextmanager
def timer(metrics: dict, key: str):
    start = time.perf_counter()

    try:
        yield
    finally:
        metrics[f"{key}_latency_sec"] = round(time.perf_counter() - start, 4)


def save_metrics(metrics: dict):
    os.makedirs("results", exist_ok=True)

    with open(METRICS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def format_metrics(metrics: dict) -> str:
    return f"""
| Metric | Value |
|---|---:|
| Total Latency | {metrics["total_latency_sec"]} sec |
| Retrieval Latency | {metrics["retrieval_latency_sec"]} sec |
| Summary Latency | {metrics["summary_latency_sec"]} sec |
| Routing Latency | {metrics["routing_latency_sec"]} sec |
| Specialist Agents Latency | {metrics["specialist_agents_latency_sec"]} sec |
| Final Summary Latency | {metrics["final_summary_latency_sec"]} sec |
| Upsert Latency | {metrics["upsert_latency_sec"]} sec |
| Prompt Tokens | {metrics["prompt_tokens"]} |
| Completion Tokens | {metrics["completion_tokens"]} |
| Total Tokens | {metrics["total_tokens"]} |
| Input Cost Rate | ${INPUT_COST_PER_MILLION} / 1M tokens |
| Output Cost Rate | ${OUTPUT_COST_PER_MILLION} / 1M tokens |
| LLM Cost | ${metrics["llm_cost_usd"]} |
| Cost-Latency Score | {metrics["cost_latency_score"]} |
"""