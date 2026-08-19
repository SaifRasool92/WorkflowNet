import time
import numpy as np
from typing import Dict, List, Callable

class WorkflowPerformanceTracker:
    """Tracks latency, throughput, and precision metrics across matching pipelines."""

    @staticmethod
    def measure_latency_and_throughput(matching_fn: Callable, candidate_skills: List[str], n_runs: int = 50) -> Dict[str, float]:
        """Measures microsecond latency and throughput (items/sec) over N executions."""
        latencies = []
        
        # Warmup run
        matching_fn(candidate_skills)
        
        for _ in range(n_runs):
            start = time.perf_counter()
            res = matching_fn(candidate_skills)
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0) # ms

        avg_latency_ms = float(np.mean(latencies))
        p95_latency_ms = float(np.percentile(latencies, 95))
        throughput_qps = float(1000.0 / avg_latency_ms) if avg_latency_ms > 0 else 0.0

        return {
            'avg_latency_ms': avg_latency_ms,
            'p95_latency_ms': p95_latency_ms,
            'throughput_ops_sec': throughput_qps
        }

    @staticmethod
    def compute_precision_at_k(recommended_ids: List[str], ground_truth_ids: List[str]) -> float:
        """Computes Precision@K score against hand-labeled ground truth matches."""
        if not recommended_ids or not ground_truth_ids:
            return 0.0
        rec_set = set(recommended_ids)
        gt_set = set(ground_truth_ids)
        return len(rec_set.intersection(gt_set)) / len(rec_set)