
import time
import os
import csv
import uuid
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from src.pipeline import RelayPipeline

class LoadTester:
    """Executes benchmark scenarios and outputs raw metrics to data/results.csv."""
    
    def __init__(self, pipeline: RelayPipeline, output_csv: str = "data/results.csv"):
        self.pipeline = pipeline
        self.output_csv = output_csv
        self._init_csv()

    def _init_csv(self):
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        if not os.path.exists(self.output_csv):
            with open(self.output_csv, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "request_id", "scenario", "source", "start_time", 
                    "first_token_ms", "total_ms", "status", "retries", 
                    "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"
                ])

    def run_scenario(self, scenario_name: str, num_requests: int, delay_between: float, source: str = "stand-in") -> pd.DataFrame:
        queries = [
            "What is the multi-head attention mechanism in Transformer?",
            "How does Moshi achieve real-time speech dialogue latency?",
            "What is dynamic chunking in sequence modeling?",
            "What is the salary of the authors of Attention Is All You Need?", # Should refuse
            "How does positional encoding work?"
        ]
        
        results = []
        for i in range(num_requests):
            req_id = str(uuid.uuid4())[:8]
            query = queries[i % len(queries)]
            start_time = time.time()
            
            res = self.pipeline.run(query)
            
            row = {
                "request_id": req_id,
                "scenario": scenario_name,
                "source": source,
                "start_time": round(start_time, 3),
                "first_token_ms": res["first_token_ms"],
                "total_ms": res["total_ms"],
                "status": res["status"],
                "retries": res["retries"],
                "tokens_in": res["tokens_in"],
                "tokens_out": res["tokens_out"],
                "retrieval_setting": "500_3_norerank",
                "answer_grade": res["answer_grade"]
            }
            
            with open(self.output_csv, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(list(row.values()))
                
            results.append(row)
            time.sleep(delay_between)
            
        return pd.DataFrame(results)
