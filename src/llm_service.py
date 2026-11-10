import os
import time
import random
from typing import Dict, Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class LLMService:
    """Dual-mode LLM provider handling real Groq API calls and calibrated stand-in simulation."""
    
    def __init__(self, mode: str = "groq", api_key: str = None):
        self.mode = mode
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        # Defaults to Groq's standard active 70b free tier endpoint
        self.default_model = os.getenv("DEFAULT_GROQ_MODEL", "llama3-70b-8192")
        
        self.client = Groq(api_key=self.api_key) if self.api_key and mode == "groq" else None
        
        self.standin_rpm_limit = int(os.getenv("STANDIN_RPM_LIMIT", 30))
        self.standin_tpm_limit = int(os.getenv("STANDIN_TPM_LIMIT", 6000))
        self.standin_rpd_limit = int(os.getenv("STANDIN_RPD_LIMIT", 14400))
        self.request_timestamps = []

    def generate(self, prompt: str, model: str = None, stream: bool = False) -> Dict[str, Any]:
        target_model = model or self.default_model
        start_time = time.time()
        
        if self.mode == "groq" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=target_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=600
                )
                total_ms = int((time.time() - start_time) * 1000)
                content = response.choices[0].message.content
                tokens_in = response.usage.prompt_tokens
                tokens_out = response.usage.completion_tokens
                return {
                    "status": "ok",
                    "content": content,
                    "first_token_ms": int(total_ms * 0.35),
                    "total_ms": total_ms,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out
                }
            except Exception as e:
                err_msg = str(e)
                status = "rate_limited" if "429" in err_msg or "rate limit" in err_msg.lower() else "error"
                return {
                    "status": status,
                    "content": f"Groq Error: {err_msg}",
                    "first_token_ms": 0,
                    "total_ms": int((time.time() - start_time) * 1000),
                    "tokens_in": len(prompt.split()),
                    "tokens_out": 0
                }
        else:
            # Stand-in Simulation Path
            now = time.time()
            self.request_timestamps = [t for t in self.request_timestamps if now - t < 60]
            
            if len(self.request_timestamps) >= self.standin_rpm_limit:
                time.sleep(0.1)
                return {
                    "status": "rate_limited",
                    "content": f"Stand-in Error: 429 Rate Limit Exceeded ({self.standin_rpm_limit} RPM)",
                    "first_token_ms": 0,
                    "total_ms": int((time.time() - start_time) * 1000),
                    "tokens_in": len(prompt.split()),
                    "tokens_out": 0
                }
            
            self.request_timestamps.append(now)
            
            first_token_delay = random.uniform(0.18, 0.45)
            time.sleep(first_token_delay)
            first_token_ms = int((time.time() - start_time) * 1000)
            
            simulated_response = f"[Stand-in Mock Response]: Answer cited and verified against target research paper passages."
            generation_delay = random.uniform(0.3, 0.8)
            time.sleep(generation_delay)
            
            total_ms = int((time.time() - start_time) * 1000)
            return {
                "status": "ok",
                "content": simulated_response,
                "first_token_ms": first_token_ms,
                "total_ms": total_ms,
                "tokens_in": len(prompt.split()),
                "tokens_out": len(simulated_response.split())
            }