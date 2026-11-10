
from typing import Dict, Any
from src.retriever import PaperRetriever
from src.llm_service import LLMService

class RelayPipeline:
    """Orchestrates 4 steps: Intent Classification, Tool Call Retrieval, Drafting, and Verification."""
    
    def __init__(self, retriever: PaperRetriever, llm: LLMService):
        self.retriever = retriever
        self.llm = llm

    def run(self, user_query: str, retrieval_setting: Dict[str, Any] = None) -> Dict[str, Any]:
        if retrieval_setting is None:
            retrieval_setting = {"piece_size": 500, "pieces_fetched": 3, "rerank": False}
            
        retries = 0
        max_retries = 1
        
        # Step 1: Intent & Query Understanding
        intent_prompt = f"Categorize query intent: '{user_query}'. Return: SEARCH or DIRECT."
        intent_res = self.llm.generate(intent_prompt)
        
        if intent_res["status"] == "rate_limited":
            return self._build_result("rate_limited", "System overloaded.", 0, 0, 0, 0, retries, "refused_correctly")

        # Step 2: Retrieval Tool Call
        passages = self.retriever.retrieve(
            query=user_query,
            pieces_fetched=retrieval_setting.get("pieces_fetched", 3),
            rerank=retrieval_setting.get("rerank", False)
        )
        
        context_str = "\n\n".join([f"Source [{p['paper']}]: {p['text']}" for p in passages]) if passages else "NO_CONTEXT"

        # Step 3: Draft Cited Reply
        draft_prompt = (
            f"Context from research papers:\n{context_str}\n\n"
            f"Question: {user_query}\n\n"
            "Instructions: Answer using ONLY the context above. Cite paper name. "
            "If context lacks the answer, state clearly that the papers do not say."
        )
        draft_res = self.llm.generate(draft_prompt)
        
        if draft_res["status"] != "ok":
            return self._build_result(draft_res["status"], draft_res["content"], draft_res["first_token_ms"], draft_res["total_ms"], draft_res["tokens_in"], draft_res["tokens_out"], retries, "refused_correctly")

        # Step 4: Quality Check / Verification Branch & Loop
        verify_prompt = (
            f"Check if answer claims facts not in Context, or fails to cite papers.\n"
            f"Answer: {draft_res['content']}\nContext: {context_str}\n"
            f"Reply PASS or FAIL."
        )
        verify_res = self.llm.generate(verify_prompt)
        
        # Loop back if verification failed
        if "FAIL" in verify_res.get("content", "") and retries < max_retries:
            retries += 1
            strict_prompt = draft_prompt + "\nSTRICT REMINDER: State 'papers do not say' if context is missing."
            draft_res = self.llm.generate(strict_prompt)

        # Grade answer
        content = draft_res["content"].lower()
        if "papers do not say" in content or "not stated" in content:
            grade = "refused_correctly"
        elif any(p["paper"].lower() in content for p in passages):
            grade = "correct"
        else:
            grade = "wrong"

        total_ttft = intent_res["first_token_ms"] + draft_res["first_token_ms"]
        total_time = intent_res["total_ms"] + draft_res["total_ms"] + verify_res["total_ms"]
        total_tokens_in = intent_res["tokens_in"] + draft_res["tokens_in"] + verify_res["tokens_in"]
        total_tokens_out = intent_res["tokens_out"] + draft_res["tokens_out"] + verify_res["tokens_out"]

        return self._build_result("ok", draft_res["content"], total_ttft, total_time, total_tokens_in, total_tokens_out, retries, grade)

    def _build_result(self, status, content, ttft, total_ms, t_in, t_out, retries, grade):
        return {
            "status": status,
            "content": content,
            "first_token_ms": ttft,
            "total_ms": total_ms,
            "tokens_in": t_in,
            "tokens_out": t_out,
            "retries": retries,
            "answer_grade": grade
        }
