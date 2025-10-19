"""
Robustness Evaluators
Đánh giá độ ổn định khi input thay đổi
"""

from langsmith.schemas import Run, Example
import random
import string


def introduce_typos(text: str, typo_rate: float = 0.1) -> str:
    """Introduce random typos into text"""
    words = text.split()
    num_typos = max(1, int(len(words) * typo_rate))
    
    for _ in range(num_typos):
        if not words:
            break
        word_idx = random.randint(0, len(words) - 1)
        word = words[word_idx]
        
        if len(word) > 3:
            # Random typo: swap adjacent chars or add extra char
            char_idx = random.randint(1, len(word) - 2)
            typo_word = list(word)
            
            if random.random() < 0.5:
                # Swap adjacent characters
                typo_word[char_idx], typo_word[char_idx + 1] = typo_word[char_idx + 1], typo_word[char_idx]
            else:
                # Double a character
                typo_word.insert(char_idx, typo_word[char_idx])
            
            words[word_idx] = ''.join(typo_word)
    
    return ' '.join(words)


def paraphrase_query(query: str) -> str:
    """Simple paraphrasing (rule-based)"""
    paraphrases = {
        "What is": "Could you tell me",
        "Give me": "Please provide",
        "Find": "Search for",
        "Show me": "Display",
        "How many": "What's the number of",
    }
    
    for original, replacement in paraphrases.items():
        if original in query:
            return query.replace(original, replacement, 1)
    
    return query


def robustness_typo_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator: Robustness to Typos
    
    Compare performance on original query vs typo-injected query.
    Note: This requires running agent twice (once with typo).
    For now, we just check if agent handled current query well.
    """
    # This is a placeholder - full implementation needs variant testing
    # See Section 5.3 in your behavior_golden20.jsonl for examples
    
    answer = run.outputs.get("answer", "") if run.outputs else ""
    success = run.outputs.get("success", False) if run.outputs else False
    
    # If agent succeeded despite potential typos in query
    score = 1.0 if success else 0.0
    
    return {
        "key": "robustness_typo_handling",
        "score": score,
        "comment": f"Agent {'succeeded' if success else 'failed'} on potentially noisy input"
    }


def error_recovery_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluator: Error Recovery
    
    Checks if agent recovers gracefully from tool failures.
    """
    tool_calls = run.outputs.get("tool_calls", []) if run.outputs else []
    tool_params = run.outputs.get("tool_params", []) if run.outputs else []
    answer = run.outputs.get("answer", "") if run.outputs else ""
    
    # Check for error handling patterns
    error_indicators = ["error", "failed", "could not", "unable to"]
    has_errors = any(indicator in answer.lower() for indicator in error_indicators)
    
    if not has_errors:
        # No errors encountered
        return {
            "key": "error_recovery",
            "score": 1.0,
            "comment": "No errors encountered"
        }