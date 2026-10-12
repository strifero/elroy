"""Benchmark decontamination for the instruction set (Chapter 7).

An SFT example is dropped if it contains the text of a benchmark problem or a
large share of a benchmark's reference solution:

  HumanEval  the first 80 characters of the docstring, whitespace-normalized,
             appear anywhere in the example; or half or more of the canonical
             solution's 13-grams (identifiers, numbers, punctuation; comments
             and docstrings removed) appear in the example's code
  MBPP       the problem sentence appears verbatim, or half the reference
             solution's 13-grams do (all 427 sanitized problems, not just test)

These are the same tests scripts/check_contamination.py reports with, so the
filter removes exactly what the check counts.
"""
import re

W = 13


def _norm_code(code: str) -> list[str]:
    code = re.sub(r"#.*", "", code)
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    return re.findall(r"[A-Za-z_]\w*|\d+|[^\s\w]", code)


def _grams(toks: list[str], w: int = W) -> set:
    return {tuple(toks[i:i + w]) for i in range(max(0, len(toks) - w + 1))}


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


class Decontaminator:
    def __init__(self) -> None:
        from datasets import concatenate_datasets, load_dataset
        self.snips: dict[str, str] = {}
        self.sol: dict[str, set] = {}
        for r in load_dataset("openai/openai_humaneval", split="test"):
            m = re.search(r'"""([\s\S]*?)"""', r["prompt"])
            d = m.group(1) if m else r["prompt"]
            self.snips[r["task_id"]] = _flat(d.strip())[:80]
            self.sol[r["task_id"]] = _grams(_norm_code(r["canonical_solution"]))
        mb = load_dataset("google-research-datasets/mbpp", "sanitized")
        for r in concatenate_datasets([mb[s] for s in mb]):
            tid = f"MBPP/{r['task_id']}"
            self.snips[tid] = _flat(r["prompt"].strip())
            self.sol[tid] = _grams(_norm_code(r["code"]))
        self.snips = {k: v for k, v in self.snips.items() if len(v) > 40}
        self.sol = {k: v for k, v in self.sol.items() if len(v) >= 3}
        self.hits: dict[str, int] = {}

    def match(self, text: str) -> str | None:
        low = _flat(text)
        for tid, s in self.snips.items():
            if s in low:
                return tid
        blocks = re.findall(r"```(?:python|py)?\n([\s\S]*?)```", text) or [text]
        g = set()
        for b in blocks:
            g |= _grams(_norm_code(b))
        if g:
            for tid, sg in self.sol.items():
                if len(g & sg) >= max(3, len(sg) // 2):
                    return tid
        return None

    def keep(self, *texts: str) -> bool:
        tid = self.match("\n".join(texts))
        if tid:
            self.hits[tid] = self.hits.get(tid, 0) + 1
            return False
        return True
