"""Chapter 7 follow-up: is the SFT data contaminated with HumanEval, and do the chat
model's new passes look memorized?

    python scripts/check_contamination.py

Three checks, cheapest first:
  1. Diff every chat-model HumanEval pass against the canonical solution (difflib ratio
     and longest common line run). A memorized answer is near-identical; a re-derivation
     is not.
  2. Search the two instruction datasets for HumanEval prompt text (the docstring, which
     is the most distinctive part) and for canonical solutions, by 13-gram overlap on
     normalized code.
  3. Report, per dataset, how many HumanEval problems have any hit, so the decision
     about swapping evol-codealpaca-v1 for the decontaminated Magicoder version is a
     number rather than a hunch.
"""
import difflib
import json
import re
import sys
from collections import defaultdict

from datasets import load_dataset

W = 13


def norm(code: str) -> list[str]:
    code = re.sub(r"#.*", "", code)
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    return re.findall(r"[A-Za-z_]\w*|\d+|[^\s\w]", code)


def grams(toks, w=W):
    return {tuple(toks[i:i + w]) for i in range(max(0, len(toks) - w + 1))}


he = load_dataset("openai/openai_humaneval", split="test")
canon = {r["task_id"]: r["prompt"] + r["canonical_solution"] for r in he}
sol_only = {r["task_id"]: r["canonical_solution"] for r in he}
doc = {r["task_id"]: re.search(r'"""([\s\S]*?)"""', r["prompt"]).group(1).strip() if '"""' in r["prompt"] else r["prompt"] for r in he}

# ---- 1. chat passes vs canonical
chat = {json.loads(l)["id"]: json.loads(l) for l in open("results/chat/humaneval.jsonl")}
base = {json.loads(l)["id"]: json.loads(l) for l in open("results/base/humaneval.jsonl")}
print("== chat-model HumanEval passes vs canonical solution")
print(f"{'task':14s} {'new?':5s} {'ratio':>6s} {'lines_in_common':>16s}")
rows = []
for tid, r in chat.items():
    if not r["pass"]:
        continue
    gen = r["code"]
    a = "\n".join(l.strip() for l in sol_only[tid].strip().splitlines() if l.strip())
    b = "\n".join(l.strip() for l in gen.strip().splitlines() if l.strip())
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    common = len(set(a.splitlines()) & set(b.splitlines()))
    rows.append((tid, not base[tid]["pass"], ratio, common, len(a.splitlines())))
rows.sort(key=lambda x: -x[2])
for tid, new, ratio, common, n in rows:
    print(f"{tid:14s} {'new' if new else '':5s} {ratio:6.2f} {common:>7d} of {n:<3d} canonical lines")
hi = [r for r in rows if r[2] >= 0.9]
print(f"\n{len(rows)} passes, {sum(1 for r in rows if r[1])} new vs base, {len(hi)} with ratio >= 0.9 vs canonical")

# ---- 2 and 3. dataset scan
sol_grams = {tid: grams(norm(s)) for tid, s in sol_only.items()}
prompt_grams = {tid: grams(norm(p), 8) for tid, p in canon.items()}
doc_snips = {tid: re.sub(r"\s+", " ", d)[:80].lower() for tid, d in doc.items()}


def scan(name, ds, fields):
    hits_sol, hits_doc = defaultdict(int), defaultdict(int)
    n = 0
    for ex in ds:
        text = "\n".join(str(ex.get(f, "")) for f in fields)
        n += 1
        low = re.sub(r"\s+", " ", text).lower()
        for tid, snip in doc_snips.items():
            if len(snip) > 40 and snip in low:
                hits_doc[tid] += 1
        code_blocks = re.findall(r"```(?:python|py)?\n([\s\S]*?)```", text) or [text]
        g = set()
        for c in code_blocks:
            g |= grams(norm(c))
        if not g:
            continue
        for tid, sg in sol_grams.items():
            if len(sg) >= 3 and len(g & sg) >= max(3, len(sg) // 2):
                hits_sol[tid] += 1
    print(f"\n== {name}: {n:,} examples scanned")
    print(f"HumanEval problems whose docstring appears verbatim: {len(hits_doc)}  ({sum(hits_doc.values())} examples)")
    print(f"HumanEval problems whose canonical solution half-matches by 13-gram: {len(hits_sol)}  ({sum(hits_sol.values())} examples)")
    both = sorted(set(hits_doc) | set(hits_sol))
    print("problems:", ", ".join(t.split('/')[1] for t in both) if both else "none")
    return set(hits_doc), set(hits_sol)


evol = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
d1, s1 = scan("evol-codealpaca-v1", evol, ["instruction", "output"])
oss = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
d2, s2 = scan("Magicoder-OSS-Instruct-75K", oss, ["problem", "solution"])

new_pass = {r[0] for r in rows if r[1]}
flagged = (d1 | s1 | d2 | s2)
print(f"\n== overlap: {len(new_pass & flagged)} of the {len(new_pass)} chat-only passes are on problems flagged in the SFT data:",
      ", ".join(sorted(t.split('/')[1] for t in new_pass & flagged)) or "none")
allp = {r[0] for r in rows}
print(f"   {len(allp & flagged)} of all {len(allp)} chat passes are on flagged problems")
