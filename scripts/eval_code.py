"""Code evaluation: HumanEval, HumanEval+ and MBPP pass@1. Chapters 5 and 7.

    python scripts/eval_code.py --ckpt checkpoints/elroy-350m/latest.pt --bench humaneval
    python scripts/eval_code.py --ckpt ... --bench mbpp --chat        # for the SFT model
    python scripts/eval_code.py --ckpt ... --bench humaneval --limit 20 --show 3

For each problem the model completes the function (base model: the prompt is
the signature and docstring; chat model: the problem is posed as a user
message and the code block in the reply is extracted). The completion is
concatenated with the problem's tests and run in a subprocess with a timeout
and memory limit. pass@1 is the fraction of problems whose tests pass, with
greedy decoding (temperature 0), which is how these numbers are usually quoted
for small models.

Running model-written code is running untrusted code. The subprocess gets a
CPU-time limit, a memory limit, an empty temp directory as cwd and no stdin,
which is enough for a home machine with no secrets in reach of the process;
it is not a security boundary. Use a VM if that matters to you.
"""

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from elroy.chat import build_prompt, extract_code  # noqa: E402
from elroy.sample import CODE_STOPS, generate, load_model, truncate_at  # noqa: E402
from elroy.tokenizer import Tokenizer  # noqa: E402


def load_bench(name: str) -> list[dict]:
    from datasets import load_dataset
    if name == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        return [{"id": r["task_id"], "prompt": r["prompt"], "entry": r["entry_point"],
                 "test": r["test"] + f"\n\ncheck({r['entry_point']})\n"} for r in ds]
    if name == "humanevalplus":
        ds = load_dataset("evalplus/humanevalplus", split="test")
        return [{"id": r["task_id"], "prompt": r["prompt"], "entry": r["entry_point"],
                 "test": r["test"] + f"\n\ncheck({r['entry_point']})\n"} for r in ds]
    if name == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        out = []
        for r in ds:
            # The standard MBPP prompt: the task text plus the tests, then the model
            # writes the function. We frame it as a docstring so the base model
            # sees a shape it knows.
            tests = "\n".join(r["test_list"])
            prompt = f'"""\n{r["prompt"]}\n{tests}\n"""\n'
            out.append({"id": f"mbpp/{r['task_id']}", "prompt": prompt, "entry": None,
                        "test": (r.get("test_setup_code") or "") + "\n" + tests + "\n", "task": r["prompt"]})
        return out
    raise ValueError(name)


def run_tests(program: str, timeout: float = 10.0, mem_mb: int = 4096) -> tuple[bool, str]:
    def limits() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2**20, mem_mb * 2**20))
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 1, int(timeout) + 1))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            p = subprocess.run([sys.executable, path], cwd=d, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=timeout, preexec_fn=limits)
            return p.returncode == 0, (p.stderr or "")[-400:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--bench", default="humaneval", choices=["humaneval", "humanevalplus", "mbpp"])
    ap.add_argument("--chat", action="store_true", help="use the chat template (SFT model)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--out", default=None, help="write per-problem results as JSONL")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.load(args.tokenizer)
    model, cfg = load_model(args.ckpt, device)
    problems = load_bench(args.bench)
    if args.limit:
        problems = problems[: args.limit]
    stops = {tok.eot, tok.special_ids.get("<|end|>", -1)}

    passed = 0
    t0 = time.time()
    results = []
    for i, prob in enumerate(problems):
        if args.chat:
            task = prob.get("task") or f"Complete the following Python function.\n\n```python\n{prob['prompt']}```"
            if prob["entry"] is not None:
                task = f"Complete this Python function so it does what the docstring says. Reply with the full function.\n\n```python\n{prob['prompt']}```"
            ids = tok.encode(build_prompt(task))
            new = generate(model, ids, args.max_new_tokens, args.temperature, stop_ids=stops, device=device)
            reply = tok.decode(new)
            code = extract_code(reply)
            program = code + "\n\n" + prob["test"]
            if prob["entry"] is not None and f"def {prob['entry']}" not in code:
                program = prob["prompt"] + code + "\n\n" + prob["test"]  # model gave only the body
        else:
            ids = tok.encode_ordinary(prob["prompt"])
            new = generate(model, ids, args.max_new_tokens, args.temperature, stop_ids=stops, device=device,
                           stop_strs=CODE_STOPS, tok=tok)
            completion = truncate_at(tok.decode(new), CODE_STOPS)
            program = prob["prompt"] + completion + "\n\n" + prob["test"]
            code = completion
        ok, err = run_tests(program)
        passed += ok
        results.append({"id": prob["id"], "pass": ok, "code": code, "err": err if not ok else ""})
        if i < args.show:
            print(f"----- {prob['id']} {'PASS' if ok else 'FAIL'} -----\n{prob['prompt']}{code}\n{err}\n")
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(problems)} pass@1 so far {passed / (i + 1):.1%} ({time.time() - t0:.0f}s)", flush=True)
    print(f"{args.bench}{' chat' if args.chat else ''}: pass@1 = {passed}/{len(problems)} = {passed / len(problems):.1%} "
          f"(T={args.temperature}, {time.time() - t0:.0f}s)")
    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
