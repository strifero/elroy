"""elroy_min.py must agree with the training code. Run with:  python tests/test_min.py

Exports a tiny random model the way scripts/export_hf.py does, loads it back
with elroy_min, and checks logits are identical and the tokenizers agree.
"""

import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from elroy import elroy_min  # noqa: E402
from elroy.model import Elroy, ModelConfig  # noqa: E402
from elroy.tokenizer import Tokenizer, count_words, train_bpe  # noqa: E402


def main() -> None:
    from safetensors.torch import save_file
    torch.manual_seed(0)
    cfg = dict(vocab_size=1024, n_layer=2, n_head=4, d_model=64, d_ff=176, seq_len=32,
               rope_theta=10000.0, tie_embeddings=True, dropout=0.0)
    m = Elroy(ModelConfig(**cfg)).eval()
    d = tempfile.mkdtemp()
    save_file({k: v.contiguous() for k, v in m.state_dict().items() if k != "lm_head.weight"},
              os.path.join(d, "model.safetensors"))
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    here = os.path.dirname(__file__)
    wc = count_words([open(os.path.join(here, "..", "README.md")).read()])
    Tokenizer(train_bpe(wc, 1024, log=None), 1024).save(os.path.join(d, "tokenizer.json"))

    mm, tk = elroy_min.load(d, device="cpu")
    x = torch.randint(0, 1024, (2, 16))
    with torch.no_grad():
        a, _ = m(x)
        b = mm(x)
    assert torch.equal(a, b), (a - b).abs().max()
    t = Tokenizer.load(os.path.join(d, "tokenizer.json"))
    s = "def f(x):\n    return x  # <|endoftext|> ok"
    assert t.encode(s) == tk.encode(s) and tk.decode(tk.encode(s)) == s
    out = elroy_min.generate(mm, tk.encode("def "), max_new_tokens=5, temperature=0.0)
    assert len(out) <= 5
    print("ok: elroy_min logits identical, tokenizer identical, generate runs")


if __name__ == "__main__":
    main()
