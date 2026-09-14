"""The chat template. Chapter 7.

A base model continues text. To make it answer questions we show it a few
hundred thousand examples in one fixed layout and train only on the answer
part, so it learns "after <|assistant|> comes a helpful reply that ends with
<|end|>". At inference we lay out the conversation the same way, stop at
<|end|>, and the model has become an assistant. The template is nothing more
than a convention, but it has to be exactly the same at training and inference.

    <|system|>{system}<|end|>
    <|user|>{user}<|end|>
    <|assistant|>{assistant}<|end|><|endoftext|>
"""

from __future__ import annotations

import re

from elroy.tokenizer import Tokenizer

SYSTEM = "You are Elroy, a small Python assistant built from scratch by Strife Technologies."


def build_prompt(user: str, system: str = SYSTEM) -> str:
    """Everything up to and including the assistant tag; the model writes the rest."""
    return f"<|system|>{system}<|end|><|user|>{user}<|end|><|assistant|>"


def format_example(tok: Tokenizer, user: str, assistant: str, system: str = SYSTEM,
                   max_len: int = 2048) -> tuple[list[int], list[int]] | None:
    """Token ids and a 0/1 loss mask for one training example.

    The mask is 0 over the system and user parts and 1 over the assistant reply
    and its closing <|end|>, so the model is never trained to predict the
    question, only the answer. Returns None if the example does not fit.
    """
    prompt_ids = tok.encode(build_prompt(user, system))
    reply_ids = tok.encode(assistant) + [tok.special_ids["<|end|>"], tok.eot]
    ids = prompt_ids + reply_ids
    if len(ids) > max_len:
        return None
    mask = [0] * len(prompt_ids) + [1] * len(reply_ids)
    return ids, mask


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(reply: str) -> str:
    """The first fenced code block in a reply, or the whole reply if there is none."""
    m = _FENCE.search(reply)
    if m:
        return m.group(1)
    # an unterminated fence at the end of a truncated reply
    i = reply.find("```")
    if i != -1:
        body = reply[i + 3:]
        body = body.split("\n", 1)[1] if "\n" in body else ""
        return body
    return reply
