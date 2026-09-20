"""Chapter 8: talk to Elroy in a browser, or from anything that speaks the OpenAI chat API.

    python scripts/serve.py --ckpt checkpoints/elroy-350m-chat/latest.pt --port 8100

Then open http://<host>:8100/ for a chat page, or point an OpenAI-compatible client
(Open WebUI, curl, the openai package) at http://<host>:8100/v1 with any API key.

    curl http://localhost:8100/v1/chat/completions -H 'Content-Type: application/json' \\
      -d '{"model":"elroy-350m-chat","messages":[{"role":"user","content":"Write fizzbuzz."}]}'

Standard library only, on top of elroy.sample. One request at a time (a lock
around generation); streaming is server-sent events in the OpenAI chunk format.
POST /reload re-reads the checkpoint, which is how the page picks up a new SFT
checkpoint while training is still running. Not a production server: no auth,
no batching, no KV cache. It exists so the model can be tried, not deployed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from elroy.chat import SYSTEM  # noqa: E402
from elroy.sample import load_model  # noqa: E402
from elroy.tokenizer import Tokenizer  # noqa: E402

STATE: dict = {}
LOCK = threading.Lock()
MIN_NEW_TOKENS = 3


def build_conversation(messages: list[dict]) -> str:
    """Lay out a multi-turn conversation in the Chapter 7 template."""
    system = SYSTEM
    turns = []
    for m in messages:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role == "system" and content:
            system = content
        elif role == "user" or (role == "assistant" and content):   # skip empty assistant turns
            turns.append((role, content))
    out = f"<|system|>{system}<|end|>"
    for role, content in turns:
        out += f"<|{role}|>{content}<|end|>"
    return out + "<|assistant|>"


@torch.no_grad()
def stream_tokens(ids: list[int], max_new_tokens: int, temperature: float, top_p: float):
    """Yield decoded text pieces one token at a time. Same sampler as elroy.sample."""
    import torch.nn.functional as F
    model, tok, device = STATE["model"], STATE["tok"], STATE["device"]
    stop = {tok.eot, tok.special_ids["<|end|>"], tok.special_ids["<|user|>"], tok.special_ids["<|system|>"]}
    x = torch.tensor([ids], dtype=torch.long, device=device)
    seq_len = model.cfg.seq_len
    out: list[int] = []
    emitted = 0
    for i in range(max_new_tokens):
        logits, _ = model(x[:, -seq_len:])
        logits = logits[0, -1].float()
        if i < MIN_NEW_TOKENS:
            # "hi" made the step-1000 SFT checkpoint answer with <|end|> and nothing else:
            # nothing in the instruction data looks like small talk. Refuse to stop before
            # a few real tokens so the reply is never empty.
            for sid in stop:
                logits[sid] = float("-inf")
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if 0 < top_p < 1:
                s, order = torch.sort(logits, descending=True)
                p = F.softmax(s, dim=-1)
                s[torch.cumsum(p, dim=-1) - p >= top_p] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(0, order, s)
            nxt = int(torch.multinomial(F.softmax(logits, dim=-1), 1))
        if nxt in stop:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        # decode the whole thing and emit the new suffix, so multibyte characters are never split
        text = tok.decode(out)
        if text.endswith("�"):
            continue
        yield text[emitted:]
        emitted = len(text)


def load(ckpt: str, tokenizer: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(ckpt, device)
    with LOCK:
        STATE.update(model=model, tok=Tokenizer.load(tokenizer), device=device, ckpt=ckpt,
                     name=os.path.basename(os.path.dirname(os.path.abspath(ckpt))) or "elroy",
                     loaded=time.strftime("%Y-%m-%d %H:%M:%S"))


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Elroy</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{margin:0;font:15px/1.5 system-ui,sans-serif;background:#fcfcfb;color:#0b0b0b}
header{padding:12px 20px;border-bottom:1px solid #e6e5e1;display:flex;gap:16px;align-items:baseline}
header b{font-size:17px} header span{color:#52514e;font-size:13px}
#log{max-width:820px;margin:0 auto;padding:16px 20px 120px}
.m{margin:14px 0;white-space:pre-wrap}.u{color:#52514e}.u:before{content:"you  ";font-weight:600;color:#0b0b0b}
.a:before{content:"elroy  ";font-weight:600;color:#2a78d6}
pre{background:#f1f0ec;padding:10px 12px;border-radius:6px;overflow-x:auto;white-space:pre}
form{position:fixed;bottom:0;left:0;right:0;background:#fcfcfb;border-top:1px solid #e6e5e1;padding:12px 20px}
.row{max-width:820px;margin:0 auto;display:flex;gap:8px}
textarea{flex:1;font:inherit;padding:10px;border:1px solid #c9c8c2;border-radius:6px;resize:none;height:44px}
button{font:inherit;padding:0 16px;border:0;border-radius:6px;background:#2a78d6;color:#fff;cursor:pointer}
button:disabled{background:#9db8d9}
</style></head><body>
<header><b>Elroy</b><span id="info"></span><span style="margin-left:auto">temperature <input id="temp" type="number" step="0.1" min="0" max="1.5" value="0.2" style="width:56px"></span></header>
<div id="log"></div>
<form id="f"><div class="row"><textarea id="q" placeholder="Ask for a Python function, or ask what a dictionary is. Shift+Enter for a new line."></textarea><button id="send">Send</button></div></form>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),f=document.getElementById('f'),send=document.getElementById('send');
let history=[];
fetch('/v1/models').then(r=>r.json()).then(d=>{document.getElementById('info').textContent=d.data[0].id+' · loaded '+d.data[0].loaded});
function render(text){const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;');let out='',i=0;const re=/```\\w*\\n([\\s\\S]*?)(```|$)/g;let m;
 while((m=re.exec(text))){out+=esc(text.slice(i,m.index))+'<pre>'+esc(m[1])+'</pre>';i=m.index+m[0].length;}return out+esc(text.slice(i));}
async function ask(text){history.push({role:'user',content:text});
 const u=document.createElement('div');u.className='m u';u.textContent=text;log.appendChild(u);
 const a=document.createElement('div');a.className='m a';log.appendChild(a);let reply='';send.disabled=true;
 const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({model:'elroy',messages:history,stream:true,temperature:parseFloat(document.getElementById('temp').value),max_tokens:400})});
 const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
 while(true){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
  let idx;while((idx=buf.indexOf('\\n\\n'))>=0){const line=buf.slice(0,idx).trim();buf=buf.slice(idx+2);
   if(!line.startsWith('data:'))continue;const data=line.slice(5).trim();if(data==='[DONE]')continue;
   const d=JSON.parse(data).choices[0].delta.content||'';reply+=d;a.innerHTML=render(reply);window.scrollTo(0,document.body.scrollHeight);}}
 history.push({role:'assistant',content:reply});send.disabled=false;q.focus();}
f.onsubmit=e=>{e.preventDefault();const t=q.value.trim();if(!t||send.disabled)return;q.value='';ask(t);};
q.onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();f.requestSubmit();}};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/v1/models"):
            self._json(200, {"object": "list", "data": [{"id": STATE["name"], "object": "model", "owned_by": "strife",
                                                          "loaded": STATE["loaded"], "ckpt": STATE["ckpt"]}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            req = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "bad json"})
        if self.path == "/reload":
            load(req.get("ckpt") or STATE["ckpt"], STATE["tok_path"])
            return self._json(200, {"reloaded": STATE["ckpt"], "at": STATE["loaded"]})
        if not self.path.startswith("/v1/chat/completions"):
            return self._json(404, {"error": "not found"})
        messages = req.get("messages") or []
        temperature = float(req.get("temperature", 0.2))
        top_p = float(req.get("top_p", 0.95))
        max_tokens = int(req.get("max_tokens") or 400)
        stream = bool(req.get("stream", False))
        prompt = build_conversation(messages)
        ids = STATE["tok"].encode(prompt)
        rid, created, name = "chatcmpl-" + uuid.uuid4().hex[:12], int(time.time()), STATE["name"]
        with LOCK:
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                def chunk(delta, finish=None):
                    d = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": name,
                         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    self.wfile.write(f"data: {json.dumps(d)}\n\n".encode())
                    self.wfile.flush()
                chunk({"role": "assistant", "content": ""})
                try:
                    for piece in stream_tokens(ids, max_tokens, temperature, top_p):
                        chunk({"content": piece})
                    chunk({}, "stop")
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                text = "".join(stream_tokens(ids, max_tokens, temperature, top_p))
                n_out = len(STATE["tok"].encode_ordinary(text))
                self._json(200, {"id": rid, "object": "chat.completion", "created": created, "model": name,
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                              "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": len(ids), "completion_tokens": n_out,
                                           "total_tokens": len(ids) + n_out}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()
    STATE["tok_path"] = args.tokenizer
    load(args.ckpt, args.tokenizer)
    print(f"serving {STATE['name']} from {args.ckpt} on http://{args.host}:{args.port}/", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
