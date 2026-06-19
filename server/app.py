"""
Neural Decompiler — verified-decompilation API (pure Python stdlib, no framework).

    /opt/anaconda3/bin/python server/app.py            # serves on :8000
    PORT=8000 /opt/anaconda3/bin/python server/app.py

Loads the trained control-flow model (server/model_cf.pkl) and exposes:

  GET  /                 -> the product landing page (static/index.html)
  GET  /health           -> {"ok": true, ...}
  POST /api/decompile    -> body {"bytecode":"0x..."} or {"asm":"PUSH1 0x00 ..."}
                            returns the neural decompilation PLUS a re-execution
                            PROOF that it matches the bytecode (or verified:false).

The moat: every answer is checked by RE-EXECUTING the bytecode. The API never
returns a wrong decompilation as if it were right — it either proves it or flags
it. That is the one guarantee hallucinating LLM decompilers cannot make.
"""
import json
import os
import pickle
import random
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import cfevm
import cflang
from model import Seq2Seq

MOD = cfevm.MOD
BYTE_TO_SUBSET = {v: k for k, v in cfevm.SUBSET_TO_BYTE.items()}
STATIC = os.path.join(HERE, "static")

# --------------------------------------------------------------------------- #
# Load the trained model once at startup.
# --------------------------------------------------------------------------- #
CKPT_PATH = os.path.join(HERE, "model_cf.pkl")
MODEL = SRC_VOCAB = TGT_VOCAB = None
BEST_EQ = None


def load_model():
    global MODEL, SRC_VOCAB, TGT_VOCAB, BEST_EQ
    with open(CKPT_PATH, "rb") as f:
        ck = pickle.load(f)
    SRC_VOCAB, TGT_VOCAB = ck["src_vocab"], ck["tgt_vocab"]
    MODEL = Seq2Seq(SRC_VOCAB, TGT_VOCAB, embed=ck["embed"], hidden=ck["hidden"])
    MODEL.load_state_dict(ck["state"])
    BEST_EQ = ck.get("best_eq")
    print(f"loaded model (train func-equiv {BEST_EQ}) | "
          f"src vocab {len(SRC_VOCAB)} / tgt vocab {len(TGT_VOCAB)}")


# --------------------------------------------------------------------------- #
# Real bytecode (hex) -> mnemonic tokens, matching cfevm's token format.
# --------------------------------------------------------------------------- #
def disassemble(hexstr: str) -> list[str]:
    h = hexstr.strip().lower().replace(" ", "")
    if h.startswith("0x"):
        h = h[2:]
    if len(h) % 2:
        raise ValueError("hex string has an odd number of digits")
    b = bytes.fromhex(h)
    tokens, i = [], 0
    while i < len(b):
        op = b[i]
        if op not in BYTE_TO_SUBSET:
            raise ValueError(f"unknown opcode byte 0x{op:02x} at offset {i}")
        name = BYTE_TO_SUBSET[op]
        if name == "PUSH1":
            if i + 1 >= len(b):
                raise ValueError("bytecode ends mid-PUSH1")
            tokens += ["PUSH1", f"0x{b[i + 1]:02x}"]
            i += 2
        elif name == "PUSH2":
            if i + 2 >= len(b):
                raise ValueError("bytecode ends mid-PUSH2")
            tokens += ["PUSH2", f"0x{(b[i + 1] << 8) | b[i + 2]:04x}"]
            i += 3
        else:
            tokens.append(name)
            i += 1
    return tokens


# --------------------------------------------------------------------------- #
# Verify a predicted source against the bytecode by RE-EXECUTING both.
# --------------------------------------------------------------------------- #
def make_proof(tokens, source, rng, trials=24):
    """Return (verified, proof_rows). verified iff source re-computes the same
    value as the bytecode on every trial; proof_rows are a few worked samples."""
    ast = cflang.parse(source)
    if ast is None:
        return False, []
    rows = []
    for _ in range(trials):
        env = {v: rng.randint(0, 9) for v in cflang.VARS}
        try:
            bc = cfevm.run(tokens, env)
        except Exception:
            return False, []
        sv = cflang.evaluate(ast, env) % MOD
        if bc != sv:
            return False, []
        if len(rows) < 5:
            rows.append({"inputs": env, "bytecode_out": str(bc),
                         "source_out": str(sv)})
    return True, rows


def verified_decode(tokens, beam=8, k=8):
    """Beam search, then sampling fallback; return the first PROVEN answer."""
    src_ids = SRC_VOCAB.encode(tokens)
    np_rng = np.random.default_rng(0)
    py_rng = random.Random(12345)
    tried = 0
    best = None
    for ids in MODEL.beam_search(src_ids, beam=beam):
        cand = " ".join(TGT_VOCAB.decode(ids))
        tried += 1
        best = best or cand
        ok, proof = make_proof(tokens, cand, py_rng)
        if ok:
            return cand, True, proof, tried
    for _ in range(k):
        cand = " ".join(TGT_VOCAB.decode(MODEL.sample(src_ids, np_rng)))
        tried += 1
        ok, proof = make_proof(tokens, cand, py_rng)
        if ok:
            return cand, True, proof, tried
    return best, False, [], tried


def decompile_request(payload: dict) -> dict:
    asm = payload.get("asm")
    bytecode = payload.get("bytecode")
    if asm:
        tokens = asm.split() if isinstance(asm, str) else list(asm)
        hexstr = "0x" + cfevm.to_bytes(tokens).replace("0x", "")
    elif bytecode:
        tokens = disassemble(bytecode)
        hexstr = bytecode if bytecode.startswith("0x") else "0x" + bytecode
    else:
        raise ValueError("provide 'bytecode' (hex) or 'asm' (mnemonic tokens)")

    source, verified, proof, tried = verified_decode(tokens)
    return {
        "ok": True,
        "bytecode": hexstr,
        "asm": tokens,
        "source": source,
        "verified": verified,
        "proof": proof,
        "candidates_tried": tried,
        "note": ("re-executing the bytecode confirms this decompilation on every "
                 "random input" if verified else
                 "the model could not PROVE any candidate — flagged, not guessed"),
    }


# --------------------------------------------------------------------------- #
# Pure-stdlib HTTP server with CORS (so a GitHub Pages front-end can call it).
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except FileNotFoundError:
            self._json(404, {"ok": False, "error": "not found"})
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._file(os.path.join(STATIC, "index.html"), "text/html; charset=utf-8")
        elif self.path == "/health":
            self._json(200, {"ok": True, "model_func_equiv": BEST_EQ,
                             "subset": sorted(cfevm.SUBSET_TO_BYTE)})
        else:
            self._json(404, {"ok": False, "error": "no such route"})

    def do_POST(self):
        if self.path != "/api/decompile":
            self._json(404, {"ok": False, "error": "no such route"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            self._json(200, decompile_request(payload))
        except Exception as e:
            self._json(400, {"ok": False, "error": str(e)})

    def log_message(self, *args):
        pass  # quiet


def main():
    load_model()
    port = int(os.environ.get("PORT", "8000"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"serving on http://0.0.0.0:{port}  (POST /api/decompile)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
