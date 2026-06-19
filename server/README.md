# Neural Decompiler — verified-decompilation API

A real, deployable web service around the from-scratch neural decompiler. It
reads EVM bytecode, reconstructs the source logic with the trained seq2seq model,
and **proves the answer by re-executing the bytecode**. If it can't prove an
answer, it returns `verified: false` instead of guessing — the one guarantee a
hallucinating LLM decompiler can't make.

Pure Python standard library + NumPy. **No web framework, no PyTorch** — same
"from scratch" ethos as the rest of the project, which also makes it trivial to
deploy (one container, one dependency).

## Run locally

```bash
# 1. build the model checkpoint (writes server/model_cf.pkl, ~1 min)
/opt/anaconda3/bin/python server/build_model.py

# 2. start the API + landing page
/opt/anaconda3/bin/python server/app.py        # http://localhost:8000
```

Open `http://localhost:8000` for the product page, or call the API directly:

```bash
curl -X POST http://localhost:8000/api/decompile \
  -H 'Content-Type: application/json' \
  -d '{"bytecode":"0x6040356002901261001257602035610015565b60045b"}'
```

```json
{
  "source": "( if ( c < 2 ) then 4 else b )",
  "verified": true,
  "proof": [{"inputs":{"a":6,"b":0,"c":4,"d":5},"bytecode_out":"0","source_out":"0"}, ...]
}
```

## Endpoints

| Method | Route | Body | Returns |
|---|---|---|---|
| `GET` | `/` | — | the product landing page |
| `GET` | `/health` | — | `{ok, model_func_equiv, subset}` |
| `POST` | `/api/decompile` | `{"bytecode":"0x.."}` or `{"asm":"PUSH1 0x00 .."}` | `source`, `verified`, `proof[]` |

## Deploy free (pick one)

The whole thing is one small container. The image already includes the trained
`model_cf.pkl`, so the host just runs it — no GPU, no training on deploy.

- **Fly.io** (free allowance, always-on small VM):
  ```bash
  fly launch --dockerfile server/Dockerfile   # from the repo root
  fly deploy
  ```
- **Render** — New → Web Service → Docker → Dockerfile path `server/Dockerfile`.
- **Hugging Face Spaces** (Docker SDK) — push this repo; Spaces builds the
  Dockerfile and gives a permanent URL.
- **Railway** — New → Deploy from repo → it detects the Dockerfile.

All inject `$PORT`; `app.py` reads it. After deploy, point the GitHub Pages
front-end at the live URL (see below).

## Front-end on GitHub Pages, API elsewhere

The landing page calls `/api/decompile` on its own origin by default. To serve
the page from GitHub Pages while the API runs on Fly/Render, set `API` near the
bottom of `server/static/index.html` to your deployed server URL (CORS is already
open on the server).
