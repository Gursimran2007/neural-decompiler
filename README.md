# Neural Decompiler

**[▶ Live interactive demo](https://gursimran2007.github.io/neural-decompiler/)** — watch the machine shred your code into EVM bytecode and rebuild the structure step by step.

**[⚙ Verified-decompilation API — live](https://gursimran1223-neural-decompiler.hf.space)** — a real deployed web service: `POST /api/decompile` returns the recovered source **plus a re-execution proof** that it matches the bytecode (or `verified:false` instead of guessing). Pure stdlib + NumPy, one container, running on Hugging Face Spaces. ([source](server/))

A neural network that **reverses compilation**: it reads flat stack-machine
bytecode and reconstructs the original nested source expression that produced it.

```
bytecode:  PUSH_8 PUSH_a SUB PUSH_3 PUSH_1 ADD MUL
              │
              ▼  (neural decompiler)
source:    ( ( 8 - a ) * ( 3 + 1 ) )
```

Everything here — the autograd engine, the GRU encoder/decoder, the attention,
the optimizer — is written **from scratch in NumPy**. No PyTorch, no TensorFlow,
no autodiff library. The only dependency is `numpy`.

---

## Why this project is different

Most "AI projects" are subjective, derivative, or thin wrappers around a
pretrained model. This one is deliberately none of those:

1. **Objective, not opinion-based.** A decompiler's output is right or wrong by a
   hard test: *does the predicted source compute the same values as the original
   program on random inputs?* That's `functional_equivalent()` — the ground
   truth. There is no human judgement, no "looks good to me." Correctness is a
   number.

2. **Self-supervised with perfect labels.** The data engine *generates* its own
   training pairs by compiling random programs. Labels are exact and infinite —
   no scraping, no annotation, no licensing. The compiler is the teacher.

3. **Genuinely deep ML.** Reconstructing nested structure from a flattened stack
   stream forces the model to learn the compiler's grammar in reverse — tracking
   stack depth to know which operands bind to which operator. That's what the
   GRU + attention actually learn to do.

4. **Built bottom-up, verified at every layer.** The autograd is checked against
   numerical finite-difference gradients (`test_autograd.py`). The data engine is
   cross-checked: every program is evaluated *both* via its AST and via its
   bytecode, and they must agree (`check_data.py`, 0 mismatches on 2000×5 runs).
   Only then is a model trained on top.

---

## Results

| Setting | Best test functional-equivalence |
|---|---|
| depth ≤ 2 (default) | **1.00** — perfect, stable across all epochs |
| depth ≤ 3 (`--depth 3 --lr 0.004`) | **1.00** — perfect at best checkpoint |
| depth ≤ 2, **obfuscated** (`--obfuscate`) | **0.94** — sees through dead-code injection |

With **verified decoding** (beam search + the bytecode oracle, see below):

| Setting | Coverage | Precision |
|---|---|---|
| toy bytecode, obfuscated | **0.99** | **1.00** |
| **real EVM bytecode**, clean | **1.00** | **1.00** |
| **real EVM bytecode**, obfuscated | **1.00** | **1.00** |
| **real EVM bytecode + control flow** (`if/else` via JUMP/JUMPI) | **1.00** | **1.00** |

The model reconstructs deeply nested expressions exactly, e.g.

```
PUSH_8 PUSH_b ADD PUSH_8 ADD PUSH_c PUSH_a MUL PUSH_3 PUSH_a SUB ADD SUB
  ->  ( ( ( 8 + b ) + 8 ) - ( ( c * a ) + ( 3 - a ) ) )
```

### Seeing through obfuscation (`obfuscate.py`)

Real reverse-engineers never get clean bytecode. With `--obfuscate`, the data
engine injects **semantic no-ops** — `x → x+0`, `x → x*1`, `x → (x+7)-7` — that
bloat the instruction stream (~1.75× longer) without changing what the program
computes. The training target stays the *clean* source, so the model must learn
to strip the junk.

The most striking result: the model learns to **simplify**, not memorize. It
routinely returns a *cleaner* answer than the label, and the objective scorer
accepts it because the values are identical:

```
PUSH_d PUSH_2 PUSH_0 ADD PUSH_0 SUB ADD
  label: ( d + ( 2 + 0 ) )
  pred : ( d + 2 )            ✓ accepted — d+(2+0) == d+2 for all inputs
```

This is why grading on functional equivalence — not string match — is the right
objective: it rewards understanding the computation, not parroting the tokens.

### Verified decoding — the model never lies (`dataset.verified_equivalent`)

LLM decompilers hallucinate: they output confident code that's subtly wrong, and
you can't tell which. We don't. The key realisation: **at inference you don't have
the source, but you DO have the bytecode** — so you can *run it* and check.

Verified decoding searches a pool of candidates — **beam search** first
(deterministic, the `beam` most-probable whole sequences), then stochastic
sampling as a fallback — and returns the first that **provably** reproduces the
bytecode's outputs on random inputs. Code that can't be verified is never
emitted; it's flagged for a human instead.

```
greedy func-equivalence : 0.93   (single best guess)
verified coverage       : 0.99   (fraction where a PROVEN answer was found)
verified precision      : 1.00   (of those, fraction truly correct vs the hidden
                                   source — i.e. the verifier is sound: "matches
                                   the bytecode" really does mean "correct")
```

Two things to read here:

- **Precision 1.00** — when the model commits, it is *never* wrong. It abstains
  on what it can't prove instead of guessing.
- **Beam search lifts coverage 0.93 → 0.99 without touching precision.** Greedy
  commits to one token at a time, so an early mistake is unrecoverable; beam
  keeps the best *whole sequences* alive, and the bytecode-verifier plucks out a
  correct one. We raise *how often we answer* without ever lowering *how often
  we're right*.

That correctness guarantee — not raw accuracy — is the genuinely defensible idea
here, and the one thing hallucinating LLM decompilers structurally can't offer.

---

## From toy to real: EVM smart-contract bytecode (`evm*.py`)

The toy VM proves the idea; the **EVM port** points it at a real, hungry market.
The Ethereum Virtual Machine is a **stack machine** — exactly the structure this
model is built for — and every contract's bytecode is **public on-chain** while
~99% have no verified source. That's a genuine reverse-engineering need.

What's *real* here (`evm.py`):

- **Genuine EVM opcodes & bytes** — `ADD=0x01`, `MUL=0x02`, `SUB=0x03`,
  `PUSH1=0x60`, `DUP1=0x80`, `SWAP1=0x90`, `CALLDATALOAD=0x35`. `to_bytes` emits
  hex a real Ethereum node executes identically (e.g. `( 0 + c )` →
  `0x6000604035 01`).
- **256-bit modular arithmetic** — every value is mod `2**256` and wraps, like
  on-chain. This is what keeps the verifier *sound*: "run the bytecode" means the
  same thing here as on Ethereum.
- **Inputs via `CALLDATALOAD`** — variables a,b,c,d are read from calldata at
  offsets `0x00/0x20/0x40/0x60`, exactly how Solidity reads function arguments.
- **Obfuscation with real stack opcodes** — `DUP1 POP`, `PUSH1 0x00 ADD`,
  `SWAP1 SWAP1` — the actual gadgets an EVM obfuscator uses.

The verifier is the same idea, now **re-executing the EVM bytecode**
(`evm_dataset.verified_equivalent`). Results: **1.00 coverage at 1.00 precision**
on both clean and obfuscated EVM bytecode.

### Honest reality check (`evm_fetch.py`)

No hand-waving about scope. This script pulls **real mainnet contracts** (WETH,
DAI, USDC, Uniswap V2/V3) from a public RPC node — no API key — disassembles
them, and measures the gap:

```
aggregate over 5 contracts, 33,337 instructions
inside our executable subset : 40.5%
the rest: JUMP/JUMPI/JUMPDEST (control flow), MLOAD/MSTORE (memory),
          SLOAD/SSTORE (storage), AND/comparisons, PUSH2..PUSH32
```

So: our model handles the **arithmetic core** (~40% of real opcodes) perfectly
and verifiably. The single biggest missing piece that gap flagged was **control
flow** (JUMP/JUMPI/JUMPDEST) — and that is the milestone below, now done.

---

## The control-flow milestone: recovering `if/else` from jump-soup (`cf*.py`)

Arithmetic is straight-line: tokens execute in order. **Control flow is the hard
part of real decompilation**, because an `if` in the source has *no `if` in the
bytecode* — Solidity compiles it to a conditional jump over a block:

```
if cond then THEN else ELSE
   ⇣ compiles to ⇣
<cond> ; PUSH2 Ltrue ; JUMPI ; <ELSE> ; PUSH2 Lend ; JUMP
Ltrue: JUMPDEST ; <THEN> ; Lend: JUMPDEST
```

Decompiling means recovering the nested `if/else` structure back out of that flat
jump-soup — following absolute byte addresses to their `JUMPDEST` landing pads
and re-nesting the branches. This is exactly what Ghidra / Hex-Rays call
**control-flow structuring**.

What's *real* here (`cfevm.py`) — a genuine **program-counter machine**, not a
straight-line one:

- **Real control-flow opcodes & bytes** — `JUMP=0x56`, `JUMPI=0x57`,
  `JUMPDEST=0x5b`, signed comparisons `SLT=0x12`, `SGT=0x13`, `EQ=0x14`, and
  `PUSH2=0x61` carrying a **2-byte jump address** — exactly how Solidity emits
  targets.
- **A two-pass assembler** resolves jump addresses (pass 1 assigns a byte
  program-counter to every instruction and records each `JUMPDEST`; pass 2 fills
  the `PUSH2` targets) — the same thing a real assembler does.
- **A PC interpreter that follows the jumps** — `JUMP`/`JUMPI` move execution to a
  byte offset that *must* land on a `JUMPDEST` (validated, like a real node),
  with 256-bit modular arithmetic and signed two's-complement comparisons.

The VM is verified before any ML (`check_cf.py`): **0 mismatches on 4000 programs
× 6 inputs**, AST-eval == bytecode-eval, both branches of every `if` exercised,
and bytecode round-trips through real hex.

### Results — the model reads the jumps

The **same from-scratch seq2seq+attention model**, trained on 1,500 control-flow
programs (1,011 with `if/else`, depth ≤ 2, including **nested** conditionals):

| Metric | Score |
|---|---|
| greedy functional-equivalence (best) | **1.00** |
| verified coverage (re-execute through the jumps) | **1.00** |
| verified precision | **1.00** |

It recovers nested branches exactly, straight out of the jump addresses:

```
PUSH1 0x40 CALLDATALOAD PUSH1 0x40 CALLDATALOAD ADD PUSH1 0x20 CALLDATALOAD
SWAP1 SLT PUSH2 0x0016 JUMPI PUSH1 0x02 PUSH2 0x002c JUMP JUMPDEST PUSH1 0x09
PUSH1 0x20 CALLDATALOAD EQ PUSH2 0x0027 JUMPI PUSH1 0x06 PUSH2 0x002b JUMP
JUMPDEST PUSH1 0x00 CALLDATALOAD JUMPDEST JUMPDEST
  ->  ( if ( ( c + c ) < b ) then ( if ( 9 == b ) then a else 6 ) else 2 )
```

The verifier is the same moat, now **re-executing bytecode that contains real
jumps** (`cfdataset.verified_equivalent`): a "verified" answer is provably a
correct decompilation of that control flow, with no source needed — exactly how
you'd check a guess against an on-chain contract.

---

## Architecture

```
bytecode tokens ─► [ GRU encoder ] ─► hidden states H (one per input token)
                                            │
source tokens   ◄─ [ GRU decoder + attention over H ] ◄─ <sos>
```

- **Embedding** — each token id becomes a learned vector.
- **GRU encoder** — reads the bytecode one step at a time, carrying a memory
  state; its gates let it track nesting depth in the stack stream.
- **Dot-product attention** — at each output step the decoder looks back over all
  encoder states and focuses on the relevant ones. This is what lets it
  reconstruct *which operands belong to which operator* instead of guessing.
- **GRU decoder + output projection** — emits the source tokens one at a time.

Training is teacher-forced cross-entropy; inference is greedy decoding.
**Adam** with **global-norm gradient clipping** (the cure for the exploding
gradients that recurrent nets are prone to), a **cosine learning-rate schedule**
(big steps early to learn fast, tiny steps late to settle into the minimum
instead of overshooting it), and **best-checkpoint restore** so a late-training
wobble can't cost us the peak model.

---

## File map

| File | What it is |
|---|---|
| `lang.py` | The toy source language: random ASTs, a renderer, an evaluator, and a recursive-descent parser. |
| `vm.py` | The compiler + stack VM: AST → bytecode (post-order), and a bytecode interpreter. |
| `obfuscate.py` | Value-preserving dead-code injection (identity transforms) to make decompilation a real reverse-engineering problem. |
| `dataset.py` | The self-supervised data engine + vocabularies + the objective `functional_equivalent` scorer. |
| `autograd.py` | A reverse-mode automatic-differentiation engine (~150 lines). Our mini-PyTorch. |
| `test_autograd.py` | Verifies every gradient against numerical finite differences. |
| `model.py` | `GRUCell`, the `Seq2Seq` encoder-decoder with attention, and the `Adam` optimizer. |
| `train.py` | Trains the model and evaluates it objectively; `--repl` for live decompilation. |
| `check_data.py` | Sanity-checks the data engine before any ML. |
| `evm.py` | **Real EVM-subset** stack machine: genuine opcodes/bytes, 256-bit modular arithmetic, assemble/disassemble, stack-opcode obfuscation. |
| `check_evm.py` | Verifies the EVM VM: AST-eval (mod 2²⁵⁶) == bytecode-eval, and bytes round-trip. |
| `evm_dataset.py` | EVM data engine + the re-execution oracle (`verified_equivalent`). |
| `evm_train.py` | Trains/evaluates on EVM bytecode with verified + beam decoding; `--obfuscate`, `--repl`. |
| `evm_fetch.py` | Pulls real mainnet contracts from a public RPC and reports the honest opcode-coverage gap. |
| `cflang.py` | **Control-flow source language**: arithmetic + comparisons + `if/else`. Generator / renderer / evaluator / parser, all agreeing on meaning. |
| `cfevm.py` | **EVM machine with real control flow**: `if` → conditional jumps, a two-pass jump-address assembler, and a program-counter interpreter that follows JUMP/JUMPI. |
| `check_cf.py` | Verifies the control-flow VM: AST-eval == bytecode-eval (following the jumps), both branches exercised, bytes round-trip. |
| `cfdataset.py` | Control-flow data engine + the re-execution oracle that follows jumps (`verified_equivalent`). |
| `cftrain.py` | Trains/evaluates on control-flow bytecode with verified + beam decoding; `--repl`. |

---

## Run it

> Use the Anaconda interpreter (has a working NumPy):
> `/opt/anaconda3/bin/python`

```bash
# 1. verify the autograd is correct (gradients vs finite differences)
/opt/anaconda3/bin/python test_autograd.py

# 2. sanity-check the data engine (AST eval == bytecode eval)
/opt/anaconda3/bin/python check_data.py

# 3. train + evaluate objectively
/opt/anaconda3/bin/python train.py

# harder regime
/opt/anaconda3/bin/python train.py --depth 3 --n 1200 --epochs 30 --lr 0.004

# hardest: decompile OBFUSCATED bytecode (see through injected dead code)
/opt/anaconda3/bin/python train.py --obfuscate --n 2000 --epochs 40 --hidden 96 --embed 32 --lr 0.004

# 4. train, then decompile your own bytecode interactively
/opt/anaconda3/bin/python train.py --repl
#   > PUSH_a PUSH_b PUSH_1 ADD ADD
#   -> ( a + ( b + 1 ) )

# --- REAL EVM track ---
# 5. verify the EVM VM (mod 2^256 eval, byte round-trip)
/opt/anaconda3/bin/python check_evm.py

# 6. train + verified-decode on REAL EVM bytecode (clean, then obfuscated)
/opt/anaconda3/bin/python evm_train.py
/opt/anaconda3/bin/python evm_train.py --obfuscate --n 2000 --epochs 40 --hidden 96

# 7. pull real mainnet contracts and see the honest coverage gap
/opt/anaconda3/bin/python evm_fetch.py

# --- CONTROL-FLOW track (recover if/else from real JUMP/JUMPI) ---
# 8. verify the control-flow VM (eval == bytecode, following the jumps)
/opt/anaconda3/bin/python check_cf.py

# 9. train + verified-decode on bytecode with REAL control flow
/opt/anaconda3/bin/python cftrain.py
/opt/anaconda3/bin/python cftrain.py --repl
#   > PUSH1 0x40 CALLDATALOAD PUSH1 0x02 SWAP1 SLT PUSH2 0x0012 JUMPI ...
#   -> ( if ( c < 2 ) then 4 else b )
```

### Key flags
`--n` programs · `--depth` max expression depth · `--epochs` · `--lr` ·
`--hidden` GRU size · `--embed` embedding size · `--obfuscate` inject dead code ·
`--repl` interactive mode.
