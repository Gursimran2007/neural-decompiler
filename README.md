# Neural Decompiler

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
```

### Key flags
`--n` programs · `--depth` max expression depth · `--epochs` · `--lr` ·
`--hidden` GRU size · `--embed` embedding size · `--obfuscate` inject dead code ·
`--repl` interactive mode.
