"""
Train the neural decompiler and evaluate it OBJECTIVELY.

    /opt/anaconda3/bin/python train.py                 # train + evaluate
    /opt/anaconda3/bin/python train.py --repl          # then paste bytecode

Two scores, both objective:
  - exact match:        predicted source tokens == original (string-identical)
  - functional equiv.:  predicted source computes the SAME values as the original
                        on random inputs (correct even if written differently)
Functional equivalence is the score that matters for a decompiler.
"""

import argparse
import math
import random
import time

import numpy as np

import dataset
import lang
from model import Adam, Seq2Seq


def evaluate(model, pairs, src_vocab, tgt_vocab, rng):
    exact = equiv = 0
    for code, src, ast in pairs:
        src_ids = src_vocab.encode(code)
        pred_ids = model.greedy(src_ids)
        pred_tokens = tgt_vocab.decode(pred_ids)
        if pred_tokens == src:
            exact += 1
        pred_str = " ".join(pred_tokens)
        if dataset.functional_equivalent(ast, pred_str, rng):
            equiv += 1
    n = len(pairs)
    return exact / n, equiv / n


def verified_decode(model, code, src_vocab, tgt_vocab, np_rng, py_rng, k=8):
    """Emit only a decompilation we can PROVE matches the bytecode.

    Try the greedy guess first; if it doesn't verify against the actual bytecode,
    draw up to k sampled candidates and return the first that does. Returns
    (tokens, verified?). If nothing verifies we still return the last guess but
    flag it as unverified — in a real tool you'd hand that to a human."""
    src_ids = src_vocab.encode(code)
    cand = tgt_vocab.decode(model.greedy(src_ids))
    if dataset.verified_equivalent(code, " ".join(cand), py_rng):
        return cand, True
    for _ in range(k):
        cand = tgt_vocab.decode(model.sample(src_ids, np_rng))
        if dataset.verified_equivalent(code, " ".join(cand), py_rng):
            return cand, True
    return cand, False


def evaluate_verified(model, pairs, src_vocab, tgt_vocab):
    """Coverage = fraction where we emit a verified answer. Then AUDIT it: using
    the hidden ground-truth source (which a real tool wouldn't have, but we do),
    confirm that 'verified against bytecode' really means 'truly correct'."""
    np_rng = np.random.default_rng(0)
    py_rng = random.Random(999)
    covered = audited_correct = 0
    for code, src, ast in pairs:
        cand, ok = verified_decode(model, code, src_vocab, tgt_vocab, np_rng, py_rng)
        if ok:
            covered += 1
            if dataset.functional_equivalent(ast, " ".join(cand), py_rng):
                audited_correct += 1
    n = len(pairs)
    return covered / n, (audited_correct / covered if covered else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="number of programs")
    ap.add_argument("--depth", type=int, default=2, help="max expression depth")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--embed", type=int, default=24)
    ap.add_argument("--repl", action="store_true")
    ap.add_argument("--obfuscate", action="store_true",
                    help="inject value-preserving dead code into the bytecode")
    args = ap.parse_args()

    # --- data ---------------------------------------------------------------
    data = dataset.generate_dataset(args.n, max_depth=args.depth, seed=0,
                                    obfuscate=args.obfuscate)
    src_vocab, tgt_vocab = dataset.build_vocabs(data)
    split = int(len(data) * 0.85)
    train_data, test_data = data[:split], data[split:]
    print(f"{len(data)} programs (depth<= {args.depth}) | "
          f"train {len(train_data)} / test {len(test_data)} | "
          f"src vocab {len(src_vocab)} / tgt vocab {len(tgt_vocab)}")

    # Pre-encode token ids once.
    encoded = [(src_vocab.encode(code), tgt_vocab.encode(src, add_sos_eos=True), ast)
               for code, src, ast in train_data]

    model = Seq2Seq(src_vocab, tgt_vocab, embed=args.embed, hidden=args.hidden)
    opt = Adam(model.params(), lr=args.lr)
    rng = random.Random(123)

    print("\nTraining (loss should fall, functional-equivalence should rise):")
    best_eq, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        # Cosine learning-rate decay: big steps early to learn fast, tiny steps
        # late so we settle into the minimum instead of overshooting it (which is
        # what made the obfuscated run diverge after its peak).
        opt.lr = args.lr * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / args.epochs))
        random.Random(epoch).shuffle(encoded)
        t0 = time.time()
        total_loss = 0.0
        for src_ids, tgt_ids, _ in encoded:
            opt.zero_grad()
            loss = model.loss(src_ids, tgt_ids)
            loss.backward()
            opt.step()
            total_loss += float(loss.data)
        avg = total_loss / len(encoded)
        ex, eq = evaluate(model, test_data, src_vocab, tgt_vocab, rng)
        flag = ""
        if eq > best_eq:                       # remember the best model so far
            best_eq, best_state = eq, model.state_dict()
            flag = "  <- best"
        print(f"  epoch {epoch:2d} | loss {avg:6.3f} | "
              f"test exact {ex:.2f} | test func-equiv {eq:.2f} | {time.time()-t0:4.1f}s{flag}")

    # Restore the best checkpoint (training can drift after its peak).
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest test functional-equivalence (greedy): {best_eq:.2f}")

    # --- verified decoding: only emit code PROVEN to match the bytecode --------
    cov, prec = evaluate_verified(model, test_data, src_vocab, tgt_vocab)
    print("\nVerified decoding (sample candidates, keep only ones that match the "
          "bytecode on random inputs):")
    print(f"  coverage  : {cov:.2f}  (fraction where we found a verified answer)")
    print(f"  precision : {prec:.2f}  (of those, fraction TRULY correct vs the "
          "hidden source — proves the verifier is sound)")

    # --- show some decompilations ------------------------------------------
    print("\nExample decompilations (test set):")
    for code, src, ast in test_data[:6]:
        pred = tgt_vocab.decode(model.greedy(src_vocab.encode(code)))
        ok = "OK " if dataset.functional_equivalent(ast, " ".join(pred), rng) else "XX "
        print(f"  [{ok}] {' '.join(code)}")
        print(f"        gold: {' '.join(src)}")
        print(f"        pred: {' '.join(pred)}")

    if args.repl:
        print("\nPaste bytecode tokens (e.g. 'PUSH_a PUSH_b ADD'), blank to quit:")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            try:
                ids = src_vocab.encode(line.split())
                pred = tgt_vocab.decode(model.greedy(ids))
                print("  ->", " ".join(pred))
            except Exception as e:
                print("  error:", e)


if __name__ == "__main__":
    main()
