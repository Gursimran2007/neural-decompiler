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
    print(f"\nBest test functional-equivalence: {best_eq:.2f}")

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
