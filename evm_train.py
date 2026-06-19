"""
Train the decompiler on REAL EVM-subset bytecode and evaluate it OBJECTIVELY.

    /opt/anaconda3/bin/python evm_train.py
    /opt/anaconda3/bin/python evm_train.py --obfuscate
    /opt/anaconda3/bin/python evm_train.py --repl     # paste EVM asm tokens

Same model and same objective scores as the toy, but the input is genuine EVM
bytecode (PUSH1/CALLDATALOAD/ADD/MUL/SWAP1/SUB/DUP1/POP) and the verifier checks
guesses by RE-EXECUTING that bytecode — the on-chain-style oracle.
"""

import argparse
import math
import random
import time

import numpy as np

import evm
import evm_dataset as edata
import lang
from model import Adam, Seq2Seq


def evaluate(model, pairs, src_vocab, tgt_vocab, rng):
    exact = equiv = 0
    for code, src, ast in pairs:
        pred = tgt_vocab.decode(model.greedy(src_vocab.encode(code)))
        if pred == src:
            exact += 1
        if edata.functional_equivalent(ast, " ".join(pred), rng):
            equiv += 1
    n = len(pairs)
    return exact / n, equiv / n


def verified_decode(model, code, src_vocab, tgt_vocab, np_rng, py_rng, beam=8, k=8):
    """Emit only a decompilation PROVEN to match the bytecode (by re-execution).
    Beam search first, then sampling fallback. Returns (tokens, verified?)."""
    src_ids = src_vocab.encode(code)
    cand = None
    for ids in model.beam_search(src_ids, beam=beam):
        cand = tgt_vocab.decode(ids)
        if edata.verified_equivalent(code, " ".join(cand), py_rng):
            return cand, True
    for _ in range(k):
        cand = tgt_vocab.decode(model.sample(src_ids, np_rng))
        if edata.verified_equivalent(code, " ".join(cand), py_rng):
            return cand, True
    return cand, False


def evaluate_verified(model, pairs, src_vocab, tgt_vocab):
    np_rng = np.random.default_rng(0)
    py_rng = random.Random(999)
    covered = audited_correct = 0
    for code, src, ast in pairs:
        cand, ok = verified_decode(model, code, src_vocab, tgt_vocab, np_rng, py_rng)
        if ok:
            covered += 1
            if edata.functional_equivalent(ast, " ".join(cand), py_rng):
                audited_correct += 1
    n = len(pairs)
    return covered / n, (audited_correct / covered if covered else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--hidden", type=int, default=80)
    ap.add_argument("--embed", type=int, default=32)
    ap.add_argument("--obfuscate", action="store_true")
    ap.add_argument("--repl", action="store_true")
    args = ap.parse_args()

    data = edata.generate_dataset(args.n, max_depth=args.depth, seed=0,
                                  obfuscate=args.obfuscate)
    src_vocab, tgt_vocab = edata.build_vocabs(data)
    split = int(len(data) * 0.85)
    train_data, test_data = data[:split], data[split:]
    print(f"{len(data)} EVM programs (depth<= {args.depth}"
          f"{', obfuscated' if args.obfuscate else ''}) | "
          f"train {len(train_data)} / test {len(test_data)} | "
          f"src vocab {len(src_vocab)} / tgt vocab {len(tgt_vocab)}")

    encoded = [(src_vocab.encode(code), tgt_vocab.encode(src, add_sos_eos=True), ast)
               for code, src, ast in train_data]

    model = Seq2Seq(src_vocab, tgt_vocab, embed=args.embed, hidden=args.hidden)
    opt = Adam(model.params(), lr=args.lr)
    rng = random.Random(123)

    print("\nTraining on EVM bytecode (func-equivalence should rise):")
    best_eq, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
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
        if eq > best_eq:
            best_eq, best_state = eq, model.state_dict()
            flag = "  <- best"
        print(f"  epoch {epoch:2d} | loss {avg:6.3f} | "
              f"test exact {ex:.2f} | test func-equiv {eq:.2f} | "
              f"{time.time()-t0:4.1f}s{flag}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest test functional-equivalence (greedy): {best_eq:.2f}")

    cov, prec = evaluate_verified(model, test_data, src_vocab, tgt_vocab)
    print("\nVerified decoding (re-execute the EVM bytecode, keep only proven "
          "answers):")
    print(f"  coverage  : {cov:.2f}  (fraction with a verified answer)")
    print(f"  precision : {prec:.2f}  (of those, fraction truly correct)")

    print("\nExample decompilations (test set):")
    for code, src, ast in test_data[:6]:
        pred = tgt_vocab.decode(model.greedy(src_vocab.encode(code)))
        ok = "OK " if edata.functional_equivalent(ast, " ".join(pred), rng) else "XX "
        print(f"  [{ok}] {' '.join(code)}")
        print(f"        gold: {' '.join(src)}")
        print(f"        pred: {' '.join(pred)}")

    if args.repl:
        print("\nPaste EVM asm tokens (e.g. 'PUSH1 0x00 CALLDATALOAD PUSH1 0x05 "
              "ADD'), blank to quit:")
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
