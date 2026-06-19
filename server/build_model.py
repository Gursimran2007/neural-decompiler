"""
Train the control-flow decompiler and SAVE the checkpoint the API server loads.

    /opt/anaconda3/bin/python server/build_model.py

Writes server/model_cf.pkl = {state, src_vocab, tgt_vocab, embed, hidden}.
Fewer epochs than cftrain.py — func-equivalence saturates by epoch ~2, and we
keep the best checkpoint, so this is enough for a production artifact.
"""
import math
import pickle
import random
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cfdataset as cdata
from model import Adam, Seq2Seq

N, DEPTH, EPOCHS, HIDDEN, EMBED, LR = 1500, 2, 14, 96, 40, 0.004
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cf.pkl")


def main():
    data = cdata.generate_dataset(N, max_depth=DEPTH, seed=0)
    src_vocab, tgt_vocab = cdata.build_vocabs(data)
    split = int(len(data) * 0.85)
    train_data, test_data = data[:split], data[split:]
    encoded = [(src_vocab.encode(c), tgt_vocab.encode(s, add_sos_eos=True), a)
               for c, s, a in train_data]

    model = Seq2Seq(src_vocab, tgt_vocab, embed=EMBED, hidden=HIDDEN)
    opt = Adam(model.params(), lr=LR)
    rng = random.Random(123)
    best_eq, best_state = -1.0, None

    print(f"{len(data)} programs | train {len(train_data)} / test {len(test_data)}")
    for epoch in range(1, EPOCHS + 1):
        opt.lr = LR * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / EPOCHS))
        random.Random(epoch).shuffle(encoded)
        t0 = time.time()
        for s, t, _ in encoded:
            opt.zero_grad()
            loss = model.loss(s, t)
            loss.backward()
            opt.step()
        eq = sum(
            cdata.functional_equivalent(
                a, " ".join(tgt_vocab.decode(model.greedy(src_vocab.encode(c)))), rng)
            for c, s, a in test_data) / len(test_data)
        flag = ""
        if eq > best_eq:
            best_eq, best_state = eq, model.state_dict()
            flag = "  <- best"
        print(f"  epoch {epoch:2d} | func-equiv {eq:.3f} | {time.time()-t0:4.1f}s{flag}")

    model.load_state_dict(best_state)
    with open(OUT, "wb") as f:
        pickle.dump({
            "state": model.state_dict(),
            "src_vocab": src_vocab,
            "tgt_vocab": tgt_vocab,
            "embed": EMBED,
            "hidden": HIDDEN,
            "best_eq": best_eq,
        }, f)
    print(f"\nsaved {OUT}  (best func-equiv {best_eq:.3f})")


if __name__ == "__main__":
    main()
