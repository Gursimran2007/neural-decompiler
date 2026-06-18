"""
The NEURAL DECOMPILER: a sequence-to-sequence model with attention, built on
our own autograd engine.

    bytecode tokens ──► [ GRU encoder ] ──► hidden states H (one per input token)
                                                  │
    source tokens   ◄── [ GRU decoder + attention over H ] ◄── start token

WHY THESE PIECES
  - Embedding: each token id becomes a learned vector (its "meaning").
  - GRU (Gated Recurrent Unit): a recurrent cell that reads a sequence one step
    at a time, carrying a hidden "memory" state. Its gates let it keep or forget
    information — crucial for tracking nesting depth in the stack stream.
  - Attention: at each output step the decoder LOOKS BACK at all encoder states
    and focuses on the relevant ones. This is what makes it reconstruct
    structure (which operands belong to which operator) instead of guessing.

We process one program at a time (batch size 1): sequences are short and it
keeps the code readable. Shapes are (1, dim) row vectors throughout.
"""

import numpy as np

from autograd import Tensor, cat, parameter


class GRUCell:
    """One GRU step: h_t = GRU(x_t, h_{t-1})."""

    def __init__(self, in_dim, hid_dim):
        s = 1.0 / np.sqrt(hid_dim)
        # Update gate (z), reset gate (r), candidate (h~). Each has W (input),
        # U (recurrent) and b (bias).
        self.Wz, self.Uz, self.bz = parameter((in_dim, hid_dim), s), parameter((hid_dim, hid_dim), s), Tensor(np.zeros((1, hid_dim)))
        self.Wr, self.Ur, self.br = parameter((in_dim, hid_dim), s), parameter((hid_dim, hid_dim), s), Tensor(np.zeros((1, hid_dim)))
        self.Wh, self.Uh, self.bh = parameter((in_dim, hid_dim), s), parameter((hid_dim, hid_dim), s), Tensor(np.zeros((1, hid_dim)))

    def __call__(self, x, h):
        z = (x @ self.Wz + h @ self.Uz + self.bz).sigmoid()      # how much to update
        r = (x @ self.Wr + h @ self.Ur + self.br).sigmoid()      # how much to forget
        h_hat = (x @ self.Wh + (r * h) @ self.Uh + self.bh).tanh()
        return (Tensor(np.ones((1, h.shape[1]))) - z) * h + z * h_hat

    def params(self):
        return [self.Wz, self.Uz, self.bz, self.Wr, self.Ur, self.br,
                self.Wh, self.Uh, self.bh]


class Seq2Seq:
    def __init__(self, src_vocab, tgt_vocab, embed=24, hidden=48):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.hidden = hidden
        es = 0.1

        self.src_emb = parameter((len(src_vocab), embed), es)
        self.tgt_emb = parameter((len(tgt_vocab), embed), es)

        self.encoder = GRUCell(embed, hidden)
        # decoder input = previous-token embedding ++ attention context
        self.decoder = GRUCell(embed + hidden, hidden)

        # output projection from [decoder hidden ++ context] to vocab logits
        so = 1.0 / np.sqrt(hidden)
        self.Wo = parameter((hidden + hidden, len(tgt_vocab)), so)
        self.bo = Tensor(np.zeros((1, len(tgt_vocab))))

    # -- save / restore weights (to keep the best checkpoint) ----------------
    def state_dict(self):
        return [p.data.copy() for p in self.params()]

    def load_state_dict(self, state):
        for p, d in zip(self.params(), state):
            p.data = d.copy()

    # -- collect every trainable tensor (for the optimizer) ------------------
    def params(self):
        return ([self.src_emb, self.tgt_emb, self.Wo, self.bo]
                + self.encoder.params() + self.decoder.params())

    # -- encoder: read the bytecode, return stacked hidden states E (T, hidden)
    def encode(self, src_ids):
        h = Tensor(np.zeros((1, self.hidden)))
        states = []
        for tid in src_ids:
            x = self.src_emb[[tid]]            # (1, embed)
            h = self.encoder(x, h)
            states.append(h)
        E = cat(states, axis=0)                # (T, hidden)
        return E, h

    # -- attention: context vector = weighted blend of encoder states --------
    def attend(self, dec_h, E):
        scores = dec_h @ E.T                   # (1, T) dot-product attention
        alpha = scores.softmax()               # (1, T) attention weights
        context = alpha @ E                    # (1, hidden)
        return context

    # -- teacher-forced forward for TRAINING: returns total loss -------------
    def loss(self, src_ids, tgt_ids):
        E, enc_last = self.encode(src_ids)
        dec_h = enc_last
        total = Tensor(0.0)
        # predict tgt_ids[1:] given tgt_ids[:-1]  (tgt has <sos> ... <eos>)
        for t in range(len(tgt_ids) - 1):
            prev = tgt_ids[t]
            gold = tgt_ids[t + 1]
            y = self.tgt_emb[[prev]]           # (1, embed)
            context = self.attend(dec_h, E)
            dec_in = cat([y, context], axis=1) # (1, embed+hidden)
            dec_h = self.decoder(dec_in, dec_h)
            logits = cat([dec_h, context], axis=1) @ self.Wo + self.bo
            probs = logits.softmax()           # (1, vocab)
            total = total + (probs[0, gold].log()) * -1   # cross-entropy term
        return total

    # -- greedy decode for INFERENCE: returns predicted token ids ------------
    def greedy(self, src_ids, max_len=40):
        E, enc_last = self.encode(src_ids)
        dec_h = enc_last
        sos = self.tgt_vocab.stoi["<sos>"]
        eos = self.tgt_vocab.stoi["<eos>"]
        prev = sos
        out = []
        for _ in range(max_len):
            y = self.tgt_emb[[prev]]
            context = self.attend(dec_h, E)
            dec_in = cat([y, context], axis=1)
            dec_h = self.decoder(dec_in, dec_h)
            logits = cat([dec_h, context], axis=1) @ self.Wo + self.bo
            nxt = int(logits.data.argmax())
            if nxt == eos:
                break
            out.append(nxt)
            prev = nxt
        return out

    # -- sampling decode: like greedy but draws from the probability distribution
    #    (temperature controls randomness). Used to produce DIVERSE candidates so
    #    verified decoding can search for one that provably matches the bytecode.
    def sample(self, src_ids, rng, temperature=0.8, max_len=40):
        E, enc_last = self.encode(src_ids)
        dec_h = enc_last
        sos = self.tgt_vocab.stoi["<sos>"]
        eos = self.tgt_vocab.stoi["<eos>"]
        prev = sos
        out = []
        for _ in range(max_len):
            y = self.tgt_emb[[prev]]
            context = self.attend(dec_h, E)
            dec_in = cat([y, context], axis=1)
            dec_h = self.decoder(dec_in, dec_h)
            logits = (cat([dec_h, context], axis=1) @ self.Wo + self.bo).data[0]
            z = logits / temperature
            z = z - z.max()
            p = np.exp(z)
            p /= p.sum()
            nxt = int(rng.choice(len(p), p=p))
            if nxt == eos:
                break
            out.append(nxt)
            prev = nxt
        return out


class Adam:
    """Adam optimizer over a list of Tensors (standard, with bias correction)."""

    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8, clip=5.0):
        self.params = params
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, b1, b2, eps, clip
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def _clip_grads(self):
        """Scale all gradients down if their combined norm exceeds `clip`.
        This is the standard cure for exploding gradients in recurrent nets."""
        total = np.sqrt(sum(float((p.grad ** 2).sum()) for p in self.params))
        if total > self.clip:
            scale = self.clip / (total + 1e-12)
            for p in self.params:
                p.grad *= scale

    def step(self):
        self.t += 1
        self._clip_grads()
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
