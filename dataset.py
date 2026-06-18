"""
The self-supervised DATA ENGINE + objective SCORING.

This module turns the language/compiler into:
  - infinite (bytecode -> source) training pairs with perfect labels
  - vocabularies to map tokens <-> integers for the neural net
  - a FUNCTIONAL-EQUIVALENCE checker: the objective truth.

Functional equivalence is the heart of "objective, not subjective". The model's
output is correct if, when we run its predicted source, it returns the SAME
numbers as the original program on many random inputs — even if it wrote the
expression differently. We never grade on opinion.
"""

import random

import lang
import obfuscate as obf
import vm

# Special tokens every seq2seq model needs.
PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIAL = [PAD, SOS, EOS, UNK]


# --------------------------------------------------------------------------- #
# Pair generation
# --------------------------------------------------------------------------- #
def generate_pair(rng: random.Random, max_depth: int, obfuscate: bool = False):
    """Return (bytecode_tokens, source_tokens, ast) for one random program.

    The returned `ast` and `source` are always the CLEAN program (the target the
    decompiler must recover). When `obfuscate` is True the BYTECODE is compiled
    from a value-preserving obfuscated tree, so input != clean but output stays
    clean — forcing the model to strip the junk."""
    ast = lang.random_ast(rng, max_depth)
    source = lang.render(ast)
    code_ast = obf.obfuscate_ast(ast, rng) if obfuscate else ast
    code = vm.compile_ast(code_ast)
    return code, lang.tokenize_source(source), ast


def generate_dataset(n: int, max_depth: int = 3, seed: int = 0, dedup: bool = True,
                     obfuscate: bool = False):
    """Generate n unique programs as (bytecode_tokens, source_tokens, ast)."""
    rng = random.Random(seed)
    pairs = []
    seen = set()
    attempts = 0
    while len(pairs) < n and attempts < n * 50:
        attempts += 1
        code, src, ast = generate_pair(rng, max_depth, obfuscate)
        key = " ".join(code)
        if dedup and key in seen:
            continue
        seen.add(key)
        pairs.append((code, src, ast))
    return pairs


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class Vocab:
    def __init__(self, tokens):
        self.itos = list(SPECIAL) + sorted(set(tokens) - set(SPECIAL))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, add_sos_eos=False):
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in tokens]
        if add_sos_eos:
            ids = [self.stoi[SOS]] + ids + [self.stoi[EOS]]
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            t = self.itos[i]
            if t == EOS:
                break
            if t in (PAD, SOS):
                continue
            out.append(t)
        return out


def build_vocabs(pairs):
    src_tokens, tgt_tokens = [], []
    for code, src, _ in pairs:
        src_tokens.extend(code)
        tgt_tokens.extend(src)
    return Vocab(src_tokens), Vocab(tgt_tokens)


# --------------------------------------------------------------------------- #
# Objective scoring
# --------------------------------------------------------------------------- #
def functional_equivalent(original_ast, predicted_source: str,
                          rng: random.Random, trials: int = 30) -> bool:
    """
    True if the predicted source computes the same value as the original program
    across many random variable assignments. This is the objective ground truth.
    """
    pred_ast = lang.parse(predicted_source)
    if pred_ast is None:
        return False
    for _ in range(trials):
        env = {v: rng.randint(-5, 5) for v in lang.VARS}
        try:
            if lang.evaluate(original_ast, env) != lang.evaluate(pred_ast, env):
                return False
        except Exception:
            return False
    return True


def exact_match(original_source_tokens, predicted_source_tokens) -> bool:
    return original_source_tokens == predicted_source_tokens
