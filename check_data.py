"""
Sanity-checks the data engine BEFORE we add any machine learning.
Run: /opt/anaconda3/bin/python check_data.py
"""

import random

import dataset
import lang
import vm


def show_samples():
    print("=== Sample (bytecode -> source) pairs ===")
    pairs = dataset.generate_dataset(n=6, max_depth=3, seed=1)
    for code, src, _ in pairs:
        print(f"  bytecode: {' '.join(code)}")
        print(f"  source  : {' '.join(src)}\n")


def check_consistency():
    """Every program must evaluate the same via AST and via its bytecode."""
    print("=== Consistency: AST eval == bytecode eval ===")
    rng = random.Random(42)
    pairs = dataset.generate_dataset(n=2000, max_depth=4, seed=7)
    bad = 0
    for code, _, ast in pairs:
        for _ in range(5):
            env = {v: rng.randint(-9, 9) for v in lang.VARS}
            if lang.evaluate(ast, env) != vm.run_bytecode(code, env):
                bad += 1
    print(f"  checked {len(pairs)} programs x5 inputs | mismatches: {bad}")


def check_equivalence_checker():
    """The objective scorer must accept correct rewrites and reject wrong ones."""
    print("=== Objective equivalence checker ===")
    rng = random.Random(0)
    _, _, ast = dataset.generate_pair(random.Random(3), max_depth=2)
    original = lang.render(ast)
    print(f"  original program: {original}")

    # A correct but differently-written guess: a+b  ==  b+a (commutativity).
    correct_guess = "( b + a )" if ast[0] == "op" and ast[1] == "+" else original
    same = "( a + b )"; comm = "( b + a )"
    print(f"  '{same}' equivalent to '{comm}' ? "
          f"{dataset.functional_equivalent(lang.parse(same), comm, rng)}")
    print(f"  '{same}' equivalent to '( a - b )' ? "
          f"{dataset.functional_equivalent(lang.parse(same), '( a - b )', rng)}")
    print(f"  malformed guess '( a + )' rejected ? "
          f"{not dataset.functional_equivalent(lang.parse(same), '( a + )', rng)}")


def show_vocabs():
    print("=== Vocabularies ===")
    pairs = dataset.generate_dataset(n=500, max_depth=3, seed=2)
    src_vocab, tgt_vocab = dataset.build_vocabs(pairs)
    print(f"  bytecode vocab ({len(src_vocab)}): {src_vocab.itos}")
    print(f"  source   vocab ({len(tgt_vocab)}): {tgt_vocab.itos}")


if __name__ == "__main__":
    show_samples()
    check_consistency()
    check_equivalence_checker()
    show_vocabs()
    print("\nData engine looks good." )
