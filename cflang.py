"""
CONTROL-FLOW source language — the step beyond pure arithmetic.

`evm_fetch.py` measured the honest gap: our arithmetic subset is ~40% of real
opcodes; the biggest missing piece is CONTROL FLOW (JUMP/JUMPI). This module is
the source language for that next milestone: arithmetic PLUS comparisons and
`if C then X else Y`.

Programs are expressions over a,b,c,d and small constants, now with:
  - comparisons   : <  >  ==        (signed, like EVM SLT/SGT/EQ)
  - conditionals  : if C then X else Y

The four pieces still all agree on meaning (generator / renderer / evaluator /
parser), so the data stays self-labelled and scoring stays objective. We keep
this in its own module so the validated arithmetic track (lang.py) is untouched.

AST nodes:
    ("num", 5)                      a constant
    ("var", "a")                    a variable
    ("op", "+",  l, r)              arithmetic (+ - *)
    ("op", "<",  l, r)              comparison (< > ==), evaluates to 1 or 0
    ("if", cond, then, else)        a conditional
"""

import random

VARS = ["a", "b", "c", "d"]
ARITH = ["+", "-", "*"]
CMP = ["<", ">", "=="]


# --------------------------------------------------------------------------- #
# 1. GENERATOR — random control-flow programs (self-labelled training data)
# --------------------------------------------------------------------------- #
def random_value(rng: random.Random, max_depth: int, leaf_prob: float = 0.45):
    """A plain arithmetic value (no control flow) — used for branches/operands."""
    if max_depth <= 0 or rng.random() < leaf_prob:
        if rng.random() < 0.5:
            return ("var", rng.choice(VARS))
        return ("num", rng.randint(0, 9))
    op = rng.choice(ARITH)
    return ("op", op,
            random_value(rng, max_depth - 1, leaf_prob),
            random_value(rng, max_depth - 1, leaf_prob))


def random_cond(rng: random.Random, max_depth: int):
    """A comparison between two small values -> a 1/0 condition."""
    op = rng.choice(CMP)
    d = max(0, max_depth - 1)
    return ("op", op, random_value(rng, d), random_value(rng, d))


def random_program(rng: random.Random, max_depth: int, if_prob: float = 0.55):
    """A program: with probability `if_prob` an if/else, otherwise arithmetic.
    Branches recurse with a decayed if-probability so trees stay shallow and
    learnable while still mixing pure-arithmetic and conditional programs."""
    if max_depth > 0 and rng.random() < if_prob:
        cond = random_cond(rng, max_depth)
        then = random_program(rng, max_depth - 1, if_prob * 0.5)
        els = random_program(rng, max_depth - 1, if_prob * 0.5)
        return ("if", cond, then, els)
    return random_value(rng, max_depth)


# --------------------------------------------------------------------------- #
# 2. RENDERER — AST -> source string (fully parenthesised, unambiguous)
# --------------------------------------------------------------------------- #
def render(ast) -> str:
    kind = ast[0]
    if kind == "num":
        return str(ast[1])
    if kind == "var":
        return ast[1]
    if kind == "if":
        return f"( if {render(ast[1])} then {render(ast[2])} else {render(ast[3])} )"
    _, op, left, right = ast
    return f"( {render(left)} {op} {render(right)} )"


# --------------------------------------------------------------------------- #
# 3. EVALUATOR — AST -> number (signed comparisons, like EVM SLT/SGT)
# --------------------------------------------------------------------------- #
def evaluate(ast, env: dict) -> int:
    kind = ast[0]
    if kind == "num":
        return ast[1]
    if kind == "var":
        return env[ast[1]]
    if kind == "if":
        return evaluate(ast[2], env) if evaluate(ast[1], env) != 0 else evaluate(ast[3], env)
    _, op, left, right = ast
    lv, rv = evaluate(left, env), evaluate(right, env)
    if op == "+":
        return lv + rv
    if op == "-":
        return lv - rv
    if op == "*":
        return lv * rv
    if op == "<":
        return 1 if lv < rv else 0
    if op == ">":
        return 1 if lv > rv else 0
    if op == "==":
        return 1 if lv == rv else 0
    raise ValueError(f"unknown op {op!r}")


# --------------------------------------------------------------------------- #
# 4. PARSER — source string -> AST (so we can run the model's GUESS and score)
# --------------------------------------------------------------------------- #
KEYWORDS = {"if", "then", "else"}


def tokenize_source(s: str) -> list[str]:
    tokens = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            i += 1
        elif ch in "()+-*<>":
            tokens.append(ch)
            i += 1
        elif ch == "=":
            if i + 1 < len(s) and s[i + 1] == "=":
                tokens.append("==")
                i += 2
            else:
                raise ValueError("single '=' is invalid")
        elif ch.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(s[i:j])
            i = j
        elif ch.isalpha():
            j = i
            while j < len(s) and s[j].isalpha():
                j += 1
            tokens.append(s[i:j])
            i = j
        else:
            raise ValueError(f"bad character {ch!r}")
    return tokens


class _Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.pos = 0

    def peek(self):
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def eat(self):
        t = self.peek()
        self.pos += 1
        return t

    def expect(self, k):
        if self.eat() != k:
            raise ValueError(f"expected {k!r}")

    def parse_cond(self):                       # lowest precedence: if/then/else
        if self.peek() == "if":
            self.eat()
            c = self.parse_cond()
            self.expect("then")
            t = self.parse_cond()
            self.expect("else")
            e = self.parse_cond()
            return ("if", c, t, e)
        return self.parse_compare()

    def parse_compare(self):                    # < > ==
        node = self.parse_addsub()
        while self.peek() in ("<", ">", "=="):
            op = self.eat()
            node = ("op", op, node, self.parse_addsub())
        return node

    def parse_addsub(self):                     # + -
        node = self.parse_term()
        while self.peek() in ("+", "-"):
            op = self.eat()
            node = ("op", op, node, self.parse_term())
        return node

    def parse_term(self):                       # *
        node = self.parse_factor()
        while self.peek() == "*":
            self.eat()
            node = ("op", "*", node, self.parse_factor())
        return node

    def parse_factor(self):                     # ( cond ) | num | var
        t = self.peek()
        if t == "(":
            self.eat()
            node = self.parse_cond()
            if self.eat() != ")":
                raise ValueError("missing )")
            return node
        if t is None:
            raise ValueError("unexpected end")
        if t in KEYWORDS:
            raise ValueError(f"unexpected keyword {t!r}")
        self.eat()
        if t.isdigit():
            return ("num", int(t))
        if t.isalpha() and t in VARS:
            return ("var", t)
        raise ValueError(f"unexpected token {t!r}")


def parse(s: str):
    """Parse source into an AST, or return None if malformed (a wrong guess)."""
    try:
        tokens = tokenize_source(s)
        p = _Parser(tokens)
        ast = p.parse_cond()
        if p.pos != len(tokens):
            return None
        return ast
    except (ValueError, IndexError):
        return None
