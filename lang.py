"""
The toy SOURCE LANGUAGE we will decompile to.

Programs are arithmetic expressions over variables a,b,c,d and single-digit
constants, using + - * and parentheses, e.g.  ((a + b) * c) - 3

We need four things here, and they must all agree on the meaning of a program:
  1. a generator      — produce random ASTs (our infinite, self-labelled data)
  2. a renderer       — AST  -> readable source string  (the decompiler's target)
  3. an evaluator     — AST  -> number, given values for a,b,c,d
  4. a parser         — source string -> AST, so we can score the model's GUESS
                        by actually running it (functional-equivalence checking)

An AST node is a small tuple:
    ("num", 5)                      a constant
    ("var", "a")                    a variable
    ("op", "+", left, right)        a binary operation
"""

import random

VARS = ["a", "b", "c", "d"]
OPS = ["+", "-", "*"]


# --------------------------------------------------------------------------- #
# 1. GENERATOR — random programs = our training data, with perfect labels
# --------------------------------------------------------------------------- #
def random_ast(rng: random.Random, max_depth: int, leaf_prob: float = 0.35):
    """Recursively build a random expression tree up to `max_depth` deep."""
    if max_depth <= 0 or rng.random() < leaf_prob:
        if rng.random() < 0.5:
            return ("var", rng.choice(VARS))
        return ("num", rng.randint(0, 9))
    op = rng.choice(OPS)
    left = random_ast(rng, max_depth - 1, leaf_prob)
    right = random_ast(rng, max_depth - 1, leaf_prob)
    return ("op", op, left, right)


# --------------------------------------------------------------------------- #
# 2. RENDERER — AST -> source string (fully parenthesised, unambiguous)
# --------------------------------------------------------------------------- #
def render(ast) -> str:
    kind = ast[0]
    if kind == "num":
        return str(ast[1])
    if kind == "var":
        return ast[1]
    _, op, left, right = ast
    return f"( {render(left)} {op} {render(right)} )"


# --------------------------------------------------------------------------- #
# 3. EVALUATOR — AST -> number, given an environment {a:..,b:..,..}
# --------------------------------------------------------------------------- #
def evaluate(ast, env: dict) -> int:
    kind = ast[0]
    if kind == "num":
        return ast[1]
    if kind == "var":
        return env[ast[1]]
    _, op, left, right = ast
    lv, rv = evaluate(left, env), evaluate(right, env)
    if op == "+":
        return lv + rv
    if op == "-":
        return lv - rv
    return lv * rv


# --------------------------------------------------------------------------- #
# 4. PARSER — source string -> AST (recursive descent, full precedence)
#    Used to turn the MODEL'S predicted string back into something runnable.
#    Returns None if the string is malformed (a wrong guess), so scoring is safe
#    and we never need Python's eval().
# --------------------------------------------------------------------------- #
def tokenize_source(s: str) -> list[str]:
    """Split a source string into tokens: numbers, vars, operators, parens."""
    tokens = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            i += 1
        elif ch in "()+-*":
            tokens.append(ch)
            i += 1
        elif ch.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(s[i:j])
            i = j
        elif ch.isalpha():
            tokens.append(ch)
            i += 1
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

    def parse_expr(self):              # handles + and - (lowest precedence)
        node = self.parse_term()
        while self.peek() in ("+", "-"):
            op = self.eat()
            node = ("op", op, node, self.parse_term())
        return node

    def parse_term(self):              # handles * (higher precedence)
        node = self.parse_factor()
        while self.peek() == "*":
            self.eat()
            node = ("op", "*", node, self.parse_factor())
        return node

    def parse_factor(self):            # numbers, vars, or ( expr )
        t = self.peek()
        if t == "(":
            self.eat()
            node = self.parse_expr()
            if self.eat() != ")":
                raise ValueError("missing )")
            return node
        if t is None:
            raise ValueError("unexpected end")
        self.eat()
        if t.isdigit():
            return ("num", int(t))
        if t.isalpha() and t in VARS:
            return ("var", t)
        raise ValueError(f"unexpected token {t!r}")


def parse(s: str):
    """Parse source into an AST, or return None if it's invalid."""
    try:
        tokens = tokenize_source(s)
        p = _Parser(tokens)
        ast = p.parse_expr()
        if p.pos != len(tokens):       # leftover tokens => malformed
            return None
        return ast
    except (ValueError, IndexError):
        return None
