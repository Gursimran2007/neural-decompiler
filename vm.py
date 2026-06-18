"""
The COMPILER and the stack VIRTUAL MACHINE.

The "compiled" form a real decompiler sees is low-level and flat. We mimic that
with classic stack-machine bytecode (like Python's own bytecode, or the JVM):

    source:    ( a + b ) * c
    bytecode:  PUSH a  PUSH b  ADD  PUSH c  MUL

Notice the bytecode is a FLAT sequence with no parentheses — the tree structure
is gone. The neural decompiler's job is to rebuild that structure from the flat
stream. That's why this is a real learning problem, not a lookup table.

We compile by post-order traversal (children before parent), which is exactly
how an expression tree turns into stack operations.
"""

OP_TO_INSTR = {"+": "ADD", "-": "SUB", "*": "MUL"}
INSTR_TO_PY = {"ADD": lambda x, y: x + y,
               "SUB": lambda x, y: x - y,
               "MUL": lambda x, y: x * y}


def compile_ast(ast) -> list[str]:
    """AST -> list of bytecode tokens (post-order)."""
    kind = ast[0]
    if kind == "num":
        return [f"PUSH_{ast[1]}"]
    if kind == "var":
        return [f"PUSH_{ast[1]}"]
    _, op, left, right = ast
    return compile_ast(left) + compile_ast(right) + [OP_TO_INSTR[op]]


def run_bytecode(code: list[str], env: dict) -> int:
    """Execute bytecode on a stack machine and return the result."""
    stack = []
    for instr in code:
        if instr.startswith("PUSH_"):
            operand = instr[5:]
            if operand.lstrip("-").isdigit():
                stack.append(int(operand))      # constant
            else:
                stack.append(env[operand])      # variable
        else:
            right = stack.pop()
            left = stack.pop()
            stack.append(INSTR_TO_PY[instr](left, right))
    return stack.pop()
