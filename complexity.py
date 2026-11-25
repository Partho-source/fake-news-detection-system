"""complexity.py - cyclomatic complexity V(G) = number of decisions + 1 (per function)."""
import ast, sys

DECISIONS = (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.IfExp, ast.comprehension)

def vg(func):
    v = 1
    for n in ast.walk(func):
        if isinstance(n, DECISIONS):
            v += 1
        elif isinstance(n, ast.BoolOp):          # each 'and' / 'or' adds one decision
            v += len(n.values) - 1
    return v

def risk(v):
    return "Simple" if v <= 10 else "Moderate" if v <= 20 else "High"

for f in ["validator.py", "preprocessor.py", "detector.py", "dataset.py", "database.py", "model_select.py", "retrain.py", "signals.py", "sources.py", "urlfetch.py", "factcheck.py", "llm.py", "config.py", "app.py", "train.py"]:
    for n in ast.walk(ast.parse(open(f).read())):
        if isinstance(n, ast.FunctionDef):
            print(f"{f:16s}{n.name:18s}V(G)={vg(n):2d}  {risk(vg(n))}")
