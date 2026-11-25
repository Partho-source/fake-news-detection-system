"""coverage_run.py - statement coverage of the unit tests (uses only the standard library)."""
import ast, os, sys, unittest

MODS = ["validator.py", "preprocessor.py", "detector.py", "dataset.py", "database.py"]
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
classes = sys.argv[1].split(",")            # e.g. TestValidator,TestPreprocessor

def statements(path):                       # executable statements of a module
    tree = ast.parse(open(path).read())
    return {n.lineno for n in ast.walk(tree) if isinstance(n, ast.stmt)
            and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))}

hit = {os.path.join(BASE, m): set() for m in MODS}
def tracer(frame, event, arg):
    name = frame.f_code.co_filename
    if name in hit:
        if event == "line": hit[name].add(frame.f_lineno)
        return tracer
sys.settrace(tracer)
import tests.test_fnds as T                 # imported after tracing starts
suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(getattr(T, c)) for c in classes)
unittest.TextTestRunner(stream=open(os.devnull, "w")).run(suite)
sys.settrace(None)

tot = cov = 0
print(f"{'Module':18s}{'Stmts':>6s}{'Miss':>6s}{'Cover':>8s}")
for m in MODS:
    p = os.path.join(BASE, m); st = statements(p); h = hit[p] & st
    tot += len(st); cov += len(h)
    print(f"{m:18s}{len(st):6d}{len(st)-len(h):6d}{100*len(h)/len(st):7.0f}%")
print(f"{'TOTAL':18s}{tot:6d}{tot-cov:6d}{100*cov/tot:7.1f}%")
