# --- test bootstrap: runnable from the repo root via `python3 tests/<name>.py` ---
import os, sys, tempfile
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
def _tmpdir(tag):
    return tempfile.mkdtemp(prefix=f"delve_{tag}_")
def _require_dataset():
    p = os.environ.get("EEDR_DATASET", os.path.join(_ROOT, "datasets", "EEDR_sessions_laps_enriched.csv"))
    if not os.path.exists(p):
        print(f"SKIP: EEDR dataset not found at {p} (set EEDR_DATASET or place the CSV in datasets/)")
        sys.exit(0)
    return p
# --- end bootstrap ---

import sys
from investigation import Executor

class FakeKernel:
    def execute(self, code, analysis_dir=None): return ("ok output", None, [])

# 1) The exact failure from the run: model returns NO code block on every attempt (reasoning ran away)
class NoCodeClient:
    def __init__(s): s.n=0; s.max_tokens_seen=[]
    def call(s, msgs, model, agent=None, max_tokens=None, temperature=0):
        s.n+=1; s.max_tokens_seen.append(max_tokens)
        return ""  # empty content (truncated by reasoning)
c=NoCodeClient()
ex=Executor(c, "ollama:kimi", max_retries=2)
r=ex.run("spec X", FakeKernel(), "registry")
print("=== TEST 1: no code on all attempts (the crash case) ===")
print("  attempts made:", c.n, "| max_tokens passed:", c.max_tokens_seen)
print("  result error:", r["error"][:70])
print("  result code/stdout:", repr(r["code"]), repr(r["stdout"]))
assert c.n == 3, "should try max_retries+1 times"
assert r["error"] and "no runnable" in r["error"], "must return an error, not crash"
assert r["stdout"] is None and r["attempts"] == 3
assert all(mt >= 20000 for mt in c.max_tokens_seen), "executor should use a generous max_tokens cap"
print("  OK: returns a clean error dict (no UnboundLocalError), 20000-token headroom used")

# 2) Recovers: no code first, then valid code -> success
class RecoverClient:
    def __init__(s): s.n=0
    def call(s, msgs, model, agent=None, max_tokens=None, temperature=0):
        s.n+=1
        return "" if s.n==1 else "```python\nprint('hi')\n```"
ex2=Executor(RecoverClient(), "ollama:kimi", max_retries=2)
r2=ex2.run("spec", FakeKernel(), "reg")
print("\n=== TEST 2: empty first, code on retry -> success ===")
print("  error:", r2["error"], "| stdout:", r2["stdout"], "| attempts:", r2["attempts"])
assert r2["error"] is None and r2["stdout"]=="ok output" and r2["attempts"]==2
print("  OK: recovers via the skip-analysis retry")

# 3) Code that errors in kernel, then fixed -> error bound correctly, then success
class ErrThenFix:
    def __init__(s): s.n=0
    def call(s, msgs, model, agent=None, max_tokens=None, temperature=0):
        s.n+=1
        return "```python\nbad\n```" if s.n==1 else "```python\ngood\n```"
class KernelErrThenOk:
    def __init__(s): s.n=0
    def execute(s, code, analysis_dir=None):
        s.n+=1
        return (None, "Traceback: NameError", []) if s.n==1 else ("fixed output", None, [])
r3=Executor(ErrThenFix(),"ollama:kimi",max_retries=2).run("spec",KernelErrThenOk(),"reg")
print("\n=== TEST 3: kernel error then fix ===")
print("  error:", r3["error"], "| stdout:", r3["stdout"], "| attempts:", r3["attempts"])
assert r3["error"] is None and r3["stdout"]=="fixed output"
print("  OK: traceback-retry path still works, error variable always bound")

print("\nALL EXECUTOR CRASH-FIX TESTS PASSED")
