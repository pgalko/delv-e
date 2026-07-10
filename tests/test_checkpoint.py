# Fresh reconstruction (original lost; see 6.18-era recovery). Kernel replay:
# executed cells enter history; restore_history replays them into a new worker
# so a resumed/extended run inherits the namespace.
import os, sys
_HERE=os.path.dirname(os.path.abspath(__file__)); _ROOT=os.path.dirname(_HERE)
sys.path[:0]=[os.path.join(_HERE,"stubs"),_ROOT]
from kernel import PersistentKernel
k1=PersistentKernel(df=None)
try:
    out,err,_=k1.execute("a = 40")
    assert not err, err
    out,err,_=k1.execute("b = a + 2\nprint(b)")
    assert not err and "42" in out
    hist=list(k1._history)
    assert len(hist)==2, f"history should hold both cells, got {len(hist)}"
finally:
    k1.cleanup()
k2=PersistentKernel(df=None)
try:
    k2.restore_history(hist)
    out,err,_=k2.execute("print(a + b)")
    assert not err and "82" in out, f"replayed namespace missing: {err or out}"
finally:
    k2.cleanup()
print("CHECKPOINT/REPLAY (history capture + restore into fresh worker) WORKS")
