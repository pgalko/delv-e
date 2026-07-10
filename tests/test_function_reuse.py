# Fresh reconstruction (original lost). Namespace registry: a function defined
# in one cell is advertised via describe_namespace and callable in later cells.
import os, sys
_HERE=os.path.dirname(os.path.abspath(__file__)); _ROOT=os.path.dirname(_HERE)
sys.path[:0]=[os.path.join(_HERE,"stubs"),_ROOT]
from kernel import PersistentKernel
k=PersistentKernel(df=None)
try:
    out,err,_=k.execute("def double(x):\n    return x * 2\nprint(double(3))")
    assert not err and "6" in out
    reg=k.describe_namespace()
    assert "double" in reg, f"registry must advertise the function: {reg[:200]}"
    out,err,_=k.execute("print(double(5))")
    assert not err and "10" in out, "function not reusable across cells"
finally:
    k.cleanup()
print("FUNCTION REUSE (registry advertises, later cells reuse) WORKS")
