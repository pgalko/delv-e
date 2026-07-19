# --- test bootstrap: runnable from the repo root via `python3 tests/<n>.py` ---
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
# --- end bootstrap ---

# The kernel namespace is shared and mutable; the Executor is blind to prior code
# BY DESIGN ("it never sees prior steps, prior specs, or prior code"). Two blind
# writers picking the same name is therefore guaranteed, not unlucky, and neither
# can know what a prior step's object actually contains.
#
# It has cost real runs in BOTH modes:
#
#   compute (lifespan): step 6 bound `records` to the baseline run. Step 8, blind
#     to step 6, rebound it to a different experiment. A later recovery step then
#     read step 8's number believing it was step 9's, and the published chart
#     plotted a series sitting at 9.9 under a caption reading "plateaus near 7.3".
#     Separately, an Executor had to GUESS what simulate_disposable_soma() returns
#     (the registry gave a signature and no return type); it guessed wrong three
#     times, abandoned its spec and truncated its output, which is the origin of
#     the only wrong number in that deliverable.
#
#   data (altitude): `common_athletes` came back a different shape than the next
#     step assumed (list of dicts, not strings). Two steps silently returned
#     nothing and a third had to rebuild it as common_athletes_v2.
#
# Two fixes, both pure plumbing, neither touching a prompt or adding context:
# the registry says what an object IS, and every step pins an immutable alias on
# what it bound.

import pandas as pd

from investigation import _referenced_names
from kernel import PersistentKernel

k = PersistentKernel(df=pd.DataFrame({"a": [1, 2, 3]}))
try:
    # ── 1) The registry carries a RETURN CONTRACT, observed from a real call ──
    k.execute("""
def simulate(K=400.0, c_m=0.04):
    return [(y, 7.31, 1.47, 861, 3.4) for y in range(50, 3001, 50)]
records = simulate()
""", step=6)
    reg = k.describe_namespace()
    assert "-> list[tuple(int,float,float,int,float)] len=60" in reg, reg
    print("return contract: the next blind Executor never has to guess: OK")

    # ── 2) Containers describe their ELEMENT SHAPE, not just "list len=60" ──
    assert "records: list[tuple(int,float,float,int,float)] len=60" in reg
    k.execute("common_athletes = [{'id': 'a01'}, {'id': 'a02'}]", step=7)
    reg = k.describe_namespace()
    assert "common_athletes: list[dict[str -> str] len=1] len=2" in reg, reg
    print("element shape: a list of dicts is not a list of strings: OK")

    # ── 3) A REBIND does not destroy the earlier object ──
    k.execute("records = [(y, 9.90, 1.2, 900, 4.1) for y in range(50, 3001, 50)]",
              step=8)
    out, err, _ = k.execute("""
print("bare  ", records[0][1])
print("s6    ", records__s6[0][1])
print("s8    ", records__s8[0][1])
""", step=9)
    assert err is None, err
    assert "bare   9.9" in out and "s6     7.31" in out and "s8     9.9" in out, out
    print("versioned alias: step 6's object survives step 8's rebind: OK")

    # ── 4) And the collision is impossible to miss ──
    reg = k.describe_namespace()
    assert "[AMBIGUOUS: rebound by steps 6, 8. Use records__s6 or records__s8" in reg, reg
    # the aliases are not listed as entries of their own: the registry does not double
    assert reg.count("records__s6:") == 0
    print("ambiguity: the bare name is flagged, the registry does not double: OK")

    # ── 5) A spec that pins an alias still resolves its registry object ──
    # Without this, a spec doing exactly the right thing would resolve NOTHING and
    # trip the self-containment tripwire, because \b does not match through "__".
    assert _referenced_names("chart mean_L from records__s6", k) == {"records"}
    assert _referenced_names("chart from records", k) == {"records"}
    assert _referenced_names("nothing named here", k) == set()
    print("spec resolution: a pinned alias resolves its object: OK")

    # ── 6) The chart pass runs AFTER synthesis and must mint no aliases ──
    k.execute("scratch = [1, 2, 3]")          # no step= : the chart pass
    reg = k.describe_namespace()
    assert "scratch:" in reg and "scratch__s" not in reg
    print("chart pass: post-synthesis steps do not version the namespace: OK")
finally:
    k.cleanup()

print("test_namespace: OK")
