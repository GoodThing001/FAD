"""Diagnose nupack behavior on the 141nt WT aptamer (each call in its own mode).

Segfaults cannot be caught, so each nupack call runs in a separate process via
the `mode` argument. Usage: python nupack_diag.py <model|mfe|pairs|pfunc|structure_prob>
"""
import sys
import time

import nupack

WT = ("UAUCCUUCGGGGCAGGGUGGAAAUCCCGACCGGCGGUAGUAAAGCACAUUUGCUUUAGAGCCC"
      "GUGACCCGUGUGCAUAAGCACGCGGUGGAUUCAGUUUAAGCUGAAGCCGACAGUGAAAGUCUGGA"
      "UGGGAGAAGGAUG")

mode = sys.argv[1] if len(sys.argv) > 1 else "mfe"
print(f"nupack {nupack.__version__} mode={mode}", flush=True)

t0 = time.time()
model = nupack.Model(material="rna", celsius=30)
print(f"model ok {time.time()-t0:.1f}s", flush=True)

s = nupack.Strand(WT, name="wt")
c = nupack.Complex([s])

if mode == "model":
    pass
elif mode == "mfe":
    t1 = time.time()
    res = nupack.mfe(c, model=model)
    print(f"mfe ok {time.time()-t1:.1f}s")
    print("structure:", res[0].structure[:80], "...")
    print("energy:", res[0].energy)
elif mode == "pairs":
    t1 = time.time()
    pp = nupack.pairs(c, model=model)
    print(f"pairs ok {time.time()-t1:.1f}s")
    arr = pp.to_array()
    print("shape:", arr.shape, "diag sum:", float(arr.sum()))
elif mode == "pfunc":
    t1 = time.time()
    pf = nupack.pfunc([s], model=model)
    print(f"pfunc ok {time.time()-t1:.1f}s")
    print("pfunc:", pf)
elif mode == "structure_prob":
    # probability of a specific structure under the Boltzmann ensemble
    t1 = time.time()
    mfe_res = nupack.mfe(c, model=model)
    struct = mfe_res[0].structure
    q = nupack.pfunc([s], model=model)
    prob = nupack.structure_probability([s], struct, model=model)
    print(f"struct prob ok {time.time()-t1:.1f}s  prob={prob}  pfunc={q}")
elif mode == "showstruct":
    mfe_res = nupack.mfe(c, model=model)
    st = mfe_res[0].structure
    print("type:", type(st))
    print("repr[:120]:", repr(st)[:120])
    print("str[:120]:", str(st)[:120])
    print("len:", len(st))
    try:
        print("to_dotparen:", nupack.to_dotparen(st) if hasattr(nupack, "to_dotparen") else "n/a")
    except Exception as e:
        print("to_dotparen err:", e)
else:
    print("unknown mode")
print("DONE", flush=True)
