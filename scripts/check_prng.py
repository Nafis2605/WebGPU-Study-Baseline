"""Gate: the Python port of the sketch PRNG (native/worker.py) matches src/prng.js bit for bit."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))
from worker import fnv1a32, mulberry32_signs  # noqa: E402

NAMES = ["dense_0/kernel/0", "block_1/wq/3", "lstm_4/recurrent_kernel/2", "x"]
N = 100000
JS = """
import { mulberry32, fnv1a32 } from "./src/prng.js";
const out = {};
for (const name of %s) {
  const r = mulberry32(fnv1a32(name)); let s = "";
  for (let i = 0; i < %d; i++) s += r() < 0.5 ? "-" : "+";
  out[name] = { h: fnv1a32(name), s };
}
console.log(JSON.stringify(out));
""" % (json.dumps(NAMES), N)

js = json.loads(subprocess.run(["node", "--input-type=module", "-e", JS], cwd=ROOT, capture_output=True, text=True,
                               check=True).stdout)
bad = 0
for n in NAMES:
    h = fnv1a32(n)
    s = "".join("-" if v < 0 else "+" for v in mulberry32_signs(h, N))
    ok = h == js[n]["h"] and s == js[n]["s"]
    bad += not ok
    print(f"{n:28s} hash {h:10d} {'OK' if ok else 'MISMATCH'}")
print("PASS" if not bad else "FAIL")
sys.exit(1 if bad else 0)
