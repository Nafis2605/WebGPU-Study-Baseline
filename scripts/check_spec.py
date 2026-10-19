"""
Gate: both builders reproduce public/spec/models.json exactly.

Builds every model with TF.js (Node, cpu backend: scripts/check_spec_js.mjs) and with Keras (tfgpu
env, CPU: native/keras_models.py), and checks per model: the ordered weight list (layer, name, shape)
equals the spec's, parameter counts equal the spec's, and the output shape is [2, 10].
Exit code 1 on any mismatch.

Usage:  .venv/Scripts/python scripts/check_spec.py [MODEL ...]
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchkit.paths import ROOT, SPEC, TFGPU, tfgpu_env  # noqa: E402


def result_line(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", **kw)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        raise SystemExit(f"{cmd[0]} produced no RESULT line:\n{p.stderr[-3000:]}")
    return json.loads(line[7:])


def main():
    only = sys.argv[1:]
    spec = {m["name"]: m for m in json.loads(SPEC.read_text(encoding="utf-8"))["models"]}
    js = result_line(["node", str(ROOT / "scripts" / "check_spec_js.mjs"), *only], cwd=ROOT)
    py = result_line([str(TFGPU / "python.exe"), str(ROOT / "native" / "keras_models.py"), *only],
                     cwd=ROOT, env=tfgpu_env(cpu_only=True))
    bad = 0
    for name, m in spec.items():
        if only and name not in only:
            continue
        want = [[l["name"], w["name"], w["shape"]] for l in m["layers"] for w in l.get("weights", [])]
        problems = []
        for side, r in (("tfjs", js.get(name)), ("keras", py.get(name))):
            if not r or not r["ok"]:
                problems.append(f"{side}: {(r or {}).get('error', 'missing')[:300]}")
                continue
            if r["weights"] != want:
                problems.append(f"{side}: weight list differs from spec")
            if r["params"] != m["params"]:
                problems.append(f"{side}: params {r['params']} != spec {m['params']}")
            if r["output_shape"] != [2, m["num_classes"]]:
                problems.append(f"{side}: output shape {r['output_shape']}")
        bad += bool(problems)
        print(f"{name:3s} {'OK ' if not problems else 'BAD'} params {m['params']:>11,d}  {'; '.join(problems)}")
    print(f"{'PASS' if not bad else 'FAIL'}: {bad} model(s) with mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
