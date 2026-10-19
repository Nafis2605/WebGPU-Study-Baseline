// Build every spec model with TF.js (cpu backend, Node) and print its weight list as JSON, for
// scripts/check_spec.py to compare with the spec and the Keras builder. Also runs one forward pass
// per model to check the output shape.
import { readFileSync } from "node:fs";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-cpu";
import { buildModel, weightSlots } from "../src/models/build.js";

await tf.setBackend("cpu");
const spec = JSON.parse(readFileSync(new URL("../public/spec/models.json", import.meta.url), "utf8"));
const only = process.argv.slice(2);
const out = {};
for (const m of spec.models) {
  if (only.length && !only.includes(m.name)) continue;
  try {
    const model = buildModel(m);
    const slots = weightSlots(model, m);
    const y = tf.tidy(() => model.apply(tf.zeros([2, ...m.input_shape])));
    out[m.name] = { ok: true, weights: slots.map(s => [s.layer, s.name, s.shape]),
      params: model.countParams(), output_shape: y.shape };
    y.dispose();
    model.dispose();
  } catch (e) {
    out[m.name] = { ok: false, error: String(e?.stack ?? e) };
  }
}
console.log("RESULT " + JSON.stringify(out));
