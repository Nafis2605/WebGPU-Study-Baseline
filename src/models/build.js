// TF.js model builder: turns one public/spec/models.json entry into a tf.Sequential.
//
// Every weight is created with a zeros initialiser (TF.js draws no random numbers, and LSTM/GRU
// orthogonal init's QR is avoided); the study's initial or trained weights are then written from
// the shared weight files (src/weights.js), the same bytes the Keras builder loads.
//
// Layer semantics match native/keras_models.py op for op:
//   dense / conv2d / maxpool2d / flatten / reshape / simple_rnn / lstm / gru  tfjs-layers built-ins
//     (LSTM/GRU recurrentActivation set to 'sigmoid' explicitly: TF.js defaults to hardSigmoid,
//     Keras to sigmoid; GRU resetAfter=false is the only variant TF.js implements)
//   positional       x + table[T, d]
//   encoder_block    pre-LN transformer block, attention from matMul (Einsum has no gradient in TF.js)
//   layer_norm       (x - mean) * rsqrt(var + eps) * gamma + beta with explicit elementwise ops, so
//                    every backend runs the same primitives (no FusedBatchNorm broadcasting)
//   global_avg_pool_1d  mean over the token axis
// Works in the browser and in Node (scripts/check_spec.mjs uses it with the cpu backend).
import * as tf from "@tensorflow/tfjs-core";
import * as tfl from "@tensorflow/tfjs-layers";

const zeros = () => tfl.initializers.zeros();

function layerNorm(x, gamma, beta, eps) {
  const { mean, variance } = tf.moments(x, -1, true);
  return tf.add(tf.mul(tf.mul(tf.sub(x, mean), tf.rsqrt(tf.add(variance, eps))), gamma), beta);
}

/** x[..., in] @ w[in, out] + b, as a 2-D matMul on the flattened leading dims. */
function linear(x, w, b) {
  const lead = x.shape.slice(0, -1);
  const y = tf.add(tf.matMul(tf.reshape(x, [-1, x.shape[x.shape.length - 1]]), w), b);
  return tf.reshape(y, [...lead, w.shape[1]]);
}

class Positional extends tfl.layers.Layer {
  static className = "StudyPositional";
  constructor(cfg) { super({ name: cfg.name }); this.cfg = cfg; }
  build() { this.table = this.addWeight("table", [this.cfg.length, this.cfg.dim], "float32", zeros()); this.built = true; }
  computeOutputShape(s) { return s; }
  call(inputs) {
    return tf.tidy(() => tf.add(Array.isArray(inputs) ? inputs[0] : inputs, this.table.read()));
  }
  getConfig() { return { ...super.getConfig(), ...this.cfg }; }
}

class LayerNorm extends tfl.layers.Layer {
  static className = "StudyLayerNorm";
  constructor(cfg) { super({ name: cfg.name }); this.cfg = cfg; }
  build(s) {
    const d = s[s.length - 1];
    this.gamma = this.addWeight("gamma", [d], "float32", zeros());
    this.beta = this.addWeight("beta", [d], "float32", zeros());
    this.built = true;
  }
  computeOutputShape(s) { return s; }
  call(inputs) {
    return tf.tidy(() => layerNorm(Array.isArray(inputs) ? inputs[0] : inputs, this.gamma.read(), this.beta.read(), this.cfg.ln_eps));
  }
  getConfig() { return { ...super.getConfig(), ...this.cfg }; }
}

class EncoderBlock extends tfl.layers.Layer {
  static className = "StudyEncoderBlock";
  constructor(cfg) { super({ name: cfg.name }); this.cfg = cfg; }
  build() {
    // addWeight order = spec weight order (ln1, q, k, v, o, ln2, ffn).
    for (const w of this.cfg.weights) this[w.name] = this.addWeight(w.name, w.shape, "float32", zeros());
    this.built = true;
  }
  computeOutputShape(s) { return s; }
  call(inputs) {
    return tf.tidy(() => {
      const x = Array.isArray(inputs) ? inputs[0] : inputs;
      const [B, T, d] = x.shape;
      const H = this.cfg.heads, dh = d / H, eps = this.cfg.ln_eps;
      const r = n => this[n].read();
      const h = layerNorm(x, r("ln1_gamma"), r("ln1_beta"), eps);
      const heads = t => tf.transpose(tf.reshape(t, [B, T, H, dh]), [0, 2, 1, 3]);   // [B, H, T, dh]
      const q = heads(linear(h, r("wq"), r("bq")));
      const k = heads(linear(h, r("wk"), r("bk")));
      const v = heads(linear(h, r("wv"), r("bv")));
      const scores = tf.mul(tf.matMul(q, k, false, true), 1 / Math.sqrt(dh));      // [B, H, T, T]
      const ctx = tf.matMul(tf.softmax(scores), v);                                // [B, H, T, dh]
      const merged = tf.reshape(tf.transpose(ctx, [0, 2, 1, 3]), [B, T, d]);
      const x1 = tf.add(x, linear(merged, r("wo"), r("bo")));
      const h2 = layerNorm(x1, r("ln2_gamma"), r("ln2_beta"), eps);
      const f = linear(tf.relu(linear(h2, r("w1"), r("b1"))), r("w2"), r("b2"));
      return tf.add(x1, f);
    });
  }
  getConfig() { return { ...super.getConfig(), ...this.cfg }; }
}

for (const C of [Positional, LayerNorm, EncoderBlock]) tf.serialization.registerClass(C);

function makeLayer(l, first, inputShape) {
  const io = first ? { inputShape } : {};
  const init = { kernelInitializer: zeros(), biasInitializer: zeros() };
  switch (l.type) {
    case "flatten": return tfl.layers.flatten({ name: l.name, ...io });
    case "reshape": return tfl.layers.reshape({ name: l.name, targetShape: l.target_shape, ...io });
    case "dense": return tfl.layers.dense({ name: l.name, units: l.units, activation: l.activation, ...init, ...io });
    case "conv2d":
      return tfl.layers.conv2d({ name: l.name, filters: l.filters, kernelSize: l.kernel_size, padding: l.padding,
        activation: l.activation, ...init, ...io });
    case "maxpool2d": return tfl.layers.maxPooling2d({ name: l.name, poolSize: l.pool_size, ...io });
    case "simple_rnn":
      return tfl.layers.simpleRNN({ name: l.name, units: l.units, activation: l.activation,
        returnSequences: l.return_sequences, ...init, recurrentInitializer: zeros(), ...io });
    case "lstm":
      return tfl.layers.lstm({ name: l.name, units: l.units, activation: l.activation,
        recurrentActivation: l.recurrent_activation, unitForgetBias: false, returnSequences: l.return_sequences,
        ...init, recurrentInitializer: zeros(), ...io });
    case "gru":
      if (l.reset_after) throw new Error("TF.js GRU implements reset_after=false only");
      return tfl.layers.gru({ name: l.name, units: l.units, activation: l.activation,
        recurrentActivation: l.recurrent_activation, returnSequences: l.return_sequences,
        ...init, recurrentInitializer: zeros(), ...io });
    case "positional": return new Positional(l);
    case "encoder_block": return new EncoderBlock(l);
    case "layer_norm": return new LayerNorm(l);
    case "global_avg_pool_1d": return tfl.layers.globalAveragePooling1d({ name: l.name, ...io });
    default: throw new Error(`unknown layer type ${l.type}`);
  }
}

/** Build the model of one spec entry (weights all zero until loaded). */
export function buildModel(spec) {
  const model = tfl.sequential({ name: spec.name });
  spec.layers.forEach((l, i) => model.add(makeLayer(l, i === 0, spec.input_shape)));
  return model;
}

/**
 * The model's weight variables in spec order: [{layer, name, shape, variable}], checked against the
 * spec (layer by layer, weight count and shapes). Throws on any mismatch.
 */
export function weightSlots(model, spec) {
  const slots = [];
  spec.layers.forEach((l, i) => {
    const layer = model.layers[i];
    const want = l.weights ?? [];
    const have = layer.weights;
    if (have.length !== want.length) {
      throw new Error(`${spec.name}/${l.name}: ${have.length} weights, spec lists ${want.length}`);
    }
    want.forEach((w, k) => {
      const v = have[k];
      if (JSON.stringify(v.shape) !== JSON.stringify(w.shape)) {
        throw new Error(`${spec.name}/${l.name}/${w.name}: shape [${v.shape}] != spec [${w.shape}]`);
      }
      slots.push({ layer: l.name, name: w.name, shape: w.shape, variable: v });
    });
  });
  return slots;
}

/** Trainable tf.Variables in spec order (what variableGrads / applyGradients use). */
export function trainableVariables(slots) {
  return slots.map(s => s.variable.read());
}

export { tf, tfl };
