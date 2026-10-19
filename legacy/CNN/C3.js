import { Cifar10Data } from '../data_cifar10.js';

let logDetails = {};

async function measureLoadTime(func, name) {
  const startTime = performance.now();
  await func();
  const endTime = performance.now();
  logDetails[name + 'LoadingTime'] = endTime - startTime;
}

async function logPerformanceMetrics(model, data) {
  const warmUpStart = performance.now();
  const testBatch = data.nextTestBatch(1);
  model.predict(testBatch.xs.reshape([1, 32, 32, 3]));
  const warmUpEnd = performance.now();
  logDetails.warmUpTime = warmUpEnd - warmUpStart;

  const trainingStart = performance.now();
  await train(model, data);
  const trainingEnd = performance.now();
  logDetails.trainingTime = trainingEnd - trainingStart;

  const profile = await tf.profile(async () => {
    const output = model.predict(testBatch.xs.reshape([1, 32, 32, 3]));
    await output.data();
  });

  const kernelMs = profile.kernels
    .map(kernel => kernel.kernelMs)
    .filter(time => !isNaN(time))
    .reduce((a, b) => a + b, 0);
  logDetails.totalKernelTime = kernelMs;

  logDetails.GPUpeakBytes = profile.peakBytes || 'Not available in this environment.';

  const topKernels = profile.kernels
    .filter(kernel => kernel && kernel.name && !isNaN(kernel.kernelMs))
    .sort((a, b) => b.kernelMs - a.kernelMs)
    .slice(0, 10)
    .map(kernel => ({
      name: kernel.name,
      kernelMs: kernel.kernelMs
    }));
  logDetails.topKernels = topKernels;

  if (profile.tensorMap) {
    for (const tensorInfo of Object.values(profile.tensorMap)) {
      if (tensorInfo.tensor) {
        tensorInfo.tensor.dispose();
      }
    }
  }

  let totalInferenceTime = 0;
  for (let i = 0; i < 100; i++) {
    const inferenceStart = performance.now();
    model.predict(testBatch.xs.reshape([1, 32, 32, 3]));
    const inferenceEnd = performance.now();
    totalInferenceTime += (inferenceEnd - inferenceStart);
  }
  logDetails.averageInferenceTime = (totalInferenceTime / 100).toFixed(6);

  const predictions = model.predict(testBatch.xs.reshape([1, 32, 32, 3]));
  const predictionData = Array.from(await predictions.data());
  logDetails.predictions = predictionData;
}

async function showExamples(data) {
  const surface = tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data' });
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];

  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([32, 32, 3]);
    });

    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 32;
    canvas.style = 'margin: 4px;';
    //await tf.browser.toPixels(imageTensor, canvas); // WebGL & WASM
    await tf.browser.draw(imageTensor, canvas); // WebGPU & CPU
    surface.drawArea.appendChild(canvas);
    imageTensor.dispose();
  }
}

async function run() {
  await tf.setBackend('webgpu');
  //tf.wasm.setThreadsCount(8);
  //await tf.ready();
  //console.log('Thread count:', tf.wasm.getThreadsCount());

  const data = new Cifar10Data();
  const model = getModel();

  await measureLoadTime(async () => {
    await data.load();
    await showExamples(data);
  }, "CIFAR-10 data");

  await measureLoadTime(async () => {
    logDetails.modelArchitecture = JSON.parse(model.toJSON());
  }, "Model");

  //await model.save('downloads://C3_model');
  //console.log('Model saved to downloads.');

  await logPerformanceMetrics(model, data);
  await showAccuracy(model, data);
  await showConfusion(model, data);

  logDetails.backend = tf.getBackend();
  logDetails.timestamp = new Date().toISOString();

  const baseFileName = window.location.pathname.split('/').pop().split('.')[0] || "logfile";
  const fileName = baseFileName + '_' + new Date().toISOString().replace(/[:]/g, '-') + '.json';
  downloadJSON(logDetails, fileName);

  window.operationComplete = true;
}

document.addEventListener('DOMContentLoaded', run);

function getModel() {
  const model = tf.sequential();

  // First Conv2D layer
  model.add(tf.layers.conv2d({
    inputShape: [32, 32, 3],
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

  // Second Conv2D layer
  model.add(tf.layers.conv2d({
    filters: 128,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

  // Flatten before Dense layers
  model.add(tf.layers.flatten());

  // 3 hidden Dense layers with 256 neurons
  model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 256, activation: 'relu' }));

  // Output layer
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  // Compile model
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

async function train(model, data) {
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = { name: 'Model Training', tab: 'Model', styles: { height: '1000px' } };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

//  const BATCH_SIZE = 512;
// const TRAIN_DATA_SIZE = 40000;
//  const TEST_DATA_SIZE = 10000;

  const BATCH_SIZE = 64;
  const TRAIN_DATA_SIZE = 5000;
  const TEST_DATA_SIZE = 1000;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [
      d.xs.reshape([TRAIN_DATA_SIZE, 32, 32, 3]),
      d.labels
    ];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [
      d.xs.reshape([TEST_DATA_SIZE, 32, 32, 3]),
      d.labels
    ];
  });

  const history = await model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks
  });

  const finalTrainAccuracy = history.history.acc[history.history.acc.length - 1];
  const finalValAccuracy = history.history.val_acc[history.history.val_acc.length - 1];

  logDetails.finalTrainingAccuracy = (finalTrainAccuracy * 100).toFixed(2) + '%';
  logDetails.finalValidationAccuracy = (finalValAccuracy * 100).toFixed(2) + '%';

  trainXs.dispose();
  trainYs.dispose();
  testXs.dispose();
  testYs.dispose();
}

const classNames = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'];

function doPrediction(model, data, testDataSize = 500) {
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, 32, 32, 3]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return Promise.all([preds.data(), labels.data()]);
}

async function showAccuracy(model, data) {
  const [predsArray, labelsArray] = await doPrediction(model, data);

  const preds = tf.tensor1d(predsArray, 'int32');
  const labels = tf.tensor1d(labelsArray, 'int32');

  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  tfvis.show.perClassAccuracy({ name: 'Accuracy', tab: 'Evaluation' }, classAccuracy, classNames);

  preds.dispose();
  labels.dispose();
}

async function showConfusion(model, data) {
  const [predsArray, labelsArray] = await doPrediction(model, data);

  const preds = tf.tensor1d(predsArray, 'int32');
  const labels = tf.tensor1d(labelsArray, 'int32');

  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  tfvis.render.confusionMatrix({ name: 'Confusion Matrix', tab: 'Evaluation' },
    { values: confusionMatrix, tickLabels: classNames });

  preds.dispose();
  labels.dispose();
}

function downloadJSON(data, filename) {
  const jsonStr = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonStr], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}
