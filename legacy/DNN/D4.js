import {Cifar10Data} from '../data_cifar10.js';

// Global object to collect all logs and metrics.
let logDetails = {};

async function measureLoadTime(func, name) {
  const startTime = performance.now();
  await func();
  const endTime = performance.now();
  // Save loading time for the given resource.
  logDetails[name + 'LoadingTime'] = endTime - startTime;
}

async function logPerformanceMetrics(model, data) {
  // Warm-up time.
  const warmUpStart = performance.now();
  const testBatch = data.nextTestBatch(1);
  model.predict(testBatch.xs.reshape([1, 32, 32, 3]));
  const warmUpEnd = performance.now();
  logDetails.warmUpTime = warmUpEnd - warmUpStart;

  // Training time.
  const trainingStart = performance.now();
  await train(model, data);
  const trainingEnd = performance.now();
  logDetails.trainingTime = trainingEnd - trainingStart;

  // Profiling kernel times.
  const profile = await tf.profile(async () => {
    const output = model.predict(testBatch.xs.reshape([1, 32, 32, 3]));
    // Force computation by awaiting the data.
    await output.data();
  });

  const kernelMs = profile.kernels
    .map(kernel => kernel.kernelMs)
    .filter(time => !isNaN(time))
    .reduce((a, b) => a + b, 0);
  logDetails.totalKernelTime = kernelMs;

  if (profile.peakBytes) {
    logDetails.GPUpeakBytes = profile.peakBytes;
  } else {
    logDetails.GPUpeakBytes = 'Not available in this environment.';
  }
  
  // --- NEW CODE: Get the top 10 kernel functions ---
  const topKernels = profile.kernels
    .filter(kernel => kernel && kernel.name && !isNaN(kernel.kernelMs))
    .sort((a, b) => b.kernelMs - a.kernelMs)
    .slice(0, 10)
    .map(kernel => ({
      name: kernel.name,
      kernelMs: kernel.kernelMs
    }));
  logDetails.topKernels = topKernels;

  // Dispose tensors from the profile.
  if (profile.tensorMap) {
    for (const tensorInfo of Object.values(profile.tensorMap)) {
      if (tensorInfo.tensor) {
        tensorInfo.tensor.dispose();
      }
    }
  }

  // Measure inference time over 100 runs.
  let totalInferenceTime = 0;
  for (let i = 0; i < 100; i++) {
    const inferenceStart = performance.now();
    model.predict(testBatch.xs.reshape([1, 32, 32, 3]));
    const inferenceEnd = performance.now();
    totalInferenceTime += (inferenceEnd - inferenceStart);
  }
  const averageInferenceTime = totalInferenceTime / 100;
  logDetails.averageInferenceTime = averageInferenceTime.toFixed(6);

  // Save predictions.
  const predictions = model.predict(testBatch.xs.reshape([1, 32, 32, 3]));
  const predictionData = Array.from(await predictions.data());
  logDetails.predictions = predictionData;
}

async function showExamples(data) {
  const surface = tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  
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
    await tf.browser.toPixels(imageTensor, canvas); // WebGL & WASM
    //await tf.browser.draw(imageTensor, canvas); // WebGPU & CPU
    surface.drawArea.appendChild(canvas);
    imageTensor.dispose();
  }
}

async function run() {
  await tf.setBackend('webgl');
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
    // Instead of displaying the model summary, we save the architecture.
    logDetails.modelArchitecture = JSON.parse(model.toJSON());
  }, "Model");

  await logPerformanceMetrics(model, data);

  await showAccuracy(model, data);
  await showConfusion(model, data);

  // Save additional details.
  logDetails.backend = tf.getBackend();
  logDetails.timestamp = new Date().toISOString();

  // Generate the JSON file name based on the running file name and current timestamp.
  const baseFileName = window.location.pathname.split('/').pop().split('.')[0] || "logfile";
  const fileName = baseFileName + '_' + new Date().toISOString().replace(/[:]/g, '-') + '.json';

  // Trigger download of the JSON file.
  downloadJSON(logDetails, fileName);

  window.operationComplete = true;
}

document.addEventListener('DOMContentLoaded', run);

function getModel() {
  const model = tf.sequential();

  // Flatten the input image (32x32 with 3 channels = 3072 features)
  model.add(tf.layers.flatten({
    inputShape: [32, 32, 3]
  }));

  // First Dense hidden layer with 128 units
  model.add(tf.layers.dense({
    units: 512,
    activation: 'relu',
    kernelInitializer: 'glorotUniform'
  }));

  // Second Dense hidden layer with 128 units
  model.add(tf.layers.dense({
    units: 512,
    activation: 'relu',
    kernelInitializer: 'glorotUniform'
  }));

  // Third Dense hidden layer with 128 units
  model.add(tf.layers.dense({
    units: 512,
    activation: 'relu',
    kernelInitializer: 'glorotUniform'
  }));

  // Fourth Dense hidden layer with 128 units
  model.add(tf.layers.dense({
    units: 512,
    activation: 'relu',
    kernelInitializer: 'glorotUniform'
  }));

  // Output layer: 10 classes
  const NUM_OUTPUT_CLASSES = 10;
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    activation: 'softmax',
    kernelInitializer: 'glorotUniform'
  }));

  // Compile the model
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}


async function train(model, data) {
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = {
    name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
  };
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

  // Save the final training and validation accuracy.
  const finalTrainAccuracy = history.history.acc[history.history.acc.length - 1];
  const finalValAccuracy = history.history.val_acc[history.history.val_acc.length - 1];

  logDetails.finalTrainingAccuracy = (finalTrainAccuracy * 100).toFixed(2) + '%';
  logDetails.finalValidationAccuracy = (finalValAccuracy * 100).toFixed(2) + '%';

  // Dispose training tensors.
  trainXs.dispose();
  trainYs.dispose();
  testXs.dispose();
  testYs.dispose();
}

const classNames = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'];

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 32;
  const IMAGE_HEIGHT = 32;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 3]);
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
  const blob = new Blob([jsonStr], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}
