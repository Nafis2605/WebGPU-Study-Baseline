import os
import sys
import json
import time
import datetime
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_cifar10 import Cifar10Data

log_details = {}

CLASS_NAMES     = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                   'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
BATCH_SIZE      = 64
TRAIN_DATA_SIZE = 5000
TEST_DATA_SIZE  = 1000
EPOCHS          = 10


def enforce_cuda():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError(
            "No CUDA GPU found. This script requires a CUDA-capable GPU. "
            "CPU fallback is disabled."
        )
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"CUDA GPU enforced: {[g.name for g in gpus]}")


def measure_load_time(func, name):
    start = time.perf_counter() * 1000
    func()
    end = time.perf_counter() * 1000
    log_details[name + 'LoadingTime'] = end - start


def _get_top_kernels(model, test_input):
    layer_times = []
    x = tf.constant(test_input)
    for layer in model.layers:
        start = time.perf_counter() * 1000
        x = layer(x)
        _ = x.numpy()
        end = time.perf_counter() * 1000
        layer_times.append({'name': layer.name, 'kernelMs': round(end - start, 6)})
    return sorted(layer_times, key=lambda k: k['kernelMs'], reverse=True)[:10]


def log_performance_metrics(model, data):
    test_batch = data.next_test_batch(1)
    test_input = test_batch['xs'].reshape(1, 32, 32, 3)

    warm_up_start = time.perf_counter() * 1000
    pred = model(test_input, training=False)
    _ = pred.numpy()
    warm_up_end = time.perf_counter() * 1000
    log_details['warmUpTime'] = warm_up_end - warm_up_start

    training_start = time.perf_counter() * 1000
    train(model, data)
    training_end = time.perf_counter() * 1000
    log_details['trainingTime'] = training_end - training_start

    try:
        tf.config.experimental.reset_memory_stats('GPU:0')
    except Exception:
        pass

    kernel_start = time.perf_counter() * 1000
    output = model(test_input, training=False)
    _ = output.numpy()
    kernel_end = time.perf_counter() * 1000
    log_details['totalKernelTime'] = kernel_end - kernel_start

    try:
        mem_info = tf.config.experimental.get_memory_info('GPU:0')
        log_details['GPUpeakBytes'] = int(mem_info.get('peak', 0))
    except Exception:
        log_details['GPUpeakBytes'] = 'Not available in this environment.'

    log_details['topKernels'] = _get_top_kernels(model, test_input)

    total_inference_time = 0.0
    for _ in range(100):
        inf_start = time.perf_counter() * 1000
        out = model(test_input, training=False)
        _ = out.numpy()
        inf_end = time.perf_counter() * 1000
        total_inference_time += inf_end - inf_start
    log_details['averageInferenceTime'] = f"{total_inference_time / 100:.6f}"

    predictions = model(test_input, training=False)
    log_details['predictions'] = predictions.numpy().flatten().tolist()


def get_model():
    model = tf.keras.Sequential([
        # First Conv2D block
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same',
                               input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Second Conv2D block
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Dense head — 3 hidden layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train(model, data):
    train_batch = data.next_train_batch(TRAIN_DATA_SIZE)
    train_xs    = train_batch['xs'].reshape(TRAIN_DATA_SIZE, 32, 32, 3)
    train_ys    = train_batch['labels']

    test_batch = data.next_test_batch(TEST_DATA_SIZE)
    test_xs    = test_batch['xs'].reshape(TEST_DATA_SIZE, 32, 32, 3)
    test_ys    = test_batch['labels']

    history = model.fit(
        train_xs, train_ys,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(test_xs, test_ys),
        shuffle=True,
        verbose=1
    )

    acc_key     = 'accuracy'     if 'accuracy'     in history.history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
    final_train_acc = history.history[acc_key][-1]
    final_val_acc   = history.history[val_acc_key][-1]
    log_details['finalTrainingAccuracy']   = f"{final_train_acc * 100:.2f}%"
    log_details['finalValidationAccuracy'] = f"{final_val_acc   * 100:.2f}%"


def do_prediction(model, data, test_data_size=500):
    test_batch = data.next_test_batch(test_data_size)
    test_xs    = test_batch['xs'].reshape(test_data_size, 32, 32, 3)
    labels     = np.argmax(test_batch['labels'], axis=1)
    preds      = np.argmax(model.predict(test_xs, verbose=0), axis=1)
    return preds, labels


def show_accuracy(model, data):
    preds, labels = do_prediction(model, data)
    class_accuracy = []
    for cls in range(10):
        mask = labels == cls
        acc  = float((preds[mask] == labels[mask]).mean()) if mask.sum() > 0 else 0.0
        class_accuracy.append({'label': CLASS_NAMES[cls], 'accuracy': acc})
    log_details['classAccuracy'] = class_accuracy


def show_confusion(model, data):
    preds, labels = do_prediction(model, data)
    cm = np.zeros((10, 10), dtype=int)
    for p, l in zip(preds, labels):
        cm[l][p] += 1
    log_details['confusionMatrix'] = cm.tolist()
    log_details['tickLabels']      = CLASS_NAMES


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Log saved to: {filename}")


def run():
    enforce_cuda()

    data  = Cifar10Data()
    model = get_model()

    measure_load_time(data.load, "CIFAR-10 data")
    measure_load_time(
        lambda: log_details.update({'modelArchitecture': json.loads(model.to_json())}),
        "Model"
    )

    log_performance_metrics(model, data)
    show_accuracy(model, data)
    show_confusion(model, data)

    log_details['backend']   = 'CUDA'
    log_details['timestamp'] = datetime.datetime.now().isoformat()

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp   = log_details['timestamp'].replace(':', '-')
    filename    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f"{script_name}_{timestamp}.json")
    save_json(log_details, filename)


if __name__ == '__main__':
    run()
