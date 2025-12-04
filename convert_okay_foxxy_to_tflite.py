import argparse
import pathlib
from typing import Dict

import numpy as np
import onnx
import tensorflow as tf


def load_onnx_parameters(onnx_path: pathlib.Path) -> Dict[str, np.ndarray]:
    """Load initializer tensors from an ONNX model."""
    model = onnx.load(onnx_path)
    return {
        tensor.name: onnx.numpy_helper.to_array(tensor).astype(np.float32)
        for tensor in model.graph.initializer
    }


def build_model(params: Dict[str, np.ndarray]) -> tf.keras.Model:
    """Recreate the simple MLP used in the ONNX graph and load weights."""
    inputs = tf.keras.Input(shape=(16, 96), name="features", dtype=tf.float32)
    x = tf.keras.layers.Reshape((16 * 96,), name="flatten")(inputs)

    dense1 = tf.keras.layers.Dense(32, use_bias=True, name="layer1")
    ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm1")
    dense2 = tf.keras.layers.Dense(32, use_bias=True, name="blocks.0.fcn_layer")
    ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="blocks.0.layer_norm")
    dense3 = tf.keras.layers.Dense(1, activation="sigmoid", use_bias=True, name="last_layer")

    x = dense1(x)
    x = ln1(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)
    x = dense2(x)
    x = ln2(x)
    x = tf.keras.layers.ReLU(name="blocks.0.relu")(x)
    outputs = dense3(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="okay_foxxy")

    # Load weights (transposed because Gemm used transB=1)
    dense1.set_weights([
        params["layer1.weight"].T,
        params["layer1.bias"],
    ])
    ln1.set_weights([
        params["layernorm1.weight"],
        params["layernorm1.bias"],
    ])
    dense2.set_weights([
        params["blocks.0.fcn_layer.weight"].T,
        params["blocks.0.fcn_layer.bias"],
    ])
    ln2.set_weights([
        params["blocks.0.layer_norm.weight"],
        params["blocks.0.layer_norm.bias"],
    ])
    dense3.set_weights([
        params["last_layer.weight"].T,
        params["last_layer.bias"],
    ])

    return model


def convert_to_tflite(model: tf.keras.Model, output_path: pathlib.Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)


def verify_against_onnx(model: tf.keras.Model, onnx_path: pathlib.Path) -> float:
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    sample = rng.standard_normal((1, 16, 96), dtype=np.float32)

    tf_output = model(sample, training=False).numpy()

    sess = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    onnx_output = sess.run(None, {sess.get_inputs()[0].name: sample})[0]

    return float(np.max(np.abs(tf_output - onnx_output)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert okay_foxxy.onnx to TFLite.")
    parser.add_argument("--onnx", type=pathlib.Path, default=pathlib.Path("okay_foxxy.onnx"))
    parser.add_argument("--tflite", type=pathlib.Path, default=pathlib.Path("okay_foxxy.tflite"))
    args = parser.parse_args()

    params = load_onnx_parameters(args.onnx)
    model = build_model(params)

    # Optional sanity check: compare to ONNX runtime output if available.
    try:
        max_diff = verify_against_onnx(model, args.onnx)
        print(f"ONNX vs TF max abs diff: {max_diff:.6f}")
    except Exception as exc:  # pragma: no cover - fallback when onnxruntime missing
        print(f"Skipping ONNX parity check: {exc}")

    convert_to_tflite(model, args.tflite)
    print(f"Saved TFLite model to {args.tflite}")


if __name__ == "__main__":
    main()
