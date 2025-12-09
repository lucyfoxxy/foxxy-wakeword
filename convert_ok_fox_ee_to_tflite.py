"""Convert ok_fox_ee.onnx into multiple TFLite variants."""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import onnx
import tensorflow as tf


@dataclass
class ConversionConfig:
    """Configuration for a TFLite conversion run."""

    suffix: str
    use_select_tf_ops: bool
    use_float16: bool


def load_onnx_parameters(onnx_path: Path) -> Dict[str, np.ndarray]:
    """Load initializer tensors from an ONNX model."""
    model = onnx.load(onnx_path)
    return {
        tensor.name: onnx.numpy_helper.to_array(tensor).astype(np.float32)
        for tensor in model.graph.initializer
    }


def build_model(params: Dict[str, np.ndarray]) -> tf.keras.Model:
    """Recreate the simple MLP defined in ok_fox_ee.onnx and load weights."""
    inputs = tf.keras.Input(shape=(16, 96), name="onnx__Flatten_0", dtype=tf.float32)
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

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ok_fox_ee")

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


def convert_model_to_tflite(model: tf.keras.Model, output_path: Path, config: ConversionConfig) -> None:
    print(
        f"Converting to {output_path.name} (select_tf_ops={config.use_select_tf_ops}, float16={config.use_float16})..."
    )
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    supported_ops: List[tf.lite.OpsSet] = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if config.use_select_tf_ops:
        supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)
    converter.target_spec.supported_ops = supported_ops

    if config.use_float16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    print(f"Saved {output_path}")


def convert_variants(model: tf.keras.Model, output_dir: Path, configs: Iterable[ConversionConfig]) -> None:
    for config in configs:
        output_path = output_dir / f"ok_fox_ee_{config.suffix}.tflite"
        convert_model_to_tflite(model, output_path, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ok_fox_ee.onnx into multiple TFLite variants.")
    parser.add_argument("--onnx", type=Path, default=Path("ok_fox_ee.onnx"), help="Path to the ONNX model")
    parser.add_argument("--output-dir", type=Path, default=Path("build/ok_fox_ee"), help="Directory for TFLite outputs")
    args = parser.parse_args()

    params = load_onnx_parameters(args.onnx)
    model = build_model(params)

    configs = [
        ConversionConfig(suffix="fp32_builtin", use_select_tf_ops=False, use_float16=False),
        ConversionConfig(suffix="fp32_select", use_select_tf_ops=True, use_float16=False),
        ConversionConfig(suffix="fp16_builtin", use_select_tf_ops=False, use_float16=True),
        ConversionConfig(suffix="fp16_select", use_select_tf_ops=True, use_float16=True),
    ]

    convert_variants(model, args.output_dir, configs)


if __name__ == "__main__":
    main()
