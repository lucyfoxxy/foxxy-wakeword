import sys
import importlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import tensor_dtype_to_np_dtype

# Ensure onnx.mapping compatibility for onnx-tf
if not hasattr(onnx, "mapping") and hasattr(onnx, "_mapping"):
    onnx.mapping = onnx._mapping  # type: ignore[attr-defined]

# Populate legacy mapping expected by onnx-tf
if hasattr(onnx, "_mapping"):
    if not hasattr(onnx._mapping, "TENSOR_TYPE_TO_NP_TYPE"):
        mapping_dict = {}
        for dtype in TensorProto.DataType.values():
            try:
                mapping_dict[dtype] = np.dtype(tensor_dtype_to_np_dtype(dtype))
            except Exception:
                continue
        onnx._mapping.TENSOR_TYPE_TO_NP_TYPE = mapping_dict  # type: ignore[attr-defined]

# Make sure onnx.helper exposes mapping attribute expected by onnx-tf
helper = importlib.import_module("onnx.helper")
if not hasattr(helper, "mapping") and hasattr(onnx, "_mapping"):
    setattr(helper, "mapping", onnx._mapping)
    sys.modules["onnx.helper"] = helper

from onnx_tf.backend import prepare
import tensorflow as tf

MODEL_PATH = Path('okay_foxxy.onnx')
SAVED_MODEL_DIR = Path('build/okay_foxxy_saved_model')
TFLITE_PATH = Path('okay_foxxy.tflite')

print('Loading ONNX model...')
onnx_model = onnx.load(MODEL_PATH)
print('Converting ONNX to TensorFlow SavedModel...')
tf_rep = prepare(onnx_model)
SAVED_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
tf_rep.export_graph(str(SAVED_MODEL_DIR))

print('Converting SavedModel to TFLite...')
converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_DIR))
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tflite_model = converter.convert()

TFLITE_PATH.write_bytes(tflite_model)
print(f'TFLite model written to {TFLITE_PATH}')
