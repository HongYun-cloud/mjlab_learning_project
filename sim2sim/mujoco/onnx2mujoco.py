import onnxruntime as ort

onnx_path = "./model/go2_best.onnx"

session = ort.InferenceSession(onnx_path)

print(session.get_inputs())
print(session.get_outputs())