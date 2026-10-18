import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
backend = HERE / "backend.py"
requirements = HERE / "requirements_gpu.txt"
server = HERE / "inference_server.py"
client = HERE / "inference_client.py"

print("===== OPENVLA GPU DEPLOYMENT FILE CHECK =====")

for path in (backend, requirements, server, client):
    assert path.exists(), f"Missing: {path}"
    print("[PASS]", path.name)

tree = ast.parse(backend.read_text(encoding="utf-8"))
classes = {
    node.name: node
    for node in tree.body
    if isinstance(node, ast.ClassDef)
}

assert "RealOpenVLA" in classes
assert "MockOpenVLA" in classes
print("[PASS] RealOpenVLA and MockOpenVLA exist")

methods = {
    node.name
    for node in classes["RealOpenVLA"].body
    if isinstance(node, ast.FunctionDef)
}

assert "__init__" in methods
assert "predict" in methods
print("[PASS] RealOpenVLA loading and prediction methods exist")

source = backend.read_text(encoding="utf-8")
assert "openvla/openvla-7b" in source
assert "bridge_orig" in source
print("[PASS] Model and normalization configuration present")

print("\nALL DEPLOYMENT FILE CHECKS PASSED")
print("GPU package compatibility: NOT VERIFIED")
print("Model download and loading: NOT TESTED")
print("Real inference: NOT TESTED")
