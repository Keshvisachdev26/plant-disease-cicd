import json, os, py_compile, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_python_files_compile():
    for f in ("app.py", "advice.py", "gradcam.py"):
        py_compile.compile(os.path.join(ROOT, f), doraise=True)


def test_class_names_has_38_classes():
    with open(os.path.join(ROOT, "class_names.json")) as f:
        names = json.load(f)
    assert len(names) == 38
    assert all("___" in n for n in names)


def test_model_file_present():
    path = os.path.join(ROOT, "plant_disease_mobilenet.keras")
    assert os.path.getsize(path) > 1_000_000


def test_every_condition_has_advice():
    from advice import ADVICE
    with open(os.path.join(ROOT, "class_names.json")) as f:
        names = json.load(f)
    missing = [n for n in names if n.split("___")[1].strip("_") not in ADVICE]
    assert not missing, f"No advice for: {missing}"
