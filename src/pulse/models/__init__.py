"""Models: a feature baseline, a 1D CNN, ONNX export, quantisation and a common runtime.

Two model families are kept on purpose. The feature baseline encodes domain
knowledge (speed, bearing geometry) and is cheap to run anywhere. The CNN
learns from the raw window without the speed input and is the artefact that
gets compressed for the edge. ``runtime`` gives both the same predict
interface so the benchmark and the devices do not care which one they run.
"""
