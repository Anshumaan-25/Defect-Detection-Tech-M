"""Defect Detection — modular AI defect-detection pipeline.

Submodules
----------
config              Central paths + dataset registry loading.
dataset_loader      Download & organize open-source datasets.
train_yolo          Train the YOLO detector (missing components / packaging).
inference_pipeline  Unified local inference across all four defect types.
"""

__version__ = "0.1.0"
