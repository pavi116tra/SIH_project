import os
import torch
from ultralytics import YOLO
from model import X3DFightClassifier

def export_fight_classifier_to_onnx(checkpoint_path="fight_detector_x3d_s.pt", output_onnx="fight_classifier.onnx"):
    """
    Exports PyTorch X3DFightClassifier checkpoint to ONNX format.
    Input shape: (batch_size=1, C=3, T=16, H=224, W=224)
    """
    print(f"Exporting PyTorch model '{checkpoint_path}' to ONNX format...")
    device = torch.device("cpu")
    model = X3DFightClassifier(pretrained=False)
    
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
            print("Loaded trained model weights.")
        except Exception as e:
            print(f"Notice loading state dict: {e}")
    else:
        print("Warning: Checkpoint not found. Exporting initialized model weights.")

    model.eval()

    # X3D-S dummy input shape: (1, 3, 16, 224, 224)
    dummy_input = torch.randn(1, 3, 16, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["video_frames"],
        output_names=["fight_logits"],
        dynamic_axes={
            "video_frames": {0: "batch_size"},
            "fight_logits": {0: "batch_size"}
        }
    )
    print(f"ONNX Fight Classifier exported successfully to '{output_onnx}'")

def export_yolo_to_onnx(yolo_path="yolov8n.pt"):
    """
    Exports Ultralytics YOLO model to ONNX format natively.
    """
    print(f"Exporting YOLO model '{yolo_path}' to ONNX...")
    model = YOLO(yolo_path)
    onnx_file = model.export(format="onnx")
    print(f"YOLO ONNX exported to '{onnx_file}'")
    return onnx_file

def run_export_pipeline():
    export_fight_classifier_to_onnx()
    try:
        export_yolo_to_onnx()
    except Exception as e:
        print(f"YOLO export note: {e}")

if __name__ == "__main__":
    run_export_pipeline()
