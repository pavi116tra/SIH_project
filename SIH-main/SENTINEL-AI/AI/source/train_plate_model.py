from ultralytics import YOLO
import os
import sys

def main():
    print("Initializing YOLOv8 training for license plate detection...")
    
    # Use the base yolov8 nano model for fastest inference speed
    model = YOLO("yolov8n.pt")
    
    # Path to the data.yaml file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(base_dir, "dataset_lp", "data.yaml")
    
    if not os.path.exists(data_yaml):
        print(f"Error: Could not find dataset config at {data_yaml}")
        sys.exit(1)
        
    print(f"Found dataset config at: {data_yaml}")
    print("Starting training process (50 epochs)...")
    
    # Train the model
    # Results will be saved to runs/detect/license_plate_detector
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        name="license_plate_detector",
        project="runs/detect",
        exist_ok=True, # Overwrite if it already exists to keep things clean
        batch=16,      # Standard batch size
        device=0       # Use GPU if available
    )
    
    print("\nTraining completed successfully!")
    print(f"Best model saved to: runs/detect/license_plate_detector/weights/best.pt")

if __name__ == "__main__":
    main()
