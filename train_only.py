from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # or specify your pretrained model path

# Define the path to the YAML file
yaml_path = "yolo/data.yaml"


# Train the model with the new data
results = model.train(
    data=yaml_path,
    epochs=2,
    batch=16,
    imgsz=640,
    project='runs',  # Log directory
    name='detect',   # Subdirectory under project for this run
    save=True,
    cache=False,
    verbose=True
)

# Print the training results
print(results)
