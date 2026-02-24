from ultralytics import YOLO

# Load a model
model = YOLO("/home/aiserver/Desktop/ffmpeg_stream/ppe.pt")  # load a custom trained model

# Export the model
model.export(format="engine", dynamic=True)