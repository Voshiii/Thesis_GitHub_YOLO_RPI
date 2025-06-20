from ultralytics import YOLO

# Select model
model = YOLO("yolo11n.pt")

# Export the model
model.export(format="imx")  # exports with PTQ quantization by default


# To run test inference:
# Load the exported model
#imx_model = YOLO("yolo11n_imx_model")

# Run inference
#results = imx_model("https://ultralytics.com/images/bus.jpg")