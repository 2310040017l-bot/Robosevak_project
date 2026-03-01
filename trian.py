from ultralytics import YOLO

# 1. Load a pretrained 'Nano' model (The "Existing Brain")
model = YOLO('yolov8n.pt') 

# 2. Start the Training
# 'data' points to the yaml file that lists where your Laboro images are
# 'epochs' is how many times the model reads the whole textbook (dataset)
# 'imgsz' is the resolution the model sees (640x640 pixels)
results = model.train(
    data='laboro_tomato.yaml', 
    epochs=50, 
    imgsz=640, 
    batch=16, 
    device=0  # Uses your HP Victus GPU
)
