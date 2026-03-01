import os
from ultralytics import YOLO

# 1. Load your trained model weights
# Using the path you provided from your 'results' directory
model_path = 'results/tomato_run/weights/best.pt'
model = YOLO(model_path)

# 2. Define the source folder containing your videos
video_folder = 'viddata'

# 3. Check if the folder exists before running
if not os.path.exists(video_folder):
    print(f"Error: The folder '{video_folder}' was not found!")
else:
    print(f"🚀 Starting inference on videos in: {video_folder}")
    
    # 4. Run prediction on the entire folder
    # save=True: saves the annotated video
    # conf=0.25: standard confidence threshold for testing
    # project='results': saves the output in your results folder for better organization
    # name='viddata_predictions': names the specific output subfolder
    results = model.predict(
        source=video_folder,
        save=True,
        conf=0.25,
        project='results',
        name='viddata_predictions',
        exist_ok=True  # Overwrites if you run it multiple times
    )

    print("\n✅ Processing Complete!")
    print(f"Find your annotated videos in: results/viddata_predictions/")
