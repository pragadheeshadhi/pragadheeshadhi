from ultralytics import YOLO

if __name__ == "__main__":
    # Initialize the model with a pre-defined configuration
    model = YOLO("yolov8n.pt")  # Use a model like yolov8n, yolov8s, etc.

    # Train the model using your dataset configuration and specify the output directory
    model.train(
        data=r"c:\PROJECT\YOLO datasets\Umpire-NonUmpire2.v1i.yolov8\data.yaml",  # Dataset configuration
        epochs=200,
        batch=16,
        imgsz=640,
        project=r"C:\PROJECT\YOLO v8 datasets\Umpire-NonUmpire2.v1i.yolov8",  # Path to save the training results
        name="Results"  # Name of the folder under the project directory
    )
