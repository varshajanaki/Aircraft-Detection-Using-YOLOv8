import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch
import gc
import shutil

# Set the working directory
ROOT_DIR = Path("e:/DRDL/YOLO/jets")
DATASET_DIR = ROOT_DIR / "military"

# GPU configuration
def setup_gpu():
    """Setup GPU device and return device configuration"""
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"\nFound {num_gpus} GPU(s)")
        
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
            print(f"GPU {i}: {gpu_name} ({gpu_mem:.2f} GB)")
        
        # Use the first GPU by default
        device = 0
        torch.cuda.set_device(device)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        
        return device
    else:
        print("\nNo GPU found, using CPU")
        return -1

def create_dataset_yaml():
    """Create a YAML file for the dataset configuration"""
    # Load class names from the existing YAML file
    with open(DATASET_DIR / "aircraft_names.yaml", "r") as f:
        data = yaml.safe_load(f)
    
    # Update paths to be absolute
    data["path"] = str(DATASET_DIR)
    data["train"] = str(DATASET_DIR / "images/aircraft_train")
    data["val"] = str(DATASET_DIR / "images/aircraft_val")
    
    # Save the updated YAML file
    yaml_path = ROOT_DIR / "aircraft_dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Dataset YAML created at {yaml_path}")
    return yaml_path

def train_model(yaml_path, model_size="n", epochs=50, batch_size=16, img_size=640, device=0):
    """Train YOLOv8 model"""
    # Initialize the model
    model = YOLO(f"yolov8{model_size}.pt")
    
    # Set up save directory
    save_dir = ROOT_DIR / "runs" / f"aircraft_yolov8{model_size}"
    
    # Train the model
    print(f"\nTraining YOLOv8{model_size} model on device {device}...")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=10,     # Early stopping patience
        save=True,       # Save checkpoints
        save_period=5,   # Save a checkpoint every 5 epochs
        device=device,   # Use selected device
        project=str(ROOT_DIR / "runs"),
        name=f"aircraft_yolov8{model_size}",
        verbose=True,
        amp=True,        # Enable mixed precision training
        cache=False,     # Disable caching to save memory
        workers=4,       # Reduce worker threads
        exist_ok=True    # Overwrite existing experiment
    )
    
    return model, results, save_dir

def validate_model(model, yaml_path, img_size=640, device=0):
    """Validate the trained model"""
    print("\nValidating model...")
    val_results = model.val(
        data=yaml_path,
        imgsz=img_size,
        batch=4,         # Reduced from 16 to 4
        device=device,   # Use selected device
        workers=0,       # Set to 0 to avoid multiprocessing issues
        cache=False,     # Disable caching to save memory
        verbose=True
    )
    
    # Extract metrics
    metrics = {
        "precision": val_results.box.maps[0],  # Precision
        "recall": val_results.box.maps[1],    # Recall
        "mAP50": val_results.box.maps[2],    # mAP at IoU 0.5
        "mAP50-95": val_results.box.map     # mAP at IoU 0.5-0.95
    }
    
    print("\nValidation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    return metrics

def plot_results(results, save_dir):
    """Plot training results"""
    # First, print available keys to help debug
    print("\nAvailable keys in results dictionary:")
    for key in results.results_dict.keys():
        print(f"  - {key}")
    
    # Create the plot with available metrics
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax = ax.flatten()
    
    # Plot loss - check if keys exist before plotting
    if 'train/box_loss' in results.results_dict and 'val/box_loss' in results.results_dict:
        ax[0].plot(results.results_dict['train/box_loss'], label='train')
        ax[0].plot(results.results_dict['val/box_loss'], label='val')
    elif 'box_loss' in results.results_dict:
        ax[0].plot(results.results_dict['box_loss'], label='loss')
    else:
        # Try to find any loss-related keys
        loss_keys = [k for k in results.results_dict.keys() if 'loss' in k.lower()]
        for key in loss_keys[:2]:  # Plot up to 2 loss keys
            ax[0].plot(results.results_dict[key], label=key)
    
    ax[0].set_title('Box Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    # Plot mAP - check if keys exist
    map_keys = [k for k in results.results_dict.keys() if 'map' in k.lower() or 'map50' in k.lower() or 'map50-95' in k.lower()]
    if 'metrics/mAP50(B)' in results.results_dict and 'metrics/mAP50-95(B)' in results.results_dict:
        ax[1].plot(results.results_dict['metrics/mAP50(B)'], label='mAP@0.5')
        ax[1].plot(results.results_dict['metrics/mAP50-95(B)'], label='mAP@0.5-0.95')
    elif len(map_keys) > 0:
        for key in map_keys[:2]:  # Plot up to 2 mAP keys
            ax[1].plot(results.results_dict[key], label=key)
    
    ax[1].set_title('mAP')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('mAP')
    ax[1].legend()
    
    # Plot precision
    prec_keys = [k for k in results.results_dict.keys() if 'prec' in k.lower()]
    if 'metrics/precision(B)' in results.results_dict:
        ax[2].plot(results.results_dict['metrics/precision(B)'])
    elif len(prec_keys) > 0:
        ax[2].plot(results.results_dict[prec_keys[0]])
    
    ax[2].set_title('Precision')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Precision')
    
    # Plot recall
    recall_keys = [k for k in results.results_dict.keys() if 'recall' in k.lower()]
    if 'metrics/recall(B)' in results.results_dict:
        ax[3].plot(results.results_dict['metrics/recall(B)'])
    elif len(recall_keys) > 0:
        ax[3].plot(results.results_dict[recall_keys[0]])
    
    ax[3].set_title('Recall')
    ax[3].set_xlabel('Epoch')
    ax[3].set_ylabel('Recall')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_results.png')
    plt.close()
    
    print(f"\nTraining plots saved to {save_dir / 'training_results.png'}")

def predict_on_test_images(model, test_dir, save_dir, conf=0.25, img_size=640, max_images=5, device=0):
    """Run prediction on test images and save results"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Get a few test images
    test_images = list(Path(test_dir).glob("*.jpg"))[:max_images]
    
    print(f"\nRunning predictions on {len(test_images)} test images...")
    for img_path in test_images:
        results = model.predict(
            source=img_path,
            conf=conf,
            save=True,
            project=str(save_dir),
            name="predictions",
            device=device,  # Use selected device
            exist_ok=True
        )
    
    print(f"Prediction results saved to {save_dir / 'predictions'}")

def save_trained_model(model, save_dir, model_size):
    """Explicitly save the trained model to ensure it's preserved"""
    save_dir = Path(save_dir)
    
    # Create a dedicated models directory if it doesn't exist
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Save the model in ONNX format (for deployment)
    onnx_path = models_dir / f"aircraft_yolov8{model_size}.onnx"
    try:
        model.export(format="onnx", imgsz=640)
        # The export saves to the same directory as the model, so we need to move it
        source_onnx = save_dir / "weights" / "best.onnx"
        if source_onnx.exists():
            shutil.copy(source_onnx, onnx_path)
            print(f"ONNX model saved to {onnx_path}")
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
    
    # Save the PyTorch model (for further training/fine-tuning)
    pt_path = models_dir / f"aircraft_yolov8{model_size}.pt"
    try:
        # Copy the best model from the training run
        source_pt = save_dir / "weights" / "best.pt"
        if source_pt.exists():
            shutil.copy(source_pt, pt_path)
            print(f"PyTorch model saved to {pt_path}")
        else:
            # If best.pt doesn't exist, save the current model state
            model.save(pt_path)
            print(f"PyTorch model saved to {pt_path}")
    except Exception as e:
        print(f"Error saving PyTorch model: {e}")
        # Fallback: try to save directly
        try:
            model.save(pt_path)
            print(f"PyTorch model saved to {pt_path} (fallback method)")
        except Exception as e2:
            print(f"Error in fallback save: {e2}")
    
    # Also save the last model as a backup
    last_pt_path = models_dir / f"aircraft_yolov8{model_size}_last.pt"
    try:
        source_last_pt = save_dir / "weights" / "last.pt"
        if source_last_pt.exists():
            shutil.copy(source_last_pt, last_pt_path)
            print(f"Last checkpoint saved to {last_pt_path}")
    except Exception as e:
        print(f"Error saving last checkpoint: {e}")
    
    return models_dir

def main():
    # Setup GPU
    device = setup_gpu()
    
    # Create dataset YAML file
    yaml_path = create_dataset_yaml()
    
    # Set training parameters
    model_size = "s"  # n (nano), s (small), m (medium), l (large), x (xlarge)
    epochs = 50
    
    # Adjust batch size based on GPU memory
    # These are recommended values, adjust based on your GPU memory
    if device >= 0:  # If using GPU
        gpu_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
        if gpu_mem > 16:  # High-end GPU (>16GB)
            batch_size = 32
        elif gpu_mem > 8:  # Mid-range GPU (8-16GB)
            batch_size = 16
        else:  # Entry-level GPU (<8GB)
            batch_size = 8
    else:  # If using CPU
        batch_size = 8
    
    img_size = 640
    
    print(f"\nTraining with batch size: {batch_size}")
    
    # Train the model
    model, results, save_dir = train_model(
        yaml_path=yaml_path,
        model_size=model_size,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        device=device
    )
    
    # Clean up memory before validation
    if device >= 0:
        torch.cuda.empty_cache()
    gc.collect()
    
    # Validate the model
    metrics = validate_model(model, yaml_path, img_size, device)
    
    # Save the model immediately after training
    models_dir = save_trained_model(model, save_dir, model_size)
    
    # Clean up memory completely
    del model
    if device >= 0:
        torch.cuda.empty_cache()
    gc.collect()
    
    # Load the saved model for validation
    model = YOLO(models_dir / f"aircraft_yolov8{model_size}.pt")
    
    # Now validate the model
    metrics = validate_model(model, yaml_path, img_size, device)
    
    # Plot training results
    plot_results(results, save_dir=save_dir)
    
    # Run predictions on test images
    predict_on_test_images(
        model=model,
        test_dir=DATASET_DIR / "images/aircraft_val",
        save_dir=save_dir,
        conf=0.25,
        img_size=img_size,
        device=device
    )
    
    # Explicitly save the trained model to ensure it's preserved
    models_dir = save_trained_model(model, save_dir, model_size)
    print(f"\nTrained models saved to {models_dir}")
    
    # Clean up GPU memory
    if device >= 0:
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\nTraining and evaluation completed successfully!")
    print(f"Your model is saved and will be available even after shutdown.")
    print(f"Main model file: {models_dir / f'aircraft_yolov8{model_size}.pt'}")

if __name__ == "__main__":
    main()