from skimage import io
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os
import tensorflow as tf
from huggingface_hub import from_pretrained_keras

# Assuming cnn_evaluation.py and cnn_plot.py are in the same directory or accessible via PYTHONPATH
from cnn_evaluation import evaluate_thresholds_cnn
from cnn_plot import plot_cnn_comprehensive_results

# --- Image Preprocessing ---
ROWS, COLS = 150, 150
IMAGE_BASE_PATH_KAGGLE = Path('datasets/kaggle') # Base path for the 500 cat/dog images

def preprocess_image_for_model(image_input_HWC_or_HW): 
    image_to_process = image_input_HWC_or_HW.copy() 
    if image_to_process.dtype != np.uint8:
        if np.issubdtype(image_to_process.dtype, np.floating) and image_to_process.max() <= 1.0:
             image_to_process = (image_to_process * 255).astype(np.uint8)
        else: image_to_process = image_to_process.astype(np.uint8)
    if image_to_process.ndim == 2: image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_GRAY2RGB)
    elif image_to_process.ndim == 3 and image_to_process.shape[2] == 4: image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_RGBA2RGB)
    if not (image_to_process.ndim == 3 and image_to_process.shape[2] == 3): raise ValueError(f"Image for preprocessing not 3-channel RGB. Shape: {image_to_process.shape}")
    # OpenCV dsize is (width, height), so COLS, ROWS
    resized_image = cv2.resize(image_to_process, (COLS, ROWS), interpolation=cv2.INTER_CUBIC)
    normalized_image = resized_image / 255.0
    return normalized_image.reshape(1, ROWS, COLS, 3)


def cnn_fixed_evaluation_set_loader(dummy_df_trigger, num_images_per_class=500): # dummy_df_trigger is not used for paths
    """
    Loads a fixed set of 500 cat and 500 dog images for evaluation.
    """
    loaded_images = []
    loaded_labels = []
    
    print(f"cnn_fixed_evaluation_set_loader: Loading {num_images_per_class} cats and {num_images_per_class} dogs...")
    
    for label_idx, label_name in enumerate(['cat', 'dog']):
        print(f"  Loading {label_name} images...")
        for img_idx in range(num_images_per_class): # From 0 to 499
            img_path = IMAGE_BASE_PATH_KAGGLE / label_name / f"{img_idx}.jpg"
            true_label = label_idx # 0 for cat, 1 for dog

            if not img_path.exists(): # Use Path object's exists() method
                print(f"    Image not found, skipping: {img_path}")
                continue
            try:
                raw_image = io.imread(str(img_path)) # io.imread expects string path
                if raw_image is None: 
                    print(f"    io.imread returned None for {img_path}. Skipping.")
                    continue

                if raw_image.ndim == 4 and raw_image.shape[0] == 1:
                    raw_image = np.squeeze(raw_image, axis=0)
                if not (raw_image.ndim == 3 or raw_image.ndim == 2): 
                    print(f"    Image {img_path} has unsupported ndim ({raw_image.ndim}) after squeeze. Skipping.")
                    continue

                processed_img = preprocess_image_for_model(raw_image)
                loaded_images.append(processed_img)
                loaded_labels.append(true_label)
            except Exception as e:
                print(f"    Error loading/processing evaluation image {img_path}: {e}. Skipping.")
    
    if not loaded_images:
        print("CNN Fixed Evaluation Set Loader: No images were successfully loaded. Check paths and image files.")
        
    return loaded_images, loaded_labels


# --- Main Pruning Script ---
if __name__ == "__main__":
    MODEL_NAME_HF = "carlosaguayo/cats_vs_dogs"
    RUN_FOLDER_NAME = "cats_vs_dogs_pruned_full_eval_set" 
    
    # Define thresholds from 0.0 to 10.0 in steps of 0.1
    # These values will be directly used as percentages in prune_keras_model_weights
    pruning_thresholds_0_to_10 = np.round(np.arange(0.0, 10.0 + 0.1, 0.1), 1).tolist()

    run_path = Path("runs") / RUN_FOLDER_NAME
    run_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading original Keras CNN model: {MODEL_NAME_HF}")
    try:
        original_cnn_model = from_pretrained_keras(MODEL_NAME_HF)
        # original_cnn_model.summary() 
    except Exception as e:
        print(f"Error loading original model: {e}")
        exit()

    # The dataset_df passed to evaluate_thresholds_cnn is just a trigger for cnn_fixed_evaluation_set_loader.
    # Its content doesn't matter as the loader loads a fixed set.
    dummy_df_for_loader_trigger = pd.DataFrame({'placeholder': [0]}) 

    print("\nStarting evaluation across pruning thresholds for CNN using the full 500+500 image set for each...")
    results_cnn_df = evaluate_thresholds_cnn(
        image_dataset_loader_func=cnn_fixed_evaluation_set_loader, # Use the new loader
        dataset_df=dummy_df_for_loader_trigger, 
        threshold_values_0_to_10=pruning_thresholds_0_to_10,
        original_model=original_cnn_model
    )

    if not results_cnn_df.empty:
        results_csv_path = run_path / "pruning_results_cnn.csv"
        results_cnn_df.to_csv(results_csv_path, index=False)
        print(f"\nPruning results saved to: {results_csv_path}")

        print("\nPlotting CNN pruning results...")
        plot_cnn_comprehensive_results(results_cnn_df, folder_path=run_path)
    else:
        print("No results generated from pruning evaluation.")

    print("\nCNN Pruning script finished.")