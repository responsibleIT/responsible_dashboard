from skimage import io
import cv2
import matplotlib.pyplot as plt
from huggingface_hub import from_pretrained_keras
import os 
import numpy as np # Make sure numpy is imported

ROWS, COLS = 150, 150
IMAGE_BASE_PATH = 'datasets/kaggle'

def preprocess_image(image_to_process):
    if image_to_process.dtype != np.uint8 and np.issubdtype(image_to_process.dtype, np.floating):
        if image_to_process.max() <= 1.0:
             image_to_process = (image_to_process * 255).astype(np.uint8)
        else: 
            image_to_process = image_to_process.astype(np.uint8)
    elif image_to_process.dtype != np.uint8:
        print(f"Warning: Unexpected image dtype {image_to_process.dtype}, attempting to convert to uint8.")
        image_to_process = image_to_process.astype(np.uint8)

    if image_to_process.ndim == 2:
        print(f"Warning: Encountered grayscale image for preprocessing. Converting to 3 channels (RGB).")
        image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_GRAY2RGB)
    elif image_to_process.ndim == 3 and image_to_process.shape[2] == 4:
        print(f"Warning: Encountered RGBA image for preprocessing. Converting to RGB.")
        image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_RGBA2RGB)
    elif image_to_process.ndim != 3 or image_to_process.shape[2] != 3:
        print(f"CRITICAL ERROR: Image for preprocessing has unexpected dimensions/channels: {image_to_process.shape}. Cannot proceed.")
        raise ValueError(f"Image has unexpected dimensions for preprocessing: {image_to_process.shape}")

    resized_image = cv2.resize(image_to_process, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    normalized_image = resized_image / 255.0
    return normalized_image.reshape(1, ROWS, COLS, 3) # This adds the batch dimension back AFTER processing

def load_images(base_path):
    images = []
    labels = []
    for label_name in ['cat', 'dog']:
        print(f"Processing label: {label_name}")
        for i in range(0, 500): 
            img_path = f'{base_path}/{label_name}/{i}.jpg'
            
            if not os.path.exists(img_path):
                print(f"File does not exist, skipping: {img_path}")
                continue

            print(f"Attempting to load: {img_path}")
            try:
                current_image = io.imread(img_path)
            except Exception as e:
                print(f"Error during io.imread for {img_path}: {e}")
                continue 

            if current_image is None:
                print(f'CRITICAL ERROR: io.imread returned None for: {img_path}')
                continue 

            if not isinstance(current_image, np.ndarray) or current_image.size == 0:
                 print(f"CRITICAL ERROR: Image loaded from {img_path} is not a valid numpy array or is empty.")
                 continue

            print(f"  Loaded {img_path}, initial shape: {current_image.shape}, dtype: {current_image.dtype}")

            # ---- SOLUTION: Handle potential extra dimension ----
            if current_image.ndim == 4 and current_image.shape[0] == 1:
                print(f"  Image {img_path} has an extra leading dimension. Squeezing it out.")
                current_image = np.squeeze(current_image, axis=0)
                print(f"  New shape for {img_path}: {current_image.shape}")
            # ---- END SOLUTION ----

            # Additional check after potential squeeze
            if current_image.ndim < 2 or current_image.ndim > 3 : # Expect 2D (grayscale) or 3D (color) now
                 print(f"CRITICAL ERROR: Image {img_path} still has problematic dimensions after potential squeeze: {current_image.shape}")
                 continue


            try:
                processed_image_data = preprocess_image(current_image) # Pass the (potentially squeezed) 3D image
                images.append(processed_image_data)
                labels.append(0 if label_name == 'cat' else 1)
            except cv2.error as e:
                print(f"OpenCV error during preprocess_image for {img_path}: {e}")
                print(f"  Image shape before resize call: {current_image.shape}, dtype: {current_image.dtype}")
                exit(1) 
            except ValueError as e: 
                print(f"ValueError during preprocess_image for {img_path}: {e}")
                exit(1)
            except Exception as e:
                print(f"Generic error during preprocess_image for {img_path}: {e}")
                exit(1)
                
    return images, labels

# ... (rest of your main execution code) ...

# --- Main execution part ---
model = from_pretrained_keras("carlosaguayo/cats_vs_dogs")
images_data, labels_data = load_images(IMAGE_BASE_PATH) 

if not images_data:
    print("No images were loaded successfully. Exiting.")
    exit()

total_predictions = len(images_data)
correct_predictions = 0

print(f"\nStarting predictions for {total_predictions} images...")
for i in range(total_predictions):
    img = images_data[i] # This is already (1, ROWS, COLS, 3)
    label = labels_data[i]
    
    prediction_output = model.predict(img) 
    prediction_value = prediction_output[0][0] 
    predicted_class = 0 if prediction_value >= 0.5 else 1

    if predicted_class == label:
        correct_predictions += 1
    
if total_predictions > 0:
    accuracy = correct_predictions / total_predictions
    print(f'\nAccuracy: {accuracy:.2%}')
    print(f"Correct predictions: {correct_predictions} out of {total_predictions}")
else:
    print("No predictions were made as no images were loaded.")