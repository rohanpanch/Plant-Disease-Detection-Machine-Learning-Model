import os
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

class_names = [
    'Alstonia Scholaris (Diseased)', 'Alstonia Scholaris (Healthy)', 'Apple (Healthy)',
    'Apple Applescab (Diseased)', 'Apple Blackrot (Diseased)', 'Apple Cedarapplerust (Diseased)',
    'Arjun (Diseased)', 'Arjun (Healthy)', 'Bell_pepper leaf (Healthy)',
    'Bell_pepper leaf spot (Diseased)', 'Blueberry (Healthy)', 'Cherry (Healthy)',
    'Cherry Powderymildew (Diseased)', 'Chinar (Healthy)', 'Corn (Healthy)',
    'Corn Commonrust (Diseased)', 'Corn Gray leaf spot (Diseased)', 'Corn NorthernLeafBlight (Diseased)',
    'Corn leaf blight (Diseased)', 'Corn rust leaf (Diseased)', 'Gauva (Diseased)', 'Grape (Healthy)',
    'Grape Blackrot (Diseased)', 'Grape Esca(BlackMeasles) (Diseased)',
    'Grape Leafblight(IsariopsisLeafSpot) (Diseased)', 'Guava (Healthy)', 'Jamun (Diseased)',
    'Jamun (Healthy)', 'Jatropha (Diseased)', 'Jatropha (Healthy)', 'Lemon (Diseased)',
    'Lemon (Healthy)', 'Mango (Diseased)', 'Mango (Healthy)',
    'Orange Haunglongbing(Citrusgreening) (Diseased)', 'Peach (Healthy)', 'Peach Bacterialspot (Diseased)',
    'Pepper,bell (Healthy)', 'Pepper,bell Bacterialspot (Diseased)', 'Pomegranate  (Healthy)',
    'Pomegranate (Diseased)', 'Pongamia Pinnata  (Healthy)', 'Pongamia Pinnata (Diseased)',
    'Potato (Healthy)', 'Potato Earlyblight (Diseased)', 'Potato Lateblight (Diseased)',
    'Raspberry (Healthy)', 'Soybean (Healthy)', 'Squash Powderymildew (Diseased)',
    'Strawberry (Healthy)', 'Strawberry Leafscorch (Diseased)', 'Tomato (Healthy)',
    'Tomato Bacterialspot (Diseased)', 'Tomato Earlyblight (Diseased)', 'Tomato Lateblight (Diseased)',
    'Tomato LeafMold (Diseased)', 'Tomato Septorialeafspot (Diseased)',
    'Tomato Spidermites Two-spottedspidermite (Diseased)', 'Tomato TargetSpot (Diseased)',
    'Tomato TomatoYellowLeafCurlVirus (Diseased)', 'Tomato Tomatomosaicvirus (Diseased)',
    'Tomato leaf mosaic virus (Diseased)', 'Tomato leaf yellow virus (Diseased)', 'grape leaf black rot (Diseased)'
]

def predict_image(model, image_path):
    img = load_img(image_path, target_size=(225, 225))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    
    return predicted_class_name

def get_actual_class(image_path):
    return image_path.split(os.sep)[-2]

# Load the model once to avoid reloading it for every image
model_path =  '/Users/rohan/Desktop/FinalProject /FYP /new code/best_model.h5'
model = load_model(model_path)

# Directory containing all the test images
test_dir = '/Users/rohan/Desktop/FinalProject/FYP/new code/PlantImages/Split/test'
correct_predictions_dir = '/Users/rohan/Desktop/FinalProject/FYP/new code/PlantImages/Split/TEST_IMAGES'

# Process each subfolder in the test directory
for subdir, dirs, files in os.walk(test_dir):
    for file in files:
        # Construct the full path to the image
        image_path = os.path.join(subdir, file)

        # Predict the class of the image
        predicted_class_name = predict_image(model, image_path)
        actual_class_name = get_actual_class(image_path)

        print(f"Actual class: {actual_class_name}")
        print(f"Predicted class: {predicted_class_name}")

        # If prediction is correct, copy the image to the correct_predictions_dir
        if actual_class_name == predicted_class_name:
            # Construct the destination path while preserving the folder structure
            relative_path = os.path.relpath(subdir, test_dir)
            correct_image_dir = os.path.join(correct_predictions_dir, relative_path)

            # Create the directory if it does not exist
            if not os.path.exists(correct_image_dir):
                os.makedirs(correct_image_dir)
            
            # Copy the image to the new directory
            shutil.copy(image_path, correct_image_dir)
