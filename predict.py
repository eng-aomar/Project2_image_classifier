import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings
import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import argparse

def process_image(image_path):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32)
    image /= 255
    return image.numpy()

def predict(image_path, model, top_k=5):
    processed_img = process_image(image_path)
    processed_img = np.expand_dims(processed_img, axis=0)
    prob_pred = model.predict(processed_img)
    probs, classes = tf.math.top_k(prob_pred, k=top_k)
    return probs.numpy().tolist()[0], classes.numpy().tolist()[0]

def main():
    parser = argparse.ArgumentParser(description='Image Classifier Predictor')
    parser.add_argument('--input', default='./test_images/hard-leaved_pocket_orchid.jpg', 
                       type=str, help='image path')
    parser.add_argument('--model', default='./best_model.h5', 
                       type=str, help='model path')
    parser.add_argument('--top_k', default=5, type=int, help='top K classes')
    parser.add_argument('--category_names', default='label_map.json',
                       type=str, help='class name mapping')
    
    args = parser.parse_args()
    
    print("Starting Prediction...")
    
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    
    # Critical fix: Load with hub.KerasLayer
    reloaded_model = tf.keras.models.load_model(
        args.model,
        custom_objects={'KerasLayer': hub.KerasLayer},
        compile=False
    )
    
    probs, classes = predict(args.input, reloaded_model, args.top_k)
    label_names = [class_names[str(int(cls)+1)] for cls in classes]
    
    print("\nResults:")
    print("==============================")
    print(f"Image: {args.input}")
    for name, prob in zip(label_names, probs):
        
        print(f"\u2022 {name}: {prob:.2%}")
        print("---------------------------")
if __name__ == "__main__":
    main()