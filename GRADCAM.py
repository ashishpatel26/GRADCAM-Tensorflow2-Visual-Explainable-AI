'''
Install Grad CAM : `!pip install tf-explain`
* src : https://github.com/sicara/tf-explain
* paper : Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
* Reference : https://arxiv.org/abs/1610.02391
* Abstract : We propose a technique for producing "visual explanations" for decisions from a large class
  of CNN-based models, making them more transparent. Our approach - Gradient-weighted Class Activation Mapping 
  (Grad-CAM), uses the gradients of any target concept, flowing into the final convolutional layer to produce 
  a coarse localization map highlighting important regions in the image for predicting the concept. Grad-CAM 
  is applicable to a wide variety of CNN model-families: 
  (1) CNNs with fully-connected layers, 
  (2) CNNs used for structured outputs, 
  (3) CNNs used in tasks with multimodal inputs or reinforcement learning, 
  without any architectural changes or re-training. We combine Grad-CAM with fine-grained visualizations to create
   a high-resolution class-discriminative visualization and apply it to off-the-shelf image classification, captioning, 
   and visual question answering (VQA) models, including ResNet-based architectures. In the context of image classification 
   models, our visualizations (a) lend insights into their failure modes, 
   (b) are robust to adversarial images, (c) outperform previous methods on localization, (d) are more faithful to the 
   underlying model and (e) help achieve generalization by identifying dataset bias. For captioning and VQA, we show that even
    non-attention based models can localize inputs. We devise a way to identify important neurons through Grad-CAM and combine it 
    with neuron names to provide textual explanations for model decisions. Finally, we design and conduct human studies to measure 
    if Grad-CAM helps users establish appropriate trust in predictions from models and show that Grad-CAM helps untrained users 
    successfully discern a 'stronger' nodel from a 'weaker' one even when both make identical predictions. 

##### Note : you can pass `model` object as any tensorflow keras model.
'''
import tensorflow as tf
import os
import cv2
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
from tf_explain.core.grad_cam import GradCAM
import numpy as np


class_mapping = {0: 'Cat', 1: 'Dog'}


def preprocessing_image(instancePath):
    original_image = plt.imread(instancePath)
    image = load_img(instancePath, target_size=(224, 224))
    image = img_to_array(image)
    image = tf.expand_dims(image, 0)
    image /= 255.0
    image = preprocess_input(image)
    return image, original_image


def predict_per(IMAGE_PATH):
    image, o_image = preprocessing_image(IMAGE_PATH)
    prediction = np.argmax(model.predict(image))
    prediction_per = np.max(model.predict(image))
    return class_mapping[prediction], prediction, prediction_per


def Grad_cam_vis(IMAGE_PATH, OUTPUT_PATH, ACTUAL_LABEL: str):
    prediction, index, confidense = predict_per(IMAGE_PATH)
    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    data = ([img], None)

    # Start explainer
    explainer = GradCAM()
    grid = explainer.explain(data, model, class_index=index)

    explainer.save(grid, ".", OUTPUT_PATH)

    im = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
    im1 = cv2.cvtColor(cv2.imread(OUTPUT_PATH), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.xlabel(f"Actaul: Healthy")
    plt.subplot(1, 2, 2)
    plt.imshow(im1)
    plt.xlabel(f"predict:{prediction}\nConfidence: {confidense}")
    plt.show()


images = os.listdir('/content/Images/')
for img in images:
  # print(img)
  os.mkdirs("Output", exist_ok = True)
  maping_result  = {'c':'Cat', 'd':'Dog'}
  actual_label = maping_result[img[0].lower()]
  IMAGE_PATH = f"/content/Images/{img}"
  OUTPUT_PATH = f"/content/Output/output_{img[:-4]}.jpg"
  pred_class, prediction, prediction_per = predict_per(IMAGE_PATH)
  Grad_cam_vis(IMAGE_PATH, OUTPUT_PATH, actual_label)
