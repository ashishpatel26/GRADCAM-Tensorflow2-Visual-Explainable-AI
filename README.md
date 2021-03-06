# GRADCAM-Tensorflow2-Visual-Explainable-AI
###  Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization

* Install Grad CAM : `!pip install tf-explain`
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

* Note : you can pass `model` object as any tensorflow keras model.
