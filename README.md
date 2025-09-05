# Exploring Foundation Models for Muscle Segmentation

This repository contains the code for the paper:

**"Exploring Foundation Models for Multi-Class Muscle Segmentation in MR Images of Neuromuscular Disorders: A Comparative Analysis of Accuracy and Uncertainty."**

---

##  Model Checkpoints

The best pretrained/fine-tuned checkpoints are available for download:

* **SAM model (LoRA)**: [Google Drive folder](https://drive.google.com/drive/folders/1xkSoc7jyghColvpSKVHvvjaWS-bqgOdQ?usp=sharing)
* **nnUNet 3D**: [Google Drive folder](https://drive.google.com/drive/folders/1-3-zFsE33FG4EuEP6QT6F_7mujICGGX4?usp=sharing)

---

##  Training Instructions

If you want to **train the models** yourself on muscle segmentation task , you can download the original [SAM](https://github.com/facebookresearch/segment-anything) or [MedSAM](https://github.com/bowang-lab/MedSAM) checkpoints from their repo. If you want to use as a starting point our checkpoints trained on our private dataset, download them using the previous link.


> Note: Our fine-tuning strategy for SAM is **adapted from the Mazurowski Lab approach**, with modifications for multi-class muscle segmentation.  For the original method and instructions on how to preprocess the dataset for fine-tuning SAM, especially on how to load the images in the **datasets** folder and how to generate the .csv in **files** , see: [Mazurowski Lab SAM fine-tuning](https://github.com/mazurowski-lab/finetune-SAM).

> nnUNet 3D training follows standard nnUNet procedures as described in the official [nnUNet repository](https://github.com/MIC-DKFZ/nnUNet).

---

##  Evaluation Files

The repository contains **evaluation scripts and files** used to compute the performance metrics for the models. These include:

* **Dice Scores, HD95 and ASSD** (utils/metrics.py): Calculation of accuracy scores for segmentation.
* **Uncertainty Maps** (utils/plot_utils.py): Visualizations of areas where the model predictions are less confident.
* **Calibration and Sharpness scores calculation** (utils/utils_uq.py): Calculation of ECE, NLL and Entropy.

---

##  Paper

For full details, please refer to the paper linked in this repository

Exploring foundation models for multi-class muscle segmentation in MR images of neuromuscular disorders: A comparative analysis of accuracy and uncertainty: [Paper](https://www.sciencedirect.com/science/article/pii/S0169260725004523)

---
