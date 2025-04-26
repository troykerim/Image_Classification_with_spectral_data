# Hyperspectral Image Classification Using 3D Convolutional Neural Networks

## Overview

This project is part of the **Astro Cultivators** (now **FOODI**) initiative from the [**Autonomous Research Center for STEAHM (ARCS)**](https://arcs.center/astro-cultivators/). It was also developed in connection with the [NASA Deep Space Food Challenge](https://www.nasa.gov/prizes-challenges-and-crowdsourcing/centennial-challenges/deep-space-food-challenge/), which aims to advance food production technologies suitable for long-duration space missions. The goal of this project was to build a machine learning model capable of diagnosing plant health through hyperspectral image classification.

The model classifies sweet potato leaves into six categories based on **plant necrosis**, which refers to the natural decay or death of plant tissue. Hyperspectral images were captured using the **Cubert Ultris S5** camera, producing a custom dataset that integrates spatial and spectral information in a three-dimensional format.

Due to the size of the dataset (~14 GB), it is not available for public use via GitHub. However, the full implementation details and results can be found in the official project report PDF included in this repository. The complete model code is located in `main.py`.

---

## Abstract

Hyperspectral imaging is a powerful technique for analyzing pixel-level information across various electromagnetic wavelengths. Unlike traditional RGB cameras, which only capture images in three color channels (red, green, blue), hyperspectral cameras can capture hundreds or thousands of spectral bands. These spectral images form three-dimensional data cubes, which include the spatial dimension (as in RGB) and an additional third dimension containing spectral data. This extra layer of information significantly enhances the ability of machine learning and deep learning models to classify and analyze image data.

This project leverages hyperspectral imaging to implement a **three-dimensional Convolutional Neural Network (3D-CNN)** for classifying the health of sweet potato leaves based on plant necrosis. Using hyperspectral images captured by the Cubert Ultris S5, the 3D-CNN model incorporates both spatial and spectral data to boost diagnostic accuracy. Developed in **Python** using the **PyTorch** framework, the model also integrates machine learning techniques such as **early stopping** and **cross-validation**, achieving an overall classification accuracy of **83% to 85%** across six necrosis-based health categories:

- Healthy  
- Moderately Healthy  
- Early Necrosis  
- Moderate Necrosis  
- Severe Necrosis  
- Dead or Inanimate Object  

This work demonstrates the powerful role of hyperspectral imaging in precision agriculture and plant health diagnostics. By integrating deep learning with high-dimensional spectral data, the project offers a scalable solution for early detection and classification of plant health conditions.

---

## Contents

- `main.py` – Source code for model training, architecture, and evaluation  
- `project_report.pdf` – Official PDF report documenting methodology, results, and analysis  
- `images/` – Visual resources and sample outputs from the model *(to be uploaded)*

---

## Future Work
 
- Upload screenshots and additional visuals from the GUI and results  
- Consider compressing and hosting a sample version of the dataset for public demos

---

## Acknowledgments

This project was supported by the Autonomous Research Center for STEAHM and contributed to the NASA Deep Space Food Challenge.