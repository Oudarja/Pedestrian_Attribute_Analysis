# Pedestrian Attribute Analysis

## Overview

Pedestrian Attribute Analysis is a computer vision project designed to detect and analyze visual attributes of pedestrians from images or video streams. The system leverages deep learning models to extract meaningful attributes such as clothing, accessories, and other identifiable characteristics useful for surveillance, smart city analytics, and research in human-centered AI.

This repository contains the implementation of models, preprocessing pipelines, and experiments conducted on the **PA-100K** pedestrian attribute dataset.

---

## Features

* Pedestrian attribute classification using deep learning
* Video and image-based inference
* Dataset preprocessing and feature extraction
* Modular architecture for experimenting with different models
* Jupyter notebook for dataset analysis and preprocessing

---

## Project Structure

```
Pedestrian_Attribute_Analysis/
│
├── HP_model/                 # Deep learning model implementations
│   ├── AF.py
│   ├── Hydraplus.py
│   └── Mnet.py
│
├── R&D/                      # Research and experimentation notebooks
│   └── Dataset_processing.ipynb
│
├── Video_test.py             # Video inference script
├── requirements.txt          # Project dependencies
├── LICENSE                   # License file
└── README.md                 # Project documentation
```

---

## Dataset

This project uses the **PA-100K** dataset, a large-scale benchmark dataset for pedestrian attribute recognition.

Dataset includes:

* 100,000 pedestrian images
* 26 attribute annotations
* Multiple attribute categories such as clothing, accessories, and demographics

**Due to size limitations, the dataset is not included in this repository**.

You can download it from the official dataset source.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Oudarja/Pedestrian_Attribute_Analysis.git
cd Pedestrian_Attribute_Analysis
```

Create a virtual environment:

```bash
python -m venv pedestrian_env
source pedestrian_env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Run Attribute Analysis on Video

```bash
python Video_test.py
```

### Dataset Processing

Use the provided Jupyter notebook:

```
R&D/Dataset_processing.ipynb
```

to preprocess and explore the dataset.

---

## Models

The repository includes implementations of several models used for pedestrian attribute recognition:

* **HydraPlus Network**
* **Multi-branch networks**
* Custom deep learning architectures for attribute prediction

These models are implemented in the `HP_model` directory.

---

## Applications

Pedestrian attribute recognition can be applied in:

* Smart surveillance systems
* Person re-identification
* Retail analytics
* Crowd behavior analysis
* Urban safety monitoring

---

## Requirements

Main dependencies include:

* Python
* PyTorch
* OpenCV
* NumPy
* Jupyter Notebook

Full list available in `requirements.txt`.

---

## Future Improvements

* Real-time inference optimization
* Integration with object detection models
* Web-based interface for visualization
* Deployment with API services

---

## License

This project is licensed under the terms specified in the `LICENSE` file.

---

---

