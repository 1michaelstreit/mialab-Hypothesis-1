# Medical Image Analysis Pipeline (MIALab)

A comprehensive pipeline for brain tissue segmentation using medical imaging data (MRI). This project supports both classical machine learning (Random Forest) and modern deep learning (3D UNet) approaches to segment brain tissues into Grey Matter, White Matter, and Cerebrospinal Fluid.

## Features

* **Multi-Modal Support:** Utilizes T1-weighted and T2-weighted MRI scans.
* **Dual-Pipeline Architecture:**
  * **Forest Mode:** Feature extraction (intensity, gradients, coordinates) coupled with a Scikit-Learn Random Forest Classifier.
  * **Deep Mode:** 3D Patch-based training using a MONAI `DynUNet` (Dynamic UNet) with PyTorch.
* **Robust Preprocessing:**
  * Atlas Registration (Affine/Linear).
  * Skull Stripping.
  * Normalization techniques: Z-Score, Min-Max, Percentile, and Histogram Matching.
* **Evaluation:** Automated calculation of metrics (Dice score, etc.) via `pymia`.

## Installation

### Prerequisites

* Python 3.8+
* CUDA-enabled GPU (recommended for Deep Learning mode)

### Dependencies

Install the required Python packages:

```bash
pip install numpy SimpleITK scikit-learn torch monai matplotlib pymia
```

```
/path/to/project/
├── mialab/
│   ├── data/
│   │   ├── atlas/          # Atlas images (T1, T2, mask)
│   │   ├── train/          # Training subjects
│   │   └── test/           # Testing subjects
```

**Each subject folder inside** **train** **or** **test** **must contain:**

* ***_T1.mha** **(T1-weighted image)**
* ***_T2.mha** **(T2-weighted image)**
* ***_GT.mha** **(Ground Truth labels)**
* ***_Mask.mha** **(Brain Mask)**
* ***_Transform.txt** **(Registration transform, if applicable)**

## Usage

**The entry point for the pipeline is** **pipeline.py**.

### 1. Random Forest Segmentation (Default)

**Extracts features, trains a Random Forest, and segments test images.**

### 2. Deep Learning Segmentation

Trains a 3D DynUNet on image patches and performs sliding-window inference on test images.

**code**

```
python pipeline.py --mode deep --result_dir ./results_dl --norm z_score
```


### 3. Preprocessing Only


**code**

```
python pipeline.py --prepro_only --result_dir ./debug_prepro
```


**Runs the preprocessing steps (registration, skull stripping, normalization) and saves the intermediate images without training a model. This is useful for debugging data quality.**


### 4. Batch Experiments (Benchmark)

**To execute the pipeline across all normalization methods and models in a single run, use** **run_experiments.py**.

**code**

```
python run_experiments.py --result_dir ./experiment_results
```

**This script will sequentially run experiments for:**

* **Modes:** **Forest and Deep.**
* **Normalizations:** **Z-Score, Min-Max, Percentile, Histogram Matching, and None.**

**Results are organized hierarchically:** **./experiment_results/`<mode>`/`<normalization>`/`<timestamp>`/**

**Optional Batch Flags:**

* **--skip_forest**: Skip Random Forest experiments.
* **--skip_deep**: Skip Deep Learning experiments.

## Command Line Arguments

| **Argument**         | **Default**         | **Description**                                                                                                                     |
| -------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **--mode**           | **forest**          | **Choice of method:** **forest** **or** **deep**.                                                                 |
| **--norm**           | **z_score**         | **Normalization method:** **z_score**, **min_max**, **percentile**, **histogram_matching**, **none**. |
| **--result_dir**     | **./mia-result**    | **Directory where results (segmentations, metrics, plots) are saved.**                                                              |
| **--data_train_dir** | **.../data/train/** | **Path to training data directory.**                                                                                                |
| **--data_test_dir**  | **.../data/test/**  | **Path to testing data directory.**                                                                                                 |
| **--data_atlas_dir** | **.../data/atlas/** | **Path to atlas data directory.**                                                                                                   |
| **--prepro_only**    | **False**           | **Flag to exit after preprocessing images (useful for Debugging).**                                                                 |


## Modules Description

**The codebase is structured into the following modules:**

* **pipeline.py**

  * **The main orchestrator. Handles argument parsing and selects the execution workflow (Forest vs. Deep).**
* **run_experiments.py**

  * **Automation script used to run the full suite of experiments (Forest/Deep + all Normalizations) in sequence. Handles error logging and result organization.**
* **preprocessing.py**

  * **Contains Filter classes for image manipulation.**
  * **ImageRegistration:** **Registers images to atlas space.**
  * **SkullStripping:** **Masks out non-brain tissue.**
  * **Normalization:** **Includes** **ZScore**, **MinMax**, **Percentile**, and **HistogramMatching** **classes.**
* **deep_learning_model.py**

  * **Contains the PyTorch/MONAI implementation.**
  * **BrainImageDataset:** **Custom Dataset class for loading 3D volumes and extracting patches.**
  * **run_deep_learning_pipeline:** **Manages the training loop and inference.**
* **feature_extraction.py**

  * **Used by the** **Random Forest** **pipeline to generate pixel-level features (coordinates, texture stats).**
* **postprocessing.py**

  * **Contains logic for cleaning up segmentation masks (e.g., Connected Components analysis).**

## Outputs

**Results are saved in the specified** **--result_dir** **inside a timestamped folder.**

* **.mha**: Predicted segmentation masks (**_SEG.mha***) and post-processed masks (**_SEG-PP.mha**).
* **results.csv**: Detailed metrics for every subject.
* **results_summary.csv**: Aggregated statistics (Mean/Std) across the dataset.
* **learning_curve.png**: (Deep Learning only) A plot of Training vs. Validation loss.
* **model.pth**: (Deep Learning only) The saved model weights.

## Acknowledgments

**This pipeline utilizes the following libraries:**

* **SimpleITK** **for image IO and processing.**
* **MONAI** **for Deep Learning architectures.**
* **pymia** **for data handling and evaluation metrics.**

## Outputs

**Results are saved in the specified** **--result_dir** **inside a timestamped folder.**

* ***.mha**: Predicted segmentation masks (_SEG.mha**) and post-processed masks (**_SEG-PP.mha**).
* **results.csv**: Detailed metrics for every subject.
* **results_summary.csv**: Aggregated statistics (Mean/Std) across the dataset.
* **learning_curve.png**: (Deep Learning only) A plot of Training vs. Validation loss.
* **model.pth**: (Deep Learning only) The saved model weights.

## Acknowledgments

**This pipeline utilizes the following libraries:**

* **SimpleITK** **for image IO and processing.**
* **MONAI** **for Deep Learning architectures.**
* **pymia** **for data handling and evaluation metrics.**

```

```
