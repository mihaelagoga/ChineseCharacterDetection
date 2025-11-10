
# ChineseCharacterDetection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

This project focuses on the semantic segmentation of Chinese characters in high-resolution street-view images. The primary challenge addressed is the extreme class imbalance present in the dataset, where Chinese characters constitute less than 1% of the pixels, with the vast majority being background.

The poject documents an experimental process comparing two distinct models:
1.  A **Simple Convolutional Neural Network (CNN)** serving as a baseline.
2.  A complex **U-Net architecture with a pre-trained ResNet-34 encoder**, utilizing transfer learning, deep supervision, and advanced loss functions to combat class imbalance.

A key technique employed is the **Sliding Window** approach, which breaks down large images into smaller, manageable patches for the model to process, mitigating issues with image resolution and memory constraints.

## Installation

To set up the environment and run this project, you will need Python 3.x and the following libraries. You can install them using pip:

```bash
pip install torch torchvision opencv-python numpy matplotlib tqdm
```

It is highly recommended to use a CUDA-enabled GPU for training the models, as this will significantly speed up the process.

## Dataset

The dataset consists of high-resolution images (`.jpg`) and their corresponding annotations in a `.jsonl` file. Each image has annotations for various objects, including polygons for Chinese characters.

**Data Preparation Pipeline:**
1.  **Parsing and Filtering:** Annotations are parsed from the `train.jsonl` file. A filter is applied to retain only the polygons corresponding to Chinese characters, discarding other objects.
2.  **Dataset Split:** The official training list is split into training (80%), validation (10%), and testing (10%) sets.
3.  **Sliding Window:** To handle the high-resolution images and focus on character details, the images are broken down into hundreds of smaller, overlapping patches (windows). This technique forms the basis of the dataset fed into the more complex model.
4.  **Data Augmentation:** For the U-Net model, data augmentation techniques like `RandomHorizontalFlip`, `RandomRotation`, and `ColorJitter` are applied to the training set to prevent overfitting.

*Note: The notebook uses hardcoded paths to the dataset. You will need to modify these paths to point to the location of your data.*

## Usage

The entire workflow, from data preparation to model training and evaluation, is contained within the Jupyter Notebook: `Chinese Character Detection.ipynb`.

To run the experiment, execute the cells in the notebook sequentially:
1.  **Data Loading and Preparation:** The initial cells load the annotation files (`train.jsonl`, `info.json`), split the data, and set up the `ChineseCharacterDataset` class, which implements the sliding window logic.
2.  **Model 1 (SimpleSegmentationModel) Training:** Run the cells that define, compile, and train the baseline CNN model. The trained model weights are saved as `simple_segmentation_model_improved.pth`.
3.  **Model 2 (UNetResNetDeepSupervision) Training:** Proceed to the cells that define the U-Net architecture, the custom loss functions (`FocalLoss`, `DiceLoss`), and execute the two-stage training process (training the decoder, then fine-tuning the entire network). The best model weights are saved as `deep_supervision_model_best.pth`.
4.  **Evaluation and Visualization:** The final cells load the saved weights for both models and perform a comprehensive quantitative (metrics) and qualitative (visual benchmark) evaluation on the test set.

## Model Architecture

Two models were implemented to compare a simple approach against a more sophisticated one.

### Model 1: SimpleSegmentationModel

*   **Description:** A basic encoder-decoder CNN.
*   **Encoder:** Consists of a sequence of `Conv2d` and `MaxPool2d` layers to down-sample the input and extract features.
*   **Decoder:** Uses `ConvTranspose2d` layers to up-sample the feature maps back to the original image dimensions, reconstructing the segmentation mask.
*   **Loss Function:** Trained using Binary Cross-Entropy (BCE) Loss with an AdamW optimizer.

### Model 2: UNetResNetDeepSupervision

*   **Description:** A U-Net architecture that replaces the standard convolutional encoder with a pre-trained ResNet-34 model, leveraging transfer learning from the ImageNet dataset.
*   **Key Features:**
    *   **Skip Connections:** Concatenates feature maps from the encoder with the corresponding layers in the decoder, preserving spatial information and improving gradient flow.
    *   **Deep Supervision:** The loss is calculated at multiple decoder stages, allowing shallower layers to guide the deeper ones and mitigating the vanishing gradient problem.
    *   **Loss Functions:** A combination of **Dice Loss** (to handle overlap between predicted and true masks) and **Focal Loss** (to diminish the contribution of easy background pixels and force the model to focus on challenging characters).
    *   **Two-Stage Training:** The encoder is initially frozen to train only the decoder. Subsequently, the entire network is unfrozen and fine-tuned with a lower learning rate.

## Evaluation

The models were evaluated both quantitatively using standard segmentation metrics and qualitatively through visual inspection.

### Quantitative Results

The metrics were calculated on the test set. A probability threshold of 0.5 was used to convert the output probability maps into binary masks.

| Metric    | Model 1 (Simple CNN) | Model 2 (U-Net + Sliding Window) |
|-----------|----------------------|----------------------------------|
| Accuracy  | 0.3670               | 0.9954                           |
| Precision | 0.0075               | 0.8432                           |
| Recall    | 0.9903               | 0.0428                           |
| F1-Score  | 0.0148               | 0.0814                           |

**Analysis:**
*   **Model 1** exhibited a tendency to **over-predict**. Its high recall and extremely low precision indicate that it found most true characters but also incorrectly labeled a vast amount of the background as positive.
*   **Model 2** struggled with **under-prediction**. Its high precision shows that when it did predict a character, it was often correct. However, the very low recall demonstrates that it was overly cautious and failed to identify most of the characters.

### Qualitative Results

Visual comparisons confirm the quantitative findings.
*   **Model 1** produces noisy predictions, highlighting general object outlines, shadows, and road markings rather than specific characters.
*   **Model 2** is much more intentional. It successfully learns to ignore the background but is not sensitive enough to detect characters unless they are in the most ideal conditions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
