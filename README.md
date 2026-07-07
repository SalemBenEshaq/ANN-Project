# Handwritten Digit Recognition — Neural Network Built From Scratch

A fully connected artificial neural network (ANN) for classifying handwritten digits (0–9), implemented entirely from scratch using NumPy — no deep learning frameworks (TensorFlow, PyTorch, Keras) involved.

## Overview

This project builds and trains a 3-layer neural network to recognize handwritten digits from grayscale images. All core components — forward propagation, activation functions, loss calculation, and the training loop — are implemented manually to demonstrate a low-level understanding of how neural networks work internally.

## Architecture

```
Input (784) -> Dense (32, ReLU) -> Dense (16, ReLU) -> Dense (10, Softmax)
```

- **Input layer:** 784 units (flattened 28x28 grayscale pixel values, normalized to [0, 1])
- **Hidden layer 1:** 32 units with ReLU activation
- **Hidden layer 2:** 16 units with ReLU activation
- **Output layer:** 10 units with Softmax activation (one per digit class, 0–9)
- **Loss function:** Categorical cross-entropy

## Data Pipeline

Images are loaded from class-labeled folders (one folder per digit, `0`–`9`), converted to NumPy arrays, flattened, and normalized. Labels are inferred from the folder name.

## Training

The model is trained using an iterative random-search optimization loop (rather than gradient-based backpropagation):

1. Randomly perturb all weights and biases.
2. Run a forward pass and compute the loss.
3. If the loss improves, keep the new weights and save a checkpoint; otherwise, revert to the last best checkpoint.
4. Repeat for a fixed number of iterations (5,000).

Model weights are checkpointed to disk (`.npz`) whenever a new best loss is found, so training can be resumed or the best model reloaded at any point.

## Results

| Metric | Score |
|---|---|
| Training Accuracy | 72.6% |
| Shared Test Set Accuracy | 73.0% |

## Files

| File | Description |
|---|---|
| `ANN Project.ipynb` | Main notebook: model definition, data loading, training, and evaluation |
| `ANN Project wights.npz` | Saved model weights (best checkpoint) |

## Requirements

- Python 3
- NumPy
- Pillow (PIL)

```bash
pip install numpy pillow
```

## Usage

1. Place training images in `Ai2_Dataset/<your_name>/<digit>/*.png`, organized into subfolders `0`–`9`.
2. Place the shared test set in `Ai2_Dataset/SharedTesting/<digit>/*.png`.
3. Run the notebook cells in order to train the model and evaluate it on the test set.
4. To reload a trained model without retraining, use `model.load("Salem2237189Model.npz")`.

## Author

Salem Eidhah Bin Ishaq
