# Classification-of-images-MLP

## Introduction
This project demonstrates the use of TensorFlow and Keras to build, train, and evaluate two neural network models for image classification tasks. The datasets used are:

1. **MNIST Dataset**: Handwritten digits (0-9).
2. **Fashion MNIST Dataset**: Images of clothing items.

Both models are simple feedforward neural networks consisting of two hidden layers and one output layer.

## Prerequisites

- Python 3.7 or later
- Jupyter Notebook or Google Colab
- TensorFlow (>= 2.0)
- NumPy
- Matplotlib
- scikit-learn

Install the required packages using:
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Running the Code

### 1. Load the Notebook
- Open the Jupyter Notebook or Google Colab.
- Copy and paste the code provided into the notebook.

### 2. Execute the Code
- Run each code cell sequentially to:
  - Load the dataset.
  - Normalize the data.
  - Build and compile the models.
  - Train the models.
  - Evaluate performance on the test set.
  - Visualize results using plots and confusion matrices.

### 3. MNIST Model

- **Training and Validation**: The model is trained on the MNIST dataset to classify digits.
- **Results**: The training accuracy, validation accuracy, and loss are displayed in plots.
- **Confusion Matrix**: A confusion matrix is generated to evaluate model performance.
- **Sample Predictions**: The model's predictions are visualized for 15 test images.

### 4. Fashion MNIST Model

- **Training and Validation**: The model is trained on the Fashion MNIST dataset to classify clothing items.
- **Results**: Similar plots and a confusion matrix are displayed.
- **Sample Predictions**: The model's predictions are shown for 5 test images with their respective class labels.

### Notes

- Both models use ReLU activation for hidden layers and Softmax for the output layer.
- The Adam optimizer is used for training.
- The data is normalized by dividing pixel values by 255.

## Example Output

### MNIST Dataset
```
Accuracy on the test set: 97.85%
```

### Fashion MNIST Dataset
```
Accuracy on the test set: 89.12%
```

## Troubleshooting

- Ensure all libraries are correctly installed.
- Use Google Colab if your local machine lacks sufficient computational resources.
- If accuracy is low, consider tweaking the model architecture, learning rate, or number of epochs.

## Acknowledgments

- TensorFlow/Keras documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- MNIST and Fashion MNIST datasets are provided by Yann LeCun and Zalando Research respectively.
