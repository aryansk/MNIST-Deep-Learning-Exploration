# MNIST Deep Learning Exploration

This repository contains implementation of various deep learning approaches for the MNIST handwritten digit classification task, using both scikit-learn and Keras frameworks.

## Project Overview

This project explores different neural network architectures and hyperparameters for classifying handwritten digits from the MNIST dataset. The implementation is divided into two main parts:

### Part 1: Feed Forward Network using Scikit-learn
- Dataset handling and preprocessing
- Implementation of basic Feed Forward Network
- Exploration of different train-test splits
- Analysis of model performance with varying iterations

### Part 2: Deep Neural Networks using Keras
- Comprehensive exploration of neural network architectures
- Analysis of different design choices:
  - Number of nodes (4 to 2056)
  - Number of layers (4 to 16)
  - Various layer-node combinations
  - Different activation functions
  - Custom activation function combinations

## Requirements

```
numpy
matplotlib
scikit-learn
tensorflow
keras
```

## Project Structure

```
├── part1_sklearn_implementation.py
├── part2_keras_implementation.py
└── README.md
```

## Features

1. **Data Preprocessing**
   - MNIST dataset loading and reshaping
   - Feature scaling and normalization
   - Train-test splitting

2. **Model Implementations**
   - Basic FFN with configurable parameters
   - Various neural network architectures
   - Multiple activation function combinations

3. **Analysis Tools**
   - Accuracy metrics
   - Training time measurements
   - Parameter count tracking
   - Performance visualization

## Key Experiments

1. **Node Count Analysis**: Testing networks with 4, 32, 64, 128, 512, and 2056 nodes
2. **Layer Depth Study**: Comparing networks with 4, 5, 6, 8, and 16 layers
3. **Activation Function Comparison**: Testing sigmoid, tanh, and ReLU
4. **Dataset Split Analysis**: Evaluating different train-test splits (60-40, 75-25, 80-20, 90-10)

## Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mnist-deep-learning-exploration.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the experiments:
```bash
python part1_sklearn_implementation.py
python part2_keras_implementation.py
```

## Results

The repository includes implementations that achieve:
- Basic FFN accuracy with various iterations (10 to 200)
- Comparative analysis of different neural network architectures
- Performance metrics for different model configurations
- Training time analysis for various architectures

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MNIST Dataset providers
- Scikit-learn and Keras documentation
- Assignment structure from CSET-335 Deep Learning course
