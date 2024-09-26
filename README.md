# Meta Tensor

**Meta Tensor** is a deep learning framework built using Python and NumPy. The framework is designed to offer an intuitive and efficient platform for building and training neural networks, focusing on flexibility and ease of use. Whether you're a researcher or a developer, Meta Tensor provides the core building blocks to construct complex models from the ground up.

## Features

- **Computational Graphs**: Construct dynamic computational graphs for backpropagation.
- **Fully-Connected Layer**: Implement dense layers to create complex neural network architectures.
- **Convolutional Layer**: Build convolutional neural networks (CNNs) for image-related tasks.
- **Pooling Layer**: Use max-pooling and average-pooling layers for down-sampling.
- **Loss Functions**:
  - **Perception Loss**: Useful for basic perception tasks.
  - **Log Loss**: Also known as logistic loss, suited for binary classification tasks.
  - **CrossEntropyWithSoftMax**: A combination of softmax and cross-entropy loss for multi-class classification.
- **Optimizers**:
  - **Gradient Descent**: Basic optimization method for minimizing loss functions.
  - **Momentum**: Enhances gradient descent with momentum to accelerate convergence.
  - **AdaGrad**: Adaptive Gradient Descent, suited for sparse data.
  - **RMSProp**: Root Mean Square Propagation, useful for non-stationary objectives.
  - **Adam**: Adaptive Moment Estimation, one of the most popular optimization algorithms.

## Installation

You can clone this repository and install Meta Tensor in editable mode to start using it.

```bash
git clone git@github.com:SightVanish/MetaTensor.git
cd metatensor
pip install -Ue .
```

## ModelZoo

Models implemented in `model/`

## Contributing

Contributions are welcome! If you have ideas or improvements, feel free to fork the repository, create a new branch, and open a pull request.

## License

This project is licensed under the MIT License.
