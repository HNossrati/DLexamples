# Multi-Layer Perceptron Implementation from Scratch in MATLAB

## Overview
This repository contains educational implementations of Multi-Layer Perceptron (MLP) neural networks built from scratch in MATLAB. It features two different approaches to computing gradients: traditional backpropagation using the chain rule and automatic differentiation (AutoDiff).

These implementations serve as intuitive learning tools for understanding the core concepts behind neural networks, gradient computation, and automatic differentiation.

## Repository Structure
- **MLP_ChainRule.m**: Implementation using traditional backpropagation with chain rule
- **MLP_AutoDiff.m**: Implementation using custom automatic differentiation
- **AutoDiff.m**: Class definition for automatic differentiation functionality

## Features

### Common Features
- Fully-connected neural network architecture
- Configurable network topology
- Minibatch gradient descent
- Data normalization options (min-max, standard, max)
- Training/validation split
- Gradient clipping
- Comprehensive visualization tools
- Test case evaluation

### Activation Functions
- Sigmoid with configurable steepness
- Hyperbolic tangent (tanh)

### Training Visualizations
- Training and validation loss curves
- Target vs. Prediction scatter plots
- Time series predictions visualization

## Example Problem
The implementations are demonstrated on a regression problem with:
- 3 input features
- 2 output targets
- Non-linear relationships between inputs and outputs:
  - Output 1: Square root of first input
  - Output 2: Complex function involving all inputs

## Implementation Details

### MLP with Chain Rule (MLP_ChainRule.m)
- Traditional backpropagation implementation
- Manual computation of gradients using the chain rule
- Step-by-step gradient flow through the network
- Clear separation of forward and backward passes

### MLP with AutoDiff (MLP_AutoDiff.m & AutoDiff.m)
- Custom automatic differentiation implementation
- Dynamic computation graph construction
- Automatic gradient tracking
- Object-oriented design for computational nodes

### Key Hyperparameters
- Learning rate: 0.25
- Batch size: Configurable (default: 1)
- Network architecture: [3, 5, 4, 2] (configurable)
- Training epochs: 800
- Training/Validation split: 80%/20%

## Usage

1. **Basic Usage**
```matlab
% Run either implementation
run MLP_ChainRule.m
% or
run MLP_AutoDiff.m
