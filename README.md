# adamw-optimizer-from-scratch
This project implements the AdamW optimizer from scratch using PyTorch and compares it against standard optimizers.

The goal is to understand how modern optimizers work internally and reproduce the behavior of AdamW with decoupled weight decay.

The implementation is compared with:

SGD

Adam

PyTorch AdamW

AdamW (from scratch)

## Optimizers Overview

Optimization algorithms are critical for training deep neural networks efficiently.

This project focuses on Adam and AdamW, two widely used optimizers in modern deep learning.

## Adam Optimizer

Adam (Adaptive Moment Estimation) combines ideas from:

Momentum

RMSProp

It keeps two moving averages of gradients.

First moment (mean)

m_t = β1 * m_{t-1} + (1 − β1) * g_t

Second moment (variance)

v_t = β2 * v_{t-1} + (1 − β2) * g_t²

The parameter update is

θ = θ − α * m̂ / (sqrt(v̂) + ε)

Where:

α is the learning rate

ε prevents division by zero

## Bias Correction

Because moving averages start at zero, Adam applies bias correction during early training.

m̂ = m / (1 − β1^t)
v̂ = v / (1 − β2^t)

This ensures unbiased estimates of the first and second moments.

## AdamW Optimizer

AdamW was introduced to fix a key issue in Adam regarding weight decay.

Traditional implementations apply L2 regularization as part of the gradient update.

AdamW instead decouples weight decay from gradient updates, leading to better generalization.

Standard Adam with L2 regularization:

g ← g + λθ

AdamW decouples this:

θ ← θ − α * gradient_update
θ ← θ − α * λ * θ

This is known as Decoupled Weight Decay.

AdamW is now widely used in:

Transformers

Vision Transformers

Large Language Models

## Implementation Details

The custom optimizer maintains for each parameter:

Step counter

First moment (m)

Second moment (v)

Update procedure:

Compute gradient

Update moments

Apply bias correction

Compute Adam update

Apply decoupled weight decay

## Experiment

We train a small MLP classifier on a synthetic dataset and compare optimizers.

Model:

Input: 20 features
Hidden layers: 64 → 32
Output: 2 classes
Activation: ReLU

Dataset:

5000 samples
20 features
Binary classification

Batch size:

64
## Optimizers Compared

SGD

Adam

PyTorch AdamW

AdamW implemented from scratch

## Results

Training loss across epochs:

| Optimizer       | Final Loss |
| --------------- | ---------- |
| SGD             | ~0.256     |
| Adam            | ~0.196     |
| AdamW (PyTorch) | ~0.197     |
| AdamW (Scratch) | ~0.193     |



The custom AdamW implementation closely matches PyTorch's implementation, validating the correctness of the algorithm.

## Optimizer Comparison Plot

Adam-based optimizers converge significantly faster than SGD in this experiment.

## How to Run

Clone the repository:

git clone https://github.com/yourusername/adamw-optimizer-from-scratch
cd adamw-optimizer-from-scratch

Install dependencies:

pip install -r requirements.txt

Run the experiment:

python optimizer_comparison.py

## Key Concepts Demonstrated

This project demonstrates understanding of:

Gradient-based optimization

Momentum methods

Adaptive learning rates

Bias correction

Decoupled weight decay

PyTorch optimizer internals

## Why This Matters

Modern architectures such as:

Transformers

GPT-style models

Vision Transformers

are typically trained using AdamW.

Understanding the optimizer implementation provides deeper insight into:

training stability

convergence speed

generalization performance
