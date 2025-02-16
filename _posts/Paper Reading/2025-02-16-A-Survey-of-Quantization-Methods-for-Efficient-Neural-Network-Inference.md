---
title: A Survey of Quantization Methods for Efficient Neural Network Inference
# description: Short summary of the post
date: 2025-02-16 15:41
categories: [Computer Science, Paper Reading]
tags: [machine-learning, embedded-machine-learning, deep-learning, quantization, uniform-quantization, non-uniform-quantization, symmetric-quantization, asymmetric-quantization, static-quantization, dynamic-quantization, quantization-granularity, quantization-aware-training, post-training-quantization, zero-shot-quantization, stochastic-quantization, simulated-quantization, integer-only-quantization, mixed-precision-quantzation, hardware-processor]     # TAG names should always be lowercase
math: true
pin: false
---

## Abstract

The problem of quantization: how can we minimize the number of bits required and at the same time maximize the accuracy of the computations?

Moving from floating-point representations to low-precision integer representations reduces the memory footprint and latency.

## Introduction

The efforts for optimal quantizations are categorized as follows.  
- Designing efficient NN model architectures:  
Focusing on optimizing the architecture of the NN model in terms of its micro-architecture (kernel types; depth-wise convolution/low-rank factorization) as well as its macro-architecture (module types; residual/inception).  
To find the right NN architecture, Automated machine learning (AutoML) and Neural Architecture Search (NAS) are used.
- Co-designing NN architecture and hardware together:  
The latency/energy overhead of an NN component is hardware-dependent.
- Pruning:  
Neurons with small saliency (sensitivity) are removed, resulting in a sparse computational graph.  
    - Unstructured pruning:  
    Neurons with small saliency are removed wherever they occur.  
    One can perform aggressive pruning with very little impact on the generalization performance of the model.  
    Leads to sparse matrix operations, which are hard to accelerate and memory-bound.
    - Structured pruning:  
    A group of parameters (e.g., convolutional filters) is removed.  
    Permits dense matrix operations.  
    Aggressive structured pruning often leads to significant accuracy degradation.
- Knowledge distillation:  
Training a large model and then using it as a teacher to train a more compact model.  
Uses the soft probabilities produced by the teacher instead of hard class labels.  
A major challenge is to achieve a high compression ratio with distillation alone.  
Tends to have considerable accuracy degradation with aggressive compression.  
Combining distillation with other methods (quantization/pruning) has shown great success.
- Quantization:  
The breakthroughs of half-precision and mixed-precision training have enabled an order of magnitude higher throughput in AI accelerators.  
Very difficult to go below half-precision without significant tuning.

## General history of quantization

Rounding and truncation have been important problems for a long time.

NNs bring unique challenges and opportunities to the problem of quantization.
- Inference and training of NNs are both computationally intensive.  
Efficient representation of numerical values is important.
- NN models are heavily over-parameterized.  
NNs are very robust to aggressive quantization and extreme discretization.

Due to the over-parameterization, there are many different models that optimize the error metric.
Therefore, it is possible to have high error/distance betweeen a quantized model and the original non-quantized model, while still attaining great generalization performance.

Different layers in an NN have different impact on the loss function, and this motivates a mixed-precision approach to quantization.

## Basic concepts of quantization

### Problem setup an notations

WLOG, we focus on the supervised learning problem, where the goal is to optimize the following empricial risk minimization function:

$$
\begin{align*}
    \mathcal{L}(\theta) = \frac{1}{N} \sum_{i = 1}^N l(x_1, y_1, \theta)
    \label{eq:1}
    \tag{1}
\end{align*}
$$

$$ \theta $$: denotes the combination of learnable parameters with respect to layers $$ (\{ W_1, W_2, \dots, W_L \}) $$.  
$$ N $$: total number of data points.  
$$ (x, \ y) $$: the input data and the corresponding label  
$$ l(x, \ y; \ \theta) $$: loss function (e.g. MSE/cross entropy)  

Let us denote  
$$ h_i $$: the input hidden activations of the $$ i $$th layer  
$$ a_i $$: the corresponding output hidden activation

In quantization, the goal is to reduce the precision of both the parameters $$ (\theta) $$ and the intermdiate activation maps $$ (h_i, \ a_i) $$ to low-precision, with minimal impact on the generalization power/accuracy of the model.

### Uniform quantization

The typical form of a quantization function is

$$
\begin{align*}
    Q(r) = \text{Int}\left( \frac{r}{S} \right) - Z
    \label{eq:2}
    \tag{2}
\end{align*}
$$

$$ Q $$: quantization operator that maps a floating point value to a quantized one  
$$ r $$: a real valued input (activation or weight)  
$$ S $$: a real valued scaling factor  
$$ Z $$: integer zero point

Int function maps a real value to an integer value through a rounding operation.
This method of quantization is also known as uniform quantization, as the resulting quantized values (quantization levels) are uniformly spaced.
In contrast, non-uniform quantization does not quantize values uniformly.

It is also possible to recover real values $$ r $$ from the quantized values $$ Q(r) $$ through a dequantization operation.

$$
\begin{align*}
    \tilde{r} = S(Q(r) + Z)
    \label{eq:3}
    \tag{3}
\end{align*}
$$

Note that recovered real value $$ \tilde{r} $$ does not exactly match the value of $$ r $$ due to the rounding operation.

### Symmetric and asymmetric quantization

The scaling factor in Eq. \eqref{eq:2} divides a given range of real values $$ r $$ into a number of partitions:

$$
\begin{align*}
    S = \frac{\beta - \alpha}{2^b - 1}
    \label{eq:4}
    \tag{4}
\end{align*}
$$

$$ [\alpha, \ \beta] $$: the clipping range; a bounded range that we are clipping the real values with  
$$ b $$: the quantization bit width

The process of choosing the clipping range is referred to as calibration.
- Asymmetric quantization:  
The clipping range is not symmetric with respect to the origin $$ (-\alpha \neq \beta) $$.  
E.g., $$ \alpha = r_{\min} $$, $$ \beta = r_{\max} $$.
- Symmetric quantization  
Chooses a symmetric clipping range $$ (-\alpha = \beta) $$.  
E.g., $$ -\alpha = \beta = \max(|r_{\max}|, \ |r_{\min}|) $$.

Asymmetric quantization results in a tighter clipping range as compared to symmetric quantization.
This is important when the target weights or activations are imbalanced (e.g., the activation after ReLU that always has non-negative values).

- Full range quantization:  
Uses the full INT8 range of $$ [-128, \ 127] $$.
- Restricted range quantization:  
Only uses the range of $$ [-127, \ 127] $$.

Using the min/max of the signal is a popular method.
However, this approach is susceptible to outlier data in the activations.
- Use percentiles of the signal.  
Instead of adopting the largest/smallest values, use $$ i $$-th largest/smallest percentiles as $$ \beta $$ and $$ \alpha $$.  
- Select $$ \alpha $$ and $$ \beta $$ that minimize KL divergence (i.e., information loss)