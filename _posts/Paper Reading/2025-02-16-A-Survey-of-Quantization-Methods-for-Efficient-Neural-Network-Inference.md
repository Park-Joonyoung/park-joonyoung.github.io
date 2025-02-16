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

Denote  
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

### Static and dynamic quantization

Another important differentiator of quantization methods is when the clipping range is determined.
Weights can processed using a static clipping range since in most cases the parameters are fixed during inference.
Whereas, the activation maps ($$ x $$ in \eqref{eq:1}) differ for each input sample.
- Dynamic quantization:  
The range is dynamically calculated for each activation map during runtime.  
Requires real-time computation of the signal statics (min/max, percentile, etc.), which can have a high overhead.  
Results in higher accuracy as the signal range is exactly calculated for each input.
- Static quantization:  
The clipping range is precalculated and static during inference.  
Does not add computational overhead.  
Typically results in lower accuracy.
Popular methods for static quantization are as follows.
    - Pre-compute the optimal range of activations:  
    Minimizing MSE between original unquantized weight distribution and the corresponding quantized values is generally used.
    - Learn/impose the clipping range during NN training

Calculating the range of a signal dynamically is very expensive.
As such, practitioners most often use static quantization, where the clliping range is fixed for all inputs.

### Quantization granularity

- Layerwise quantization:  
The clipping range is determined by considering all of the weights in convolutional filters of a layer, then uses the same clipping range for all of the convolutional filters.  
Simple to implement.  
Results in sub-optimal accuracy.
- Groupwise quantization:  
Group multiple different channels inside a layer to calculate the clipping range.  
Is helpful for cases where the distribution of the parameters across a single convolution/activation varies a lot.  
Requires extra cost of accounting for different scaling factors.
- Channelwise quantization:  
Use a fixed value for each convolutional filter.  
Each channel is assigned a dedicated scaling factor.  
Ensures a better quantization resolution and results in higher accuracy.  
- Sub-channelwise quantization:  
The clippling range is determined with respect to any groups of parameters in a convolution of fully-connected layer.  
Considerable overhead.

Channelwise quantization is widely adopted.

### Non-uniform quantization

Non-uniform quantization allows quantization steps, as well as quantization levels, to be non-uniformly spaced.
The formal definition of the non-uniform quantization is as below.

$$
\begin{align*}
    Q(r) = X_i, \text{ if } r \in [\delta_i, \ \delta_{i + 1}]
\end{align*}
\label{eq:5}
\tag{5}
$$

$$ X_i $$: the quantization levels  
$$ \delta_i $$: the quantization steps (thresholds)

When the value of a real number $$ r $$ falls in between the quantization step $$ \delta_i $$ and $$ \delta_{i + 1} $$, quantizer $$ Q $$ projects it to the corresponding quantization level $$ X_i $$.

Non-uniform quantization may achieve higher accuracy for a fixed bit-width, because one could better capture the distributions by focusing more on important value regions or finding appropriate dynamic ranges.

The category of non-uniform quantizations is as follows.
- Logarithmic distribution:  
The quantization steps and levels increase exponentially.

- Binary-code based quantization:  
A real-number vector $$ \mathbf{r} \in \mathbb{R}^n $$ is quantized into $$ m $$ binary vectors.

$$
\begin{align*}
    \mathbf{r} \approx \sum_{i = 1}^m \alpha_i \mathbf{b}_i
\end{align*}
$$

<p class="indented-paragraph">
$$ \alpha $$: the scaling factor<br>
$$ \mathbf{b} \in {\{ -1, \ +1 \}}^n \in \mathbb{R}^n $$: the binary vector
</p>
