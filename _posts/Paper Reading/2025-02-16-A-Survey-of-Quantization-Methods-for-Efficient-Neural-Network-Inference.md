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
    - Focusing on optimizing the architecture of the NN model in terms of its micro-architecture (kernel types; depth-wise convolution/low-rank factorization) as well as its macro-architecture (module types; residual/inception).  
    - To find the right NN architecture, Automated machine learning (AutoML) and Neural Architecture Search (NAS) are used.
- Co-designing NN architecture and hardware together:  
    - The latency/energy overhead of an NN component is hardware-dependent.
- Pruning:  
    - Neurons with small saliency (sensitivity) are removed, resulting in a sparse computational graph.  
    - Unstructured pruning:  
        - Neurons with small saliency are removed wherever they occur.  
        - One can perform aggressive pruning with very little impact on the generalization performance of the model.  
        - Leads to sparse matrix operations, which are hard to accelerate and memory-bound.
    - Structured pruning:  
        - A group of parameters (e.g., convolutional filters) is removed.  
        - Permits dense matrix operations.  
        - Aggressive structured pruning often leads to significant accuracy degradation.
- Knowledge distillation:  
    - Training a large model and then using it as a teacher to train a more compact model.  
    - Uses the soft probabilities produced by the teacher instead of hard class labels.  
    - A major challenge is to achieve a high compression ratio with distillation alone.  
    - Tends to have considerable accuracy degradation with aggressive compression.  
    - Combining distillation with other methods (quantization/pruning) has shown great success.
- Quantization:  
    - The breakthroughs of half-precision and mixed-precision training have enabled an order of magnitude higher throughput in AI accelerators.  
    - Very difficult to go below half-precision without significant tuning.

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

$$ Q $$: quantization operator that maps a floating-point value to a quantized one  
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
    - The clipping range is not symmetric with respect to the origin $$ (-\alpha \neq \beta) $$.  
    - E.g., $$ \alpha = r_{\min} $$, $$ \beta = r_{\max} $$.
- Symmetric quantization  
    - Chooses a symmetric clipping range $$ (-\alpha = \beta) $$.  
    - E.g., $$ -\alpha = \beta = \max(\vert r_{\max} \vert, \ \vert r_{\min} \vert) $$.

Asymmetric quantization results in a tighter clipping range as compared to symmetric quantization.
This is important when the target weights or activations are imbalanced (e.g., the activation after ReLU that always has non-negative values).

- Full range quantization:  
    - Uses the full INT8 range of $$ [-128, \ 127] $$.
- Restricted range quantization:  
    - Only uses the range of $$ [-127, \ 127] $$.

Using the min/max of the signal is a popular method.
However, this approach is susceptible to outlier data in the activations.
- Use percentiles of the signal.  
    - Instead of adopting the largest/smallest values, use $$ i $$-th largest/smallest percentiles as $$ \beta $$ and $$ \alpha $$.  
- Select $$ \alpha $$ and $$ \beta $$ that minimize KL divergence (i.e., information loss)

### Static and dynamic quantization

Another important differentiator of quantization methods is when the clipping range is determined.
Weights can processed using a static clipping range since in most cases the parameters are fixed during inference.
Whereas, the activation maps ($$ x $$ in \eqref{eq:1}) differ for each input sample.
- Dynamic quantization:  
    - The range is dynamically calculated for each activation map during runtime.  
    - Requires real-time computation of the signal statics (min/max, percentile, etc.), which can have a high overhead.  
    - Results in higher accuracy as the signal range is exactly calculated for each input.
- Static quantization:  
    - The clipping range is precalculated and static during inference.  
    - Does not add computational overhead.  
    - Typically results in lower accuracy.
    - Popular methods for static quantization:
        - Pre-compute the optimal range of activations:  
            - Minimizing MSE between original unquantized weight distribution and the corresponding quantized values is generally used.
        - Learn/impose the clipping range during NN training

Calculating the range of a signal dynamically is very expensive.
As such, practitioners most often use static quantization, where the clliping range is fixed for all inputs.

### Quantization granularity

- Layerwise quantization:  
    - The clipping range is determined by considering all of the weights in convolutional filters of a layer, then uses the same clipping range for all of the convolutional filters.  
    - Simple to implement.  
    - Results in sub-optimal accuracy.
- Groupwise quantization:  
    - Group multiple different channels inside a layer to calculate the clipping range.  
    - Is helpful for cases where the distribution of the parameters across a single convolution/activation varies a lot.  
    - Requires extra cost of accounting for different scaling factors.
- Channelwise quantization:  
    - Use a fixed value for each convolutional filter.  
    - Each channel is assigned a dedicated scaling factor.  
    - Ensures a better quantization resolution and results in higher accuracy.  
- Sub-channelwise quantization:  
    - The clippling range is determined with respect to any groups of parameters in a convolution of fully-connected layer.  
    - Considerable overhead.

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

#### 1. Rule-based non-uniform quantization

- Logarithmic distribution:  
    - The quantization steps and levels increase exponentially.

- Binary-code based quantization:  
    - A real-number vector $$ \mathbf{r} \in \mathbb{R}^n $$ is quantized into $$ m $$ binary vectors.

$$
\begin{align*}
    \mathbf{r} \approx \sum_{i = 1}^m \alpha_i \mathbf{b}_i
\end{align*}
$$

<p class="indent-1">
$$ \alpha $$: the scaling factor<br>
$$ \mathbf{b} \in {\{ -1, \ +1 \}}^n \in \mathbb{R}^n $$: the binary vector
</p>

Since there are no closed-form solutions for minimizing the error between $$ \mathbf{r} $$ and $$ \displaystyle \sum_{i = 1}^m \alpha_i \mathbf{b}_i $$, previous researches relied on heuristic solutions.

#### 2. Optimization-based non-uniform quantization

More recent work formulates non-uniform quantization as an optimization problem.
The quantization steps/levels in the quantizer $$ Q $$ are adjusted to minimize the difference between the original tensor and the quantized counterpart.

$$
\begin{align*}
    \min_Q (\| Q(r) - r \|)^2
\end{align*}
\label{eq:6}
\tag{6}
$$

Furthermore, the quantizer itself can be jointly trained with the model parameters.
The quantization steps/levels are generally trained with iterative optimization or gradient descent.

#### 3. Clustering

Some works use k-means on different tensors to determine the quantization steps and levels, while other work applies a Hessian-weighted k-means clustering on weights to minimize the performance loss.

Non-uniform quantization schemes are typically difficult to deploy efficiently on general computation hardware (e.g., CPU, GPU).
As such, the uniform quantization is the de-facto method due to its simplicity and its efficient mapping to hardware.

### Fine-tuning methods

#### 1. Quantization-aware training (QAT)

Quantization may disturb trained model parameters and push the model away from the point to which it had converged when it was trained with floating-point precision.
By retraining the NN model with quantized parameters, the model can converge to a point with better loss.

In QAT, the usual forward and backward pass are performed on the quantized model in floating-point, but the model parameters are quantized after each gradient update.

The way the non-differentiable quantization operator in Eq. \eqref{eq:2} is treated is highly important.
Without any approximation, the gradient of the quantization operator is zero almost everywhere, since the rounding operation in Eq. \eqref{eq:2} is a piecewise flat operator.
- Straight through estimator (STE):  
    - Approximate the gradient of the quantization operator with STE.  
    - STE ignores the rounding operation and approximates it with an identity function.
- Stochastic neuron approach
- Combinatorial optimization
- Target propagation
- Gumbel-softmax
- Regularization operators
    - Uses regularization operators to quantize the weights (Non-STE)
    - This method removes the need to use the non-differentiable quantization.
    - ProxQuant: uses W-shape non-smooth regularization function
    - Using pulse training to approximate the derivative of discontinuous points
    - Replacing the quantized weights with an affine combination of floating-point and quantized parameters
    - AdaRound: an adapted rounding method as an alternative to round-to-nearest method.

Non-STE methods require a lot of tuning and so far STE is the most commonly used method.

Some prior work found it effective to learn quantization parameters during QAT as well.
- PACT: Learns the clipping ranges of activations under uniform quantization.
- QIT: Learns quantization steps and levels as an extension to a non-uniform quantization setting.
- LSQ/LSQ+: Introduces a new gradient estimate to learn scaling factors for non-negative/general activations during QAT.

The main disadvantage of QAT is the computational cost of retraining the NN model.
Therefore, it is a waste to apply QAT to models that have a short lifetime.
Moreover, QAT needs a sufficient amount of training data for retraining.

#### 2. Post-training quantization (PTQ)

PTQ performs the quantization and the adjustment of the weights without any fine-tuning.
Thus, the overhead of PTQ is very low and often negligible.
Also, unlike QAT, it can be used in situations where data is limited or unlabeled.
However, PTQ results in lower accuracy compared to QAT, especially for low-precision quantization.
There are several approaches to mitigate the accuracy degradation of PTQ.
- Bias correction:
    - Observes inherent bias in the mean and variance of the weight values and corrects it.
- Equalizing the weight ranges:
    - ACIQ: Analytically computes the optimal clipping range and the channel-wise bitwidth setting for PTQ.
    - OMSE: Removes channel-wise quantization on activation (since it is hard to deploy on hardware) and conducts PTQ by optimizing the L2 distance between the quantized tensor and the corresponding floating-point tensor.
    - Outlier channel splitting (OCS): Duplicates and halves the channels containing outlier values.
    - AdaRound: an adaptive rounding method that better reduces the loss.

#### 3. Zero-shot quantization (ZSQ)

In many cases, access to the original training data is not possible during the quantization procedure since the training dataset is either too large or sensitive due to security or private concerns.
In these circumstances, it is difficult to perform QAT or PTQ.
To resolve this challenge, ZSQ is proposed.
ZSQ performs the quantization without any access to the training/validation data, which is particularly important for machine learning as a service (MLaaS).

There are two different levels of zero-shot quantization.
- Level 1: No data and no fine-tuning (ZSQ + PTQ)
    - Allows faster and easier quantization without any fine-tuning.
- Level 2: No data but requires fine-tuning (ZSQ + QAT)
    - Results in higher accuracy.

A popular branch of research in ZSQ is to generate synthetic data that is similar to the real data from which the target pre-trained model is trained.
The synthetic data is then used for calibrating and/or fine-tuning the quantized model.
There are several ways to produce synthetic data.
- Exploit generative adversarial networks (GANs):
    - Fails to capture the internal statistics (e.g., distributions of the intermediate layer activations) of the real data.
- Generating data by minimizing the KL divergence of the internal statistics.
- ZeroQ:
    - The synthetic data can be used for sensitivity measurement as well as calibration.
    - Enables mixed-precision PTQ without access to the training/validation data.

### Stochastic quantization

Stochastic quantization maps the floating number up or down with a probability associated to the magnitude of the weight update.
For instance, the Int function in Eq. \eqref{eq:2} can be defined as

$$
\begin{align*}
    \text{Int}(x) =
    \begin{cases}
        \lfloor x \rfloor && \text{with probability} && \lceil x \rceil - x \\
        \lceil x \rceil && \text{with probability} && x - \lfloor x \rfloor
    \end{cases}
\end{align*}
$$

or, for binary quantization,

$$
\begin{align*}
    \text{Binary}(x) =
    \begin{cases}
        -1 && \text{with probability} && 1 - \sigma(x) \\
        +1 && \text{with probability} && \sigma(x)
    \end{cases}
\end{align*}
$$

where Binary is a function to binarize the real value $$ x $$ and $$ \sigma(\cdot) $$ is the sigmoid function.

QuantNoise quantizes a different random subset of weights during each forward pass and trains the model with unbiased gradients.
This allows lower-bit precision quantization without a significant accuracy drop.

A major challenge with stochastic quantization methods is the overhead of creating random numbers for every single weight update.
Thus, they are not yet widely adopted in practice.

## Advanced concepts: quantization below 8 bits

### Simulated and integer-only quantization

In simulated (or fake) quantization, the quantized model parameters are stored in low-precision, but the operations are carried out with floating-point arithmetic.
The quantized parameters need to be dequantized before the floating-point operations.
Conversely, in integer-only quantization, all the operations are performed using low-precision integers.
The entire inference is carried out without any floating-point dequantization for any parameters or activations.

Performing the inference in full-precision with floating-point arithmetic may help the final quantization accuracy, but low-precision logic has benefits in terms of latency, power consumption, and area efficiency.
In general, integer-only quantization is more desirable as compared to simulated quantization.
However, where problems are bandwidth-bound rather than compute-bound, fake quantization methods can be useful.

### Mixed-precision quantization

In this approach, each layer is quantized with different bit precision.
The layers of an NN are grouped into sensitive/insensitive to quantization, and higher/lower bits are used for them.
One can minimize accuracy degradation and still benefit from reduced memory footprint and faster speedup with low precision quantization.

One challenge with mixed-precision quantization is that the search space for selecting layers to quantize is exponential in the number of layers.
- Exploration-based methods:
    - RL based method:
        - Determines the quantization policy automatically with a reinforcement learning.
    - Using neural architecture search (NAS)
    - Require large computational resources.
    - Performance is sensitive to hyperparameters and initialization.
- Periodic function regularization:
    - Automatically distinguish different layers with respect to accuracy while learning their respective bitwidths.
    - HAWQ
        - Uses second-order sensitivity of the model (Hessian).
    - HAWQv2
        - Extends the method used in HAWQ to mixed-precision activation quantization.
    - HAWQv3
        - Introduces an integer-only, hardware-aware quantization.
        - Proposes a fast integer linear programming method to find the optimal bit precision for a given application-specific constraint (e.g., model size or latency).

### Hardware-aware quantization

The benefits of quantization are hardware-dependent, with many factors such as on-chip memory, bandwidth, and cache hierarchy.
Some works use a reinforcement learning agent to determine the hardware-aware mixed-precision setting for quantization, based on a look-up table of latency with respect to different layers with different bitwidths.

### Distillation-assisted quantization

Model distillation is a method in which a large model with higher accuracy is used as a teacher model to help the training of a compact student model.
Model distillation uses the soft probabilities produced by the teacher.
The overall loss function incorporates both the student loss and the distillation loss.

$$
\begin{align*}
    \mathcal{L} = \alpha \mathcal{H}(y, \ \sigma(z_s)) + \beta \mathcal{H}(\sigma(z_t, \ T), \ \sigma(z_s, \ T))
\end{align*}
\label{eq:7}
\tag{7}
$$

$$ \alpha, \ \beta $$: weighting coefficients to tune the amount of loss from the student model/distillation loss  
$$ y $$: the ground-truth class label  
$$ \mathcal{H} $$: the cross-entropy loss function  
$$ z_s, \ z_t $$: logits generated by the student/teacher model  
$$ \sigma $$: the softmax function  
$$ T $$: the temperature of the softmax function, defined as follows:

$$
\begin{align*}
    p_i = \frac{\exp{(z_i / T)}}{\sum_j \exp{(z_j / T)}}
\end{align*}
$$

### Vector quantization

There are a lot of interesting ideas in the classical quantization methods in digital signal processing that have been applied to vector quantization.
- Clustering weights:
    - clusters the weights into different groups and uses the centroid of each group as quantized values during inference.
    - Using a k-means clustering is sufficient to reduce the model size up to $$ 8 \times $$ without significant accuracy degradation.
    - Jointly applying k-means based vector quantization with pruning and Huffman coding can further reduce the model size.

$$
\begin{align*}
    \min_{c_1, \dots, c_k} \sum_i {\| w_i - c_j \|}^2
\end{align*}
$$

<p class="indent-2">
$$ i $$: the index of weights in a tensor<br>
$$ c_1, \dots, c_k $$: the $$ k $$ centroids found by the clustering<br>
After clustering, weight $$ w_i $$ will have a cluster index $$ j $$ related to $$ c_j $$ in the look-up table.
</p>

## Quantization and hardware processors

Edge devices have tight resource constraints including computing, memory, and power budget.
In addition, many edge processors do not support floating-point operations, especially in microcontrollers.

- ARM Cortex-M:
    - A group of 32-bit RISC ARM processor cores that are designed for low-cost and power-efficient embedded devices.
    - Because some of the ARM Cortex-M cores do not include floating-point units, the models should first be quantized before deployment.
- CMSIS-NN:
    - A library from ARM that helps quantizing and and deploying NN models onto the ARM Cortex-M cores.
    - Leverages fixed-point quantization with power-of-two scaling factors.
- GAP-8:
    - A RISC-V SoC for edge inference with a CNN accelerator.
    - Only supports integer arithmetic.
- Google Edge TPU:
    - A purpose-built ASIC chip.
    - Designed for small and low-power devices.
    - Only supports 8-bit arithmetic.
    - NN models must be quantized using QAT or PTQ of TensorFlow.

---

Sources:
- [Gholami, A., Kim, S., Dong, Z., Yao, Z., Mahoney, M. W., & Keutzer, K. (2021). A Survey of Quantization Methods for Efficient Neural Network Inference. ArXiv.](https://arxiv.org/abs/2103.13630)
