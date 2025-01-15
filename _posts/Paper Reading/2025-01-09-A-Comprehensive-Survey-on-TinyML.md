---
title: A Comprehensive Survey on TinyML
# description: Short summary of the post
date: 2025-01-09 16:00
categories: [Computer Science, Paper Reading]
tags: [tinyml, embedded-machine-learning, deep-learning, edge-intelligence]     # TAG names should always be lowercase
math: true
pin: false
---

## Summary

- Tiny machine learning (TinyML) is a promising AI alternative focusing on technologies and applications for extremely low-profile devices.
- TinyML focuses on deploying compressed and optimized ML models on tiny, low-power devices such as battery-powered microcontrollers and embedded systems.
- Compared to traditional ML models, TinyML has advantages such as reduced latency, offline capability, improved latency and security, low energy consumption, and reduced cost.
- TinyML relies mostly on software, which enables the deployment of machine learning models on resource-constrained hardware.
it employs two primary techniques at the software level: model compression and knowledge distillation.
- The three major TinyML Frameworks are Tensorflow Lite, Edge Impulse, and Arm NN.
- TinyML has been used in a number of fields, including healthcare, anomaly detection in the industry, environment, and smart farming.

## Background

The term IoT means a network that not only connects computers but extends this connection to any other device or thing.
The IoT architecture can be broken down into four fundamental layers:

1. The perception layer: it is composed of sensors that gather data and physical measurements, and actuators that execute tasks or actions based on sensor data.
2. The network of transport layer: it comprises the infrastructure for internet gateways and data acquisition systems to transmit and gather data from different devices to an on-premise location.
3. The middleware or processing layer: it includes high-performance machines for data analysis and data storage.
4. The application of service/interface layer: it grants user access to services and presents them through interfaces or APIs.

The IoT architecture may be separated into three tiers based on computing capacity: cloud computing, fog computing, and edge computing.
Cloud computing is the topmost layer, fog computing is in the middle, and edge computing is the lowest.
The computing paradigm is shifting from cloud computing to end-edge-cloud computing (EECC), which also supports AI evolving from a centralized AI deployment to a distributed artificial intelligence (DAI).
This new paradigm is empowered by the heterogeneous computing capabilities of on-devices, edge, and cloud servers that are managed to meet the requirements raised by resource-intensive and distributed AI computation.

Edge AI evolved as a response to the limits of cloud-based AI, which is not necessarily appropriate for real-time applications and devices with limited processing power and bandwidth.
Mobile ML is one example of the use of intelligence on the edge layer.
A Neural Processing Unit (NPU) was incorporated in current mobiles to enable the execution of ML algorithms.
NPU is substantially quicker than traditional processors such as CPU or GPU in performing matrix multiplications, which are the essential operations of neural networks.

### The strengths of TinyML

In the IoT workflow, TinyML refers to the use of ML models that are small and resource-efficient enough to run on devices with limited computational capabilities, such as MCU found in IoT devices.
TinyML was inspired by Mobile ML's features and its development grew as a result of the technical breakthrough in the field of IoT and MCUs.

1. Reducing latency  
TinyML models can run on the device itself; thus the response time is much faster than sending the data to the cloud for processing.
This is critical for applications that require real-time decisions, such as image and speech recognition.
When compared to cloud-based ML models, the deployment of models on TinyML systems significantly reduces latency, with a range of 0 to 5 ms, while offering high accuracy with only a slight decrease from 95% to 85% due to compression and optimization for use on restricted devices.
2. Offline capability  
TinyML models can run even when there is no Internet connection, whereas cloud ML models require such connectivity.
3. Improving privacy and security  
TinyML keeps the data on the device; thus, sensitive information does not have to be sent to the cloud for processing.
4. Low energy consumption  
TinyML reduces the quantity of data that has to be transported and processed.
Furthermore, TinyML algorithms are frequently intended to be computationally efficient, which can help in lowering the power consumption of the device on which they are executed.
5. Reducing cost  
TinyML models can save on costs associated with sending data to the cloud for processing and storage, such as bandwith and storage costs.

## Hardware  Constraints

Currently, the majority of hardware boards use the ARM Cortex processor with CPU clock frequencies ranging from 100 MHz to 480 MHz.
The emergence of the 32bits generation of IoT-ready microcontrollers and the support for single instruction multiple data (SIMD) and digital signal processing (DSP) instructions made it possible for Cortex-M-based devices to perform previously unachievable tasks.
MCUs also include on-chip SRAM and embedded Flash; thus, models that can fit within the memory limits are free from the costly DRAM accesses that limit classical ML.

Deep learning hardware accelerators are specialized chips or circuits designed to improve the performance and efficiency of neural networks.
They provide parallel processing capabilities and optimized data flow to reduce computation time, energy consumption, and memory usage.
A tensor processing unit (TPU) is a custom-built AI accelerator designed by Google to perform highly efficient matrix calculations, the foundation of many machine learning algorithms.
Edge TPU is a form of TPU intended exclusively to run TinyML models with great performance and minimal power consumption.

## Software Optimization

TinyML relies mostly on software, which optimizes model size and computation and therefore enables the deployment of machine learning models on resource-constrained hardware.
Model compression is a strategy for reducing a machine learning model's size and processing needs.
This method can result in a 20% to 30% decrease in memory space required for network parameter storing.
There are several ways to compress a model:
- Pruning: The pruning method begins with training the network and then selecting the key links by locating the weights that are greater than a specific threshold.
Weights below this level are eliminated, resulting in a trimmed model.
The trimmed model may not provide the same accuracy as the dense network, but retraining the residual weights can restore the accuracy.
- Quantization: Quantization is used to reduce the precision of the weights.
Activation is reduced from 32-bit or 64-bit to 8-bit or lower fixed-point numbers.
The objective of quantization is to strike a compromise between model accuracy and the precision of the weights and activations.
- Low-rank factorization: Low-rank factorization is a mathematical approach for approximating a high-dimensional matrix with low-dimensional matrices while maintaining as much information as possible from the original matrix.
The model is represented more compactly with fewer parameters and is more computationally efficient.
- Huffman coding: Huffman coding is a lossless compression method that assigns binary codes to each symbol in the data set.
More frequently appearing symbols have shorter binary codes, while less frequently appearing symbols have longer codes.

In addition to model compression techniques, knowledge distillation is another important technique used in TinyML.
Knowledge distillation is a machine learning approach in which a smaller, more compact model (referred to as a student model) is trained to mimic the outputs of a larger, more accurate model (called a teacher model).
The student model can learn useful information about the problem from the teacher model, even if the student model is not as complex or accurate as the teacher model.
The knowledge distillation training procedure consists of two steps: (1) training the teacher model on the original training data, and (2) training the student model with the teacher model's predictions as the target.

However, the traditional techniques for compressing TinyML models mentioned above can lead to a significant loss of accuracy due to poor matrix characteristics resulting from high compression rates.
This prompted the development of tiny neural networks, which are compact neural network models with a restricted number of parameters.
It is designed to function effectively on embedded devices with low processing resources.
Compared to larger models, tiny neural networks are trained on a fraction of an original dataset and employ a simpler design.
This allows them to be trained and deployed more quickly and efficiently while retaining a high degree of precision.

The deployment of models to embedded devices cannot currently support model training due to limited resources.
Typically, models are trained on the cloud or on a more capable device before being distributed to the embedded device.
There are three methods for deploying models: hand coding, code generation, and ML interpreters.
Both hand coding and code generation provide low-level optimizations, but hand coding takes time while code generation has portability problems.
An ML interpreter is a tool used to implement machine learning algorithms on embedded devices with limited processing capabilities, including MCUs.

Aside from the interpreter, a TinyML framework typically includes TinyML libraries and tools for data processing, as well as a Tiny Inference Engine, which is a low-level software library or hardware accelerator designed to efficiently perform the computation required for machine learning inferences.
Overall, the Tiny Inference Engine provides the computing capabilities needed to execute the models, whereas the TinyML Interpreter handles model execution.
- TensorFlow Lite (TFL): TFL is a Google open-source deep learning framework designed for inference on embedded devices.
It is made up of two primary parts: the Converter and the Interpreter.
The TensorFlow Converter is used to convert TensorFlow code into a compressed flat buffer (.tflite), shrink the size of the model, and optimize the code with minimal accuracy loss.
TFL currently supports quantization, pruning, and clustering.
- Edge Impulse: Edge Impulse is a cloud-based solution that facilitates the creation and deployment of machine learning models for TinyML devices.
The process entails collecting data using IoT devices, extracting features, training models, and finally deploying and optimizing the models for TinyML devices.
It employs the EON compiler for model deployment and also supports TFLM.
Edge Impulse utilizes TensorFlow's Model Optimization Toolkit to quantize models, lowering the precision of their weights from float32 to int8 with minimal impact on accuracy.
- Arm NN: Arm NN is a Linux-based, open-source software framework for machine learning inference on embedded devices developed by Arm.
Arm NN makes use of fixed-point arithmetic, quantizing model parameters to either 8-bit or 16-bit integers for deployment to microcontrollers for inferencing.

---

Sources:
- [Y. Abadade, A. Temouden, H. Bamoumen, N. Benamar, Y. Chtouki and A. S. Hafid, "A Comprehensive Survey on TinyML," in IEEE Access, vol. 11, pp. 96892-96922, 2023](https://ieeexplore.ieee.org/abstract/document/10177729)