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


---

Sources:
- [Y. Abadade, A. Temouden, H. Bamoumen, N. Benamar, Y. Chtouki and A. S. Hafid, "A Comprehensive Survey on TinyML," in IEEE Access, vol. 11, pp. 96892-96922, 2023](https://ieeexplore.ieee.org/abstract/document/10177729)