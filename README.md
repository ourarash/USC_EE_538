# 🧠 Machine Learning Hardware and Systems  
**EE508 - University of Southern California**  
*Instructor: Arash Saifhashemi*

---

## 📄 Quick Links

- [Catalog Description (PDF)](https://web-app.usc.edu/soc/syllabus/20251/30507.pdf)

---

## 📝 Catalogue Description

**ML kernels:** Convolutions, transformers, embeddings  
**Accelerators:** GPUs, input/weight/output stationary accelerators  
**Distributed ML:** Data, model, and hybrid parallel  
**Private ML:** Homomorphic encryption and multi-party computing accelerators

---

## 📚 Course Description

This course offers a unique perspective for ECE students interested in building Machine Learning (ML) hardware and systems—such as GPUs and accelerators—and designing scalable ML systems, including cloud-based ML training and inference pipelines.

You will learn about:
- Computation and memory access kernels in ML models (convolutions, transformers, embedding tables)
- Transforming convolutions to matrix operations and accelerating them on hardware
- Three hardware design paradigms: input, output, and weight stationary accelerators
- In-depth understanding of ML hardware accelerators (GPUs, TPUs)
- Scaling ML systems with model, data, and hybrid parallelism
- Privacy in ML and accelerating private ML systems using homomorphic encryption and multi-party computing

---

## 🎯 Learning Objectives

By the end of this course, you will be able to:

- 🔄 Transform convolutional neural network pipelines into matrix computations, map them to GPUs/TPUs/ML accelerators, and measure per-iteration training runtime
- ⚡ Build hardware accelerator blocks for matrix computations using input/output/weight stationary methods for ML inference and training
- 📊 Quantify performance bottlenecks in ML systems through code instrumentation and system monitoring
- 🖥️ Train large ML models using model or data parallel computations on multiple GPUs
- 🔒 Design privacy-preserving ML models using multi-party computing algorithms and map them to GPU hardware with PyTorch Crypten
- ⏱️ Quantify the latency impact of private ML and identify execution bottlenecks

---

## 🛠️ Recommended Preparation

- Coding matrix algebraic algorithms
- Understanding of gradient descent
- Experience with PyTorch or TensorFlow
- Familiarity with memory systems of CPUs and GPUs

---

## 📝 Course Notes

- **Grading:** Letter grading on a 4-point scale
- **Format:** In-person (including DEN-based participation)
- **Materials:** All lecture materials posted on D2L Brightspace
- **Discussion:** Moderated on Piazza
- **Programming:** All exercises via GitHub

---

## 💻 Technological Proficiency & Hardware/Software

- Labs use USC’s HPC CARC GPUs via cloud-based virtual instance reservations
- If CARC is busy, Google Colab will be used for GPU access
- Tutorials for Colab and CARC will be provided, with hands-on demos in discussion sections

---

## 📂 Repository Contents

This repository contains:
- Lecture slides and materials
- ML kernels (convolutions, transformers, embeddings)
- Hardware accelerators (GPUs, TPUs, input/output/weight stationary)
- Distributed and private ML topics

---

## 📖 Lectures Notes

Please refer to the individual [PDF files](lectures) for specific lecture content.


| Lecture Topic              | PDF Link                                                   |
| -------------------------- | ---------------------------------------------------------- |
| Lecture 1: Introduction    | [pdf](lectures/ML_Lecture1Sp25.pdf)                        |
| Lecture 2: ML Review - 1   | [pdf](lectures/ML_Lecture2Sp25_ML_Review.pdf)              |
| Lecture 3: ML Review - 2   | [pdf](lectures/ML_Lecture3Sp25_ML_Review.pdf)              |
| CDNN Part 1                | [pdf](lectures/ML_LectureSp25_CDNN.pdf)                    |
| CDNN Part 2                | [pdf](lectures/ML_LectureSp25_CDNN2.pdf)                   |
| Kernel Design 1            | [pdf](lectures/ML_LectureSp25_KernelDesign1.pdf)           |
| Kernel Design 2 (Modeling) | [pdf](lectures/ML_LectureSp25_KernelDesign2(modeling).pdf) |
| Eyeriss                    | [pdf](lectures/ML_LectureSp25_Eyeriss.pdf)                 |
| Transform Optimization     | [pdf](lectures/ML_LectureSp25_TransformOptimization.pdf)   |
| TPU Architecture           | [pdf](lectures/ML_LectureSp25_tpu.pdf)                     |
| GPU Architecture           | [pdf](lectures/ML_LectureSp25_GPU.pdf)                     |
| CUDA                       | [pdf](lectures/ML_LectureSp25_cuda.pdf)                    |
| Quantization               | [pdf](lectures/ML_LectureSp25_quantization.pdf)            |
| Pruning                    | [pdf](lectures/ML_LectureSp25_pruning.pdf)                 |
| Intro to NLP               | [pdf](lectures/ML_LectureSp25_IntroToNlp.pdf)              |
| Intro to NLP (RNN)         | [pdf](lectures/ML_LectureSp25_IntroToNlp_rnn.pdf)          |
| Transformers - 1           | [pdf](lectures/ML_LectureSp25_transformers.pdf)            |
| Transformers - 2           | [pdf](lectures/ML_LectureSp25_transformers2.pdf)           |
| Transformers Training      | [pdf](lectures/ML_LectureSp25_trainingTransformers.pdf)    |
| FlashAttention             | [pdf](lectures/ML_LectureSp25_flashAttention.pdf)          |
| Federated Learning         | [pdf](lectures/ML_LectureSp25_FederatedLearning.pdf)       |