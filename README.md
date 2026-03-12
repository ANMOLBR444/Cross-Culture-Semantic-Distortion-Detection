# Semantic Distortion Detection using LLM Canonicalization

## Overview
This project implements a **semantic distortion detection system** that identifies inconsistencies between the intended meaning of a sentence and its surface linguistic form. The system follows a **teacher-guided approach** where a Large Language Model (LLM) generates a simplified canonical representation of the input text. The original and canonicalized sentences are then compared using **sentence embeddings and cosine similarity** to detect semantic distortions.

The application includes an **interactive Streamlit interface** that allows users to input sentences and analyze semantic consistency in real time.

---

## Motivation
Human language frequently includes **idioms, metaphors, sarcasm, and stylistic expressions**, which can make semantic interpretation difficult for traditional NLP systems. Many existing approaches depend on **supervised learning and labeled datasets**, which can limit scalability and generalization across diverse linguistic expressions.

This project proposes a **training-free inference-time framework** that combines:
- LLM-based canonicalization
- Sentence embeddings
- Cosine similarity

to detect semantic distortions without requiring additional model training.

---

## Methodology

The system follows a **three-stage pipeline**:

### 1. Canonicalization
The user inputs a sentence through the Streamlit interface. The sentence is sent to an **Ollama-hosted LLM**, which generates a **canonicalized version** that preserves the original meaning while reducing idiomatic or stylistic complexity.

### 2. Sentence Embeddings
Both the **original sentence** and the **canonicalized sentence** are passed to a **Sentence Transformer model**, which converts each sentence into a **high-dimensional vector embedding** representing semantic meaning.

### 3. Semantic Similarity
The cosine similarity between the two embeddings is calculated to measure semantic alignment.

- **High similarity → Semantic consistency**
- **Low similarity → Possible semantic distortion**


---

## Technologies Used

- **Python**
- **Streamlit** – Interactive web interface
- **Ollama** – Local LLM inference for canonicalization
- **Sentence Transformers** – Sentence embeddings
- **Scikit-learn / NumPy** – Cosine similarity computation

---

## Features

- LLM-based **semantic canonicalization**
- **Inference-time semantic distortion detection**
- **Model-agnostic architecture**
- Real-time **Streamlit interface**
- Interpretable **cosine similarity scoring**
- Lightweight and modular NLP pipeline

