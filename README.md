# Fake News Detection Using BERT and Capsule Networks

##  Overview

The rapid growth of digital media and social networking platforms has made misinformation and fake news a significant global challenge. This project presents an AI-powered Fake News Detection system that automatically classifies news articles and social media content as **Real** or **Fake** using a hybrid deep learning architecture combining **BERT (Bidirectional Encoder Representations from Transformers)** and **Capsule Networks**.

The system leverages the contextual understanding capabilities of BERT and the hierarchical feature extraction strength of Capsule Networks to improve classification performance and robustness.

##  Objectives

* Detect and classify fake news automatically.
* Reduce the spread of misinformation.
* Provide confidence-based predictions.
* Support verification of textual news content.
* Explore advanced NLP and Deep Learning techniques for misinformation detection.


##  System Architecture

Input News Article

↓

Text Preprocessing

↓

BERT Encoder

↓

Capsule Network

↓

Classification Layer

↓

Prediction (Real / Fake / Misleading)

↓

Confidence Score


##  Model Architecture

### BERT Encoder

* Utilizes pre-trained BERT for contextual text representation.
* Captures semantic relationships and contextual meaning of words.
* Generates high-quality embeddings for downstream classification.

### Capsule Network

* Extracts hierarchical and spatial relationships from BERT embeddings.
* Preserves important feature information often lost in traditional neural networks.
* Enhances classification performance for complex news content.

### Classification Layer

* Produces final prediction probabilities.
* Outputs:

  * Real News
  * Fake News

##  Data Preprocessing

The preprocessing pipeline includes:

* Text normalization
* Lowercase conversion
* URL removal
* Special character removal
* Whitespace normalization
* Text length optimization
* Tokenization using BERT Tokenizer

##  Performance

| Metric   | Score       |
| -------- | ----------- |
| Accuracy | 89.5%       |
| F1 Score | 89.4%       |
| Classes  | Real / Fake |

The hybrid BERT + Capsule Network architecture demonstrated improved performance compared to a baseline BERT-only model.

---

## Technologies Used

### Programming Language

* Python

### Deep Learning

* PyTorch

### Natural Language Processing

* Transformers
* BERT

### Data Processing

* Pandas
* NumPy

### Evaluation

* Scikit-learn


## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/fake-news-detection-bert-capsule.git
cd fake-news-detection-bert-capsule
```

### Install Dependencies

```bash
pip install -r requirements.txt


##  Academic Contribution

This project was developed as a Final Year Project in Software Engineering and demonstrates the practical application of:

* Natural Language Processing (NLP)
* Deep Learning
* Transformer Models
* Capsule Networks
* AI-based Information Verification

---

##  Author

**Muhammad Ahmad**

Software Engineering Graduate
COMSATS University Islamabad

Interests:

* Artificial Intelligence
* Machine Learning
* Natural Language Processing
* Large Language Models (LLMs)
* Deep Learning
* Data Science

---

##  Acknowledgements
Special thanks to the open-source community and the researchers behind BERT, Capsule Networks, and modern NLP frameworks that made this work possible.
