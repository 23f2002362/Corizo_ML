# Corizo AI/ML Learning Program

A comprehensive machine learning project repository containing two major projects and supporting materials from the Corizo AI learning program.

##  Projects

### Project 1: Cardiovascular Disease Prediction
- **Objective**: Predict cardiovascular disease using patient medical data
- **Dataset**: `cardio_train.csv` (70,002+ records)
- **Type**: Binary Classification
- **Features**: Patient demographics, medical measurements, lifestyle factors
- **Target**: Cardiovascular disease presence (0/1)

### Project 2: Spotify Songs' Genre Classification  
- **Objective**: Classify music tracks into different genres
- **Dataset**: `spotify dataset.csv` 
- **Type**: Multi-class Classification
- **Features**: Audio characteristics, track metadata, popularity metrics
- **Genres**: Pop, Hip Hop, Dance Pop, Indie Poptimism, Electropop, etc.

##  Repository Structure

```
├── Project 1 - Cardiovascular Disease Prediction/
│   ├── cardio_train.csv
│   └── Project documentation
├── Project 2 - Spotify Songs' Genre Segmentation/
│   ├── spotify dataset.csv
│   └── Project documentation  
├── class/
│   ├── ai_august_12.py          # Neural Networks (MNIST)
│   ├── ai_august_13.py          # ML + NLP + Sentiment Analysis
│   └── AI Aug *.html            # Class session notes (13 sessions)
├── csves/
│   ├── Iris.csv                 # Classic ML dataset
│   ├── Mall_Customers.csv       # Customer segmentation
│   ├── House.csv               # Real estate prediction
│   └── Other practice datasets
└── cardio_train.csv            # Main cardiovascular dataset
```

##  Key Learning Areas

### Machine Learning Concepts
- **Neural Networks & Deep Learning**
  - Multi-layer perceptrons
  - Activation functions (ReLU, Softmax)
  - Model optimization (Adam optimizer)
  
- **Classification Tasks**
  - Binary classification (Disease prediction)
  - Multi-class classification (Genre/Species classification)
  - Performance evaluation metrics

- **Data Preprocessing**
  - Feature scaling and normalization
  - Data encoding and transformation
  - Train-test splitting

### Technical Stack
- **Deep Learning**: TensorFlow, Keras
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **NLP**: spaCy, TextBlob

### Practical Applications
- **Healthcare**: Cardiovascular disease prediction
- **Music Technology**: Genre classification and recommendation
- **Natural Language Processing**: Sentiment analysis
- **Computer Vision**: Handwritten digit recognition (MNIST)

##  Getting Started

### Prerequisites
```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn spacy textblob seaborn
```

### Running the Projects

1. **Cardiovascular Disease Prediction**
   ```python
   # Load and explore the dataset
   import pandas as pd
   df = pd.read_csv('cardio_train.csv', sep=';')
   ```

2. **Neural Network Training (MNIST)**
   ```python
   python class/ai_august_12.py
   ```

3. **Multi-task ML (Iris + NLP)**
   ```python
   python class/ai_august_13.py
   ```

##  Datasets Overview

| Dataset | Records | Features | Task Type |
|---------|---------|----------|-----------|
| Cardiovascular | 70,002 | 11 | Binary Classification |
| Spotify Songs | ~32,000 | 23 | Multi-class Classification |
| Iris | 152 | 4 | Multi-class Classification |
| Mall Customers | 201 | 4 | Clustering/Segmentation |
| House Prices | Variable | 3+ | Regression |

##  Learning Outcomes

- Hands-on experience with real-world datasets
- Implementation of neural networks from scratch
- Understanding of various ML algorithms and their applications
- Data preprocessing and feature engineering skills
- Model evaluation and performance metrics
- Integration of multiple ML libraries and tools

##  Class Materials

The `class/` directory contains 13 comprehensive learning sessions covering:
- Basic Python for ML
- Data manipulation with Pandas
- Neural network fundamentals
- Deep learning with TensorFlow/Keras
- Natural Language Processing
- Sentiment Analysis
- Computer Vision basics

##  Contributing

This repository represents educational work from the Corizo AI program. Feel free to:
- Explore the code and datasets
- Try different ML approaches
- Enhance existing models
- Add new features or visualizations

##  License

Educational project - please respect dataset licensing and attribution requirements.

---

**Created as part of the Corizo AI/ML Learning Program**
