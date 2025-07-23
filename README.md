# Thematic and Emotional Analysis of Song Lyrics using Machine Learning

## Project Overview
This is a high-performance deep learning application that analyzes song lyrics to classify them into emotional and thematic categories such as *romantic*, *sad*, and *violent*. This project leverages **BERT embeddings** and a **custom neural network** architecture for classification, and was built using TensorFlow and HuggingFace Transformers on a high-performance computing (HPC) cluster.

## Motivation
Music reflects a wide range of human emotions and themes. By analyzing lyrical content, we can explore patterns in cultural expression across time and genre. This project uses machine learning and NLP to:
- Discover dominant lyrical themes.
- Identify emotional tones across decades of music.
- Enable future applications like mood-based song recommendations.

## Dataset
- **Source**: Kaggle dataset of song lyrics from 1950 to 2019.
- **Preprocessing**: 
  - Cleaned to remove special characters, duplicates, and blank entries.
  - Filtered to include only English lyrics.
  - Labeled into themes (e.g., romantic, sad, violent).

## Model Architecture
- **Input**: Preprocessed lyrics converted into `input_ids` and `attention_mask` using BERT tokenizer.
- **Embedding**: BERT (`bert-base-uncased`) model generates contextual embeddings.
- **Neural Network**:
  - Dense layer with 512 units (ReLU activation).
  - Output layer with softmax activation for multi-class classification.
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam with exponential decay learning rate
- **Metrics**: Accuracy

## 🛠️ Implementation
### Key Technologies:
- Python, TensorFlow, HuggingFace Transformers
- NumPy, Pandas, Scikit-learn
- Jupyter Notebook
- SLURM Scheduler (for HPC job execution)

### Training Setup:
- Dataset split: 90% Training, 10% Validation
- Batch Size: 32
- Epochs: 10
- Parallelized training on GPU nodes via SLURM

## Results
- Achieved high validation accuracy.
- Clear class-wise separation observed using softmax probabilities.
- Strong correlation between lyric content and predicted themes.
