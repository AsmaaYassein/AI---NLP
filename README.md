# Email Spam Detection Project

## Overview

This project focuses on detecting email spam using AI and Natural Language Processing (NLP) techniques. Various machine learning models are implemented and compared to identify the best performing classifier for the task. The project uses labeled email datasets to train and evaluate models based on their accuracy, precision, recall, F1-score, and other performance metrics.

---

## Project Files

- **`.gitattributes`**: Configuration for version control and GitHub compatibility.
- **`AI proposal.pdf`**: A document detailing the initial project proposal, goals, and methodology.
- **`Email Spam Detection-AI&NLP project.ipynb`**: Jupyter Notebook containing the code, data preprocessing, model training, and evaluation.
- **`README.md`**: Current file containing the project description and details.
- **`Spam_Detection_Presentation.pdf`**: A presentation summarizing the project, including results and insights.

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - Data Analysis: `numpy`, `pandas`
  - Visualization: `matplotlib`, `seaborn`
  - NLP: `nltk` (stopwords, tokenization, lemmatization)
  - Machine Learning: `scikit-learn` (models, metrics, preprocessing)

---

## Data Preprocessing

1. **Text Cleaning**:
   - Removal of special characters using `re`.
   - Tokenization and lemmatization with `nltk`.

2. **Feature Extraction**:
   - Text converted into numerical representation using `TfidfVectorizer`.

3. **Dataset Splitting**:
   - Data split into training and testing sets using `train_test_split`.

4. **Label Encoding**:
   - Labels are encoded into numerical values for model compatibility.

---

## Models Implemented

Several machine learning models were implemented and evaluated for spam detection:

1. **Naive Bayes**
2. **Logistic Regression**
3. **K-Nearest Neighbors (KNN)**
4. **Decision Tree**
5. **Random Forest**

---

## Performance Metrics

The following metrics were used to evaluate model performance:
- **Accuracy**
- **Mean Squared Error**
- **Classification Report** (Precision, Recall, F1-Score)
- **Confusion Matrix**

---

## Model Comparisons

| Model                 | Accuracy (%) | Mean Squared Error | Key Observations                              |
|-----------------------|--------------|--------------------|-----------------------------------------------|
| **Naive Bayes**       | 98.16        | 0.018              | High precision and recall for spam detection. |
| **Logistic Regression**| 96.61        | 0.034              | Reliable, but less recall compared to Naive Bayes. |
| **KNN**               | 91.67        | 0.083              | Struggles with detecting spam accurately.     |
| **Decision Tree**     | 96.99        | 0.030              | Balanced precision and recall.                |
| **Random Forest**     | 98.16        | 0.018              | Best performance with low error rate.         |

---

## Best Model

The **Random Forest** and **Naive Bayes** classifiers performed the best, achieving the highest accuracy (98.16%) and the lowest mean squared error (0.018). Both models demonstrated excellent spam detection capabilities.

---

## How to Use

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook to train and evaluate the models:
   ```bash
   jupyter notebook Email Spam Detection-AI&NLP project.ipynb
   ```

---

## Future Improvements

- **Hyperparameter Tuning**:
  Optimize model parameters for even better performance.
  
- **Advanced Feature Engineering**:
  Explore additional NLP techniques like Word Embeddings or Topic Modeling.

- **Deployment**:
  Create a web-based or mobile application to classify emails in real-time.

---

## Acknowledgments

Special thanks to the datasets and tools that enabled this project. Also, gratitude to the open-source community for providing excellent libraries like `nltk` and `scikit-learn`. 

--- 

### License

This project is licensed under the MIT License. See the LICENSE file for details. 
