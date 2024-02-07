# 02-Data-Science-My-Open-The-Iris

## Task

The task involves performing analysis on the Iris dataset using various machine learning models to classify iris flowers into different species. The analysis includes data loading, dataset summary, univariate and multivariate visualization, and testing different classification models.

## Description

The provided Python script is written in a Jupyter Notebook (.ipynb) format and utilizes popular machine learning libraries to perform the following tasks:

1. **Load Dataset**: The script loads the Iris dataset from a specified URL using Pandas.

2. **Summarize Dataset**: It provides a summary of the dataset, including its shape, the first 10 rows, statistical summary, and the distribution of instances among different classes.

3. **Univariate Visualization**: The script generates histograms for each attribute in the dataset, providing a visual representation of the distribution of values.

4. **Multivariate Visualization**: It creates a scatter matrix, allowing visualization of the relationships between pairs of attributes.

5. **Test Different Models**: The script trains and evaluates various machine learning models, including Decision Tree, Gaussian Naive Bayes, K-Nearest Neighbors, Logistic Regression, Linear Discriminant Analysis, and Support Vector Machine (SVM). The evaluation is based on cross-validation accuracy.

## Installation

To run the Jupyter Notebook, ensure you have the required libraries installed. You can install them using:

```bash
pip install pandas matplotlib scikit-learn
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/iris-dataset-analysis.git
   ```

2. Navigate to the project directory:

   ```bash
   cd iris-dataset-analysis
   ```

3. Open the Jupyter Notebook:

   ```bash
   jupyter notebook iris_analysis.ipynb
   ```

   Run each cell in the notebook to execute the analysis. The results, including the accuracy of different machine learning models, will be displayed within the notebook.