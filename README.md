# Description

Time do to an end-to-end project in data science. which means:

Loading the dataset.
Summarizing the dataset.
Visualizing the dataset.
Evaluating some algorithms.
Making some predictions.
A must-see example of data science is the iris dataset. We will predict which class of iris plant a plant belongs to based on its characteristics.

# Task

Part I Load data
Create a function load_dataset(). It doesn't take any parameter. You will load the dataset and returns it.

Part II Summarizing the dataset
Summarizing the dataset: Create a function summarize_dataset(dataset), it will print (in this order):

its shape
its ten first lines
its statistical summary
Its distribution
Example:

Dataset dimension:
(37, 5)
First 10 rows of dataset:
sepal-length  sepal-width  petal-length  petal-width        class
0           5.1          3.5           1.4          0.2  Iris-setosa
1           4.9          3.0           1.4          0.2  Iris-setosa
2           4.7          3.2           1.3          0.2  Iris-setosa
3           4.6          3.1           1.5          0.2  Iris-setosa
4           5.0          3.6           1.4          0.2  Iris-setosa
5           5.4          3.9           1.7          0.4  Iris-setosa
6           4.6          3.4           1.4          0.3  Iris-setosa
7           5.0          3.4           1.5          0.2  Iris-setosa
8           4.4          2.9           1.4          0.2  Iris-setosa
9           4.9          3.1           1.5          0.1  Iris-setosa


Statistical summary:
sepal-length  sepal-width  petal-length  petal-width
count    12.000000   12.000000    12.000000   12.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25        5.100000     2.800000      1.600000     0.300000
50        5.800000     3.000000      4.350000     1.300000
75        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000


Class Distribution:
class
Iris-setosa        12
Iris-versicolor    12
Iris-virginica     13
dtype: int64

Part III

# Install 

pip install numpy 
pip install pandas 
pip install matplotlib 
pip install sklearn 

# Usage