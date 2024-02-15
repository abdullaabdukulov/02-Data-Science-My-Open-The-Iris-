# Welcome to My-Open-The-Iris
***

# Task

<img src="https://storage.googleapis.com/qwasar-public/track-ds/open_iris.jpg" width="600">

***
<img src="https://storage.googleapis.com/qwasar-public/track-ds/iris_plot_histogram.png" width="350">

***
<img src="https://storage.googleapis.com/qwasar-public/track-ds/iris_plot_scatter.png" width="350">

</code></pre>

<h4>Build multiple different models from different algorithms.</h4>
<pre class=" language-plain"><code class=" language-plain"># DecisionTree
model = DecisionTreeClassifier()

# GaussianNB
model = GaussianNB()

# KNeighbors
model = KNeighborsClassifier()

# LogisticRegression
model = LogisticRegression(solver='liblinear', multi_class='ovr')

# LinearDiscriminant
model = LinearDiscriminantAnalysis()

# SVM
model = SVC(gamma='auto')
</code></pre>

<h2>Description</h2>
<p>Time do to an end-to-end project in data science. which means:</p>
<ol>
<li>Loading the dataset.</li>
<li>Summarizing the dataset.</li>
<li>Visualizing the dataset.</li>
<li>Evaluating some algorithms.</li>
<li>Making some predictions.</li>
</ol>
<p>A must-see example of data science is the <code>iris dataset.</code> We will predict which <code>class of iris plant</code> a plant belongs to based on its characteristics.</p>
<img src="https://storage.googleapis.com/qwasar-public/track-ds/iris_dataset_class_1.jpg" width="300">
<img src="https://storage.googleapis.com/qwasar-public/track-ds/iris_dataset_class_2.jpg" width="300">
<img src="https://storage.googleapis.com/qwasar-public/track-ds/iris_dataset_class_3.jpg" width="300">
<p><em>Iris versicolor - Iris setosa - Iris virginica</em></p>


#### Output should like:
```python
DecisionTree: 0.927191 (0.043263)
GaussianNB: 0.928858 (0.052113)
KNeighbors: 0.937191 (0.056322)
LogisticRegression: 0.920897 (0.043263)
LinearDiscriminant: 0.923974 (0.040110)
SVM: 0.973972 (0.032083)
```

# Installation
<ul>
<li>pandas to load the data</li>
<li>matplotlib to do the visualization</li>
<li>sklearn to do the prediction</li>
</ul>

```bash
  # Write it in your teminal
    pip install -r requirements.txt
```

# Usage 

For working with this project you should run the `my_open_the_iris.ipynb` file. So open it with Jupyter Notebook
