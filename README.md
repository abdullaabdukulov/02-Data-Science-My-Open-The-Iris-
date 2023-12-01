# 02-Data-Science-My-Open-The-Iris

<div class="row">
<div class="col tab-content">
<div class="tab-pane active show" id="subject" role="tabpanel">
<div class="row">
<div class="col-md-12 col-xl-12">
<div class="markdown-body">
<p class="text-muted m-b-15">
</p><h1>My Open The Iris</h1>
<p>Remember to git add &amp;&amp; git commit &amp;&amp; git push each exercise!</p>
<p>We will execute your function with our test(s), please DO NOT PROVIDE ANY TEST(S) in your file</p>
<p>For each exercise, you will have to create a folder and in this folder, you will have additional files that contain your work. Folder names are provided at the beginning of each exercise under <code>submit directory</code> and specific file names for each exercise are also provided at the beginning of each exercise under <code>submit file(s)</code>.</p>
<hr>
<table>
<thead>
<tr>
<th>My Open The Iris</th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td>Submit directory</td>
<td>.</td>
</tr>
<tr>
<td>Submit file</td>
<td>my_open_the_iris.ipynb</td>
</tr>
</tbody>
</table>
<h3>Description</h3>
<img src="https://storage.googleapis.com/qwasar-public/track-ds/open_iris.jpg" width="600">
<p>Open the iris!</p>
<p><em>A common mistake businesses make is to assume machine learning is magic, so it's okay to skip thinking about what it means to do the task well.</em></p>
<h2>Introduction</h2>
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
<h3>Where to get started?</h3>
<p>Environment. We will use Jupyter.</p>
<p>In Data Science, the winning combo is pandas (and/or numpy), matplotlib, sklearn (and/or keras).
In this project, we will use:</p>
<ul>
<li>pandas to load the data</li>
<li>matplotlib to do the visualization</li>
<li>sklearn to do the prediction</li>
</ul>
<h3>Load dataset</h3>
<pre class=" language-plain"><code class=" language-plain">url = "URL"
dataset = read_csv(url)
</code></pre>
<h3>Summarizing the dataset</h3>
<p>A - Printing dataset dimension</p>
<pre class=" language-plain"><code class=" language-plain">print(dataset.shape)
# should something like: (150, 5)
</code></pre>
<p>B - It is also always a good idea to eyeball your data.</p>
<pre class=" language-plain"><code class=" language-plain">print(dataset.head(20))
</code></pre>
<p>C - Statistical Summary
The statistical summary includes the count, mean, the min and max values, and some percentiles.</p>
<pre class=" language-plain"><code class=" language-plain">print(dataset.describe())
</code></pre>
<p>D - Class Distribution
Group by to see how our data are distributed.</p>
<pre class=" language-plain"><code class=" language-plain">print(dataset.groupby('class').size())
</code></pre>
<h3>Visualization</h3>
<p>After having a basic idea about our dataset, we need to extend it with some visualizations.</p>
<p>For this dataset, we will focus on two types of plots:</p>
<ul>
<li>Univariate plots to better understand each attribute.</li>
<li>Multivariate plots to better understand the relationships between attributes.</li>
</ul>
<p>A - Univariate</p>
<pre class=" language-plain"><code class=" language-plain">from pandas import read_csv
from matplotlib import pyplot

dataset.hist()
pyplot.show()
</code></pre>
<img src="https://storage.googleapis.com/qwasar-public/track-ds/iris_plot_histogram.png" width="350">
<p>It looks like perhaps two of the input variables have a Gaussian distribution. This is useful to note as we can use algorithms that can exploit this assumption.</p>
<p>B - Multivariate</p>
<pre class=" language-plain"><code class=" language-plain">from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

scatter_matrix(dataset)
pyplot.show()
</code></pre>
<img src="https://storage.googleapis.com/qwasar-public/track-ds/iris_plot_scatter.png" width="350">
<p>We can note the diagonal grouping of some pairs of attributes. It suggests a high correlation and a predictable relationship. :-)</p>
<h3>Building our code to evaluate some algorithms</h3>
<p>it is time to create some data models and estimate their accuracy.</p>
<p>Here is what we are going to cover in this step:</p>
<p>Separate a validation dataset.</p>
<pre class=" language-plain"><code class=" language-plain">array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
</code></pre>
<h3>Experiment!</h3>
<p>Build multiple different models from different algorithms.</p>
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
<p>How to run the model?</p>
<pre class=" language-plain"><code class=" language-plain">cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
</code></pre>
<h3>Improving</h3>
<img src="https://storage.googleapis.com/qwasar-public/track-ds/iris_meme_theory_vs_reality.jpg" width="350">
<p>Improving your data and your model is an iterative process, and you will have to loop through this process repeatedly.</p>
<p>Now it's time to do it!</p>
<h2>Technical specifications</h2>
<p>You will create an end-to-end analysis of the dataset.</p>
<h3>Part I Load data</h3>
<p>Create a function <code>load_dataset()</code>. It doesn't take any parameter. You will load the dataset and returns it.</p>
<h3>Part II Summarizing the dataset</h3>
<p>Summarizing the dataset:
Create a function <code>summarize_dataset(dataset)</code>, it will print (in this order):</p>
<pre class=" language-plain"><code class=" language-plain">its shape
its ten first lines
its statistical summary
Its distribution
</code></pre>
<p>Example:</p>
<pre class=" language-plain"><code class=" language-plain">Dataset dimension:
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
</code></pre>
<h3>Part III</h3>
<p>Create two functions <code>print_plot_univariate(dataset)</code> and <code>print_plot_multivariate(dataset)</code>. Each function will setup and show its corresponding plot.</p>
<h3>Part IV</h3>
<p>Create a function <code>my_print_and_test_models(dataset)</code>, it will (in this order)
DecisionTree, GaussianNB, KNeighbors, LogisticRegression, LinearDiscriminant, and SVM</p>
<p>Remember to split your dataset in two: train and validation.</p>
<p>Following this format:</p>
<pre class=" language-plain"><code class=" language-plain"># print('PERCENTs: PERCENTf (PERCENTf)' PERCENT (model_name, cv_results.mean(), cv_results.std()))
DecisionTree: 0.927191 (0.043263)
GaussianNB: 0.928858 (0.052113)
KNeighbors: 0.937191 (0.056322)
LogisticRegression: 0.920897 (0.043263)
LinearDiscriminant: 0.923974 (0.040110)
SVM: 0.973972 (0.032083)
</code></pre>
<h2>Iris Dataset</h2>
<p><a href="https://storage.googleapis.com/qwasar-public/track-ds/iris.csv" target="_blank">Iris Dataset</a></p>
<p>Gandalf will not accept any <code>pip install XXXX</code> inside your file.</p>

<p></p>
</div>

</div>
</div>
</div>
<div class="tab-pane" id="resources" role="tabpanel">
<div class="row">
<div class="col-xl-12">
<div class="row text-center">
<div class="col">
<a target="_blank" href="https://towardsdatascience.com/3-great-design-patterns-for-data-science-workflows-d3bf162d74e6/archive/cs/cs107/cs107.1174/guide_make.html">https://towardsdatascience.com/3-great-design-patterns-for-data-science-workflows-d3bf162d74e6/archive/cs/cs107/cs107.1174/guide_make.html</a>
</div>
</div>

</div>
</div>
</div>
</div>
</div>
