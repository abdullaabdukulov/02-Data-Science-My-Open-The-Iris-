
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Bu funksiya iris.csv nomli csv faylni o'qib beryabdi
def load_dataset():
    url = "iris.csv"  
    dataset = pd.read_csv(url)
    return dataset

# Bu funksiya datasetning o'lchami, boshidagi 10 ta qatorini ,dataset haqida malumot va clas turi bo'yicha malumotlarni chiqaradi
def summarize_dataset(dataset):
    print("Dataset dimension:")
    print(dataset.shape)
    print("\nFirst 10 rows of dataset:")
    print(dataset.head(10))
    print("\nStatistical summary:")
    print(dataset.describe())
    print("\nClass Distribution:")
    print(dataset.groupby('class').size())

# Bu funksiya datasetning har bir ustunini histogram qilib chiqaradi

def print_plot_univariate(dataset):
    dataset.hist()
    plt.show()
# Bu funksiya bir o'zgaruvchini boshqa bir o'zgaruvchi bilan  munosabatini tekshiradi va histogram qilib ko'rsatadi
def print_plot_multivariate(dataset):
    scatter_matrix(dataset)
    plt.show()

# Bu yozgan funksiyamning maqsadi kategoriyalashtirish madellari bilan datasetni turini aniqlash 
# uchun bir xil hodisa borligini test qilish . datasetni train_test_split bilan train va validation qismlarga ajratish
# modellarni turish va baholash cross_val_score orqali kross-validatsiya natijalarni hisoblash
# Har bir modelni train qilib, validation qismiga ishlatib, natijalarni hisoblab chiqaradi.
def my_print_and_test_models(dataset):
    array = dataset.values
    X = array[:, 0:4]
    y = array[:, 4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
    models = [
        ('DecisionTree', DecisionTreeClassifier()),
        ('GaussianNB', GaussianNB()),
        ('KNeighbors', KNeighborsClassifier()),
        ('LogisticRegression', LogisticRegression(solver='liblinear', multi_class='ovr')),
        ('LinearDiscriminant', LinearDiscriminantAnalysis()),
        ('SVM', SVC(gamma='auto'))
    ]
    for model_name, model in models:
        kfold = 10
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        print(f'{model_name}: {cv_results.mean():.6f} ({cv_results.std():.6f})')
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        print(f'\n{model_name} - Classification Report:')
        print(classification_report(Y_validation, predictions))
        print(f'\n{model_name} - Confusion Matrix:')
        print(confusion_matrix(Y_validation, predictions))
        print('\n' + '='*50)


iris_dataset = load_dataset()
summarize_dataset(iris_dataset)
print_plot_univariate(iris_dataset)
print_plot_multivariate(iris_dataset)
my_print_and_test_models(iris_dataset)
