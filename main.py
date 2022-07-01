import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

iris = datasets.load_iris()
x = iris.data
y = iris.target
class_names = iris.target_names

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0)

classifier = svm.SVC(kernel="linear", C=0.02).fit(x_train, y_train)

np.set_printoptions(precision=2)

title_options = [("Confusion matrix, without normalization", None),
                 ("Normalized confusion matrix","true")]

for title, normalize in title_options:
    display = ConfusionMatrixDisplay.from_estimator(
        classifier,
        x_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    display.ax_.set_title(title)
    print(title)
    print(display.confusion_matrix)

plt.show()
