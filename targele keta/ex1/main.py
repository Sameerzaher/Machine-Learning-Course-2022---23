from sklearn import tree
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import datasets
#from sklearn.metrics import f1_score
#import some data to play with
iris = datasets.load_iris()
mylist = []
#do loop
clf = tree.DecisionTreeClassifier()
clf.max_depth = 1
clf.criterion = 'entropy'
clf = clf.fit(iris.data, iris.target)
print("Decision Tree: ")
accuracy = cross_val_score(clf, iris.data, iris.target, scoring='accuracy', cv=10)
print("Average Accuracy of DT with depth ", clf.max_depth, " is: ", round(accuracy.mean(), 3))
#mylist.append(accuracy.mean())loop,
# can be used to plot later…
precision = cross_val_score(clf, iris.data, iris.target, scoring='precision_weighted', cv=10)
print("Average precision_weighted of DT with depth ", clf.max_depth, " is: ", round(precision.mean(), 3))

#f1=f1_score(clf, iris.data, iris.ta)
X = range(10)
plt.plot(X, [x * x for x in X])
plt.xlabel("This is the X axis")
plt.ylabel("This is the Y axis")
#plt.show()