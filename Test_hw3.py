import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import  cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report,confusion_matrix




X_train=train_X[:2301,:]
X_test=train_X[2301:,:]
y_train=test_y[:2301]
y_test=test_y[2301:]
acc1=[]
ap1=[]
recall=[]
f1_s=[]
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred,)
    f1 = f1_score(y_test, y_pred,average='weighted')
    print("Confusion matrix for",name)
    print(confusion_matrix(y_test,y_pred))
    acc1.append(acc)
    ap1.append(ap)
    recall.append(rec)
    f1_s.append(f1)

y_pos = np.arange(len(names))

print(f1_s)
plt.bar(y_pos, f1_s, align='center', alpha=0.5)
plt.xticks(y_pos, names)
plt.ylabel('f1')
plt.title('f1 vs classifier')
plt.show()
print(ap1)
plt.bar(y_pos, ap1, align='center', alpha=0.5)
plt.xticks(y_pos, names)
plt.ylabel('Precision')
plt.title('precision vs classifier')
plt.show()
print(recall)
plt.bar(y_pos, recall, align='center', alpha=0.5)
plt.xticks(y_pos, names)
plt.ylabel('recall')
plt.title('recall vs classifier')
plt.show()