import numpy as np
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


from sklearn.svm import LinearSVC






raw_data = open("spambase.csv", 'rt')
df = np.loadtxt(raw_data, delimiter=",")


X=df[:,:56]
Y = df[:,57]
kf = KFold(n_splits=10, shuffle=False)
def svm():
    classifiers = [LinearSVC(C=0.01),LinearSVC(C=0.01),LinearSVC( C=1), LinearSVC(C=10),LinearSVC(C=100)]
    names = "Linear SVM"

    f1 = []
    f_average = []

    for name, clf in zip(names, classifiers):
        for train_index, test_index in kf.split(df):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            # Prepare the plot for this classifier

            clf.fit(x_train, y_train)
            # Predict
            y_pred = clf.predict(x_test)
            # print y_pred
            f1.append(f1_score(y_test, y_pred, average='weighted'))
            # print sum(int(f1))
        f_average.append((sum(f1) / (len(f1))))

    print f_average

    C=[0.01,0.1,1,10,100]

    plt.title("SVM")
    plt.plot(C,f_average)

    plt.show()
    return f_average
f_average_svm=svm()

def gini():
    classifiers_DT = [DecisionTreeClassifier(criterion="gini", max_leaf_nodes=2),DecisionTreeClassifier(criterion="gini", max_leaf_nodes=5),
                      DecisionTreeClassifier(criterion="gini", max_leaf_nodes=10),
                      DecisionTreeClassifier(criterion="gini", max_leaf_nodes=20)]
    DT_name = "Decision Tree"
    f1 = []
    f_average=[]

    for name, clf in zip(DT_name, classifiers_DT):
        for train_index ,test_index in kf.split(df):
            x_train,x_test=X[train_index],X[test_index]
            y_train,y_test=Y[train_index],Y[test_index]

        # Prepare the plot for this classifier

            clf.fit(x_train, y_train)
        # Predict
            y_pred = clf.predict(x_test)
        #print y_pred
            f1.append(f1_score(y_test, y_pred, average='weighted'))
    #print sum(int(f1))
        f_average.append((sum(f1)/(len(f1))))

    print f_average
    K=[2,5,10,20]
    plt.title("GINI")
    plt.plot(K,f_average)

    plt.show()
    return f_average
f_average=gini()
#Using Information Gain......
def ig():
    classifiers_DT_info = [DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=2),
                           DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=5),
                           DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=10),
                           DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=20)]
    DT_name = "Decision Tree"
    f1 = []
    K = [2, 5, 10, 20]
    f_average_ig=[]
    for name, clf in zip(DT_name, classifiers_DT_info):
        for train_index ,test_index in kf.split(df):
            x_train,x_test=X[train_index],X[test_index]
            y_train,y_test=Y[train_index],Y[test_index]

        # Prepare the plot for this classifier

            clf.fit(x_train, y_train)
        # Predict
            y_pred = clf.predict(x_test)
        #print y_pred
            f1.append(f1_score(y_test, y_pred, average='weighted'))
    #print sum(int(f1))
        f_average_ig.append((sum(f1)/(len(f1))))

    print f_average_ig
    plt.title("IG")
    plt.plot(K,f_average_ig)

    plt.show()
    return f_average_ig
f_average_ig=ig()

def compare():
    C = [0.01, 0.1, 1, 10, 100]
    K = [2, 5, 10, 20]
    max_svm=f_average_svm.index(max(f_average_svm))
    max_dt_gini = f_average.index(max(f_average))
    max_dt_ig = f_average_ig.index(max(f_average_ig))
    best_svm=C[max_svm]
    best_k_gini = K[max_dt_gini]
    best_k_ig = K[max_dt_ig]
    train_set = df[:len(df) / 2, :]
    test_set = df[len(df) - len(df) / 2:, :]
    X_train = train_set[:, :56]
    Y_train = train_set[:, 57]
    X_test=  test_set[:,:56]
    Y_test= test_set[:,57]
    f_measure=[]
    class_precision=[]
    class_recall=[]
    classifier = [LinearSVC(C=best_svm),DecisionTreeClassifier(criterion="gini", max_leaf_nodes=best_k_gini),DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=best_k_ig),LinearDiscriminantAnalysis()]
    name = "Linear SVM", "Decision Tree","Decision Tree","LDA"
    for name, clf in zip(name, classifier):

        clf.fit(X_train, Y_train)
        # Predict
        Y_pred = clf.predict(X_test)
        #print y_pred
        f_measure.append(f1_score(Y_test, Y_pred, average='weighted'))
        #print sum(int(f1))
        class_precision.append(average_precision_score(Y_test, Y_pred))
        class_recall.append(recall_score(Y_test, Y_pred, average='weighted'))
        print("Confusion Matrix for",name)
        print(confusion_matrix(Y_test, Y_pred))
        print("")
    objects = ('SVM', 'Decision Tree(GINI)', 'Decision Tree(IG)','LDA')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, f_measure, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('F-Measure')
    plt.title('Classifiers V/S F-Measure')
    plt.show()

    plt.bar(y_pos, class_precision, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Precision')
    plt.title('Classifiers V/S Precision')
    plt.show()

    plt.bar(y_pos, class_recall, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Recall')
    plt.title('Classifiers V/S Recall')
    plt.show()
    return f_measure,class_recall,class_precision

f_measure,class_precision,class_recall=compare()

def random_forest(f_measure,class_recall,class_precision):
    confusion_matrix=[]
    train_set = df[:len(df) / 2, :]
    test_set = df[len(df) - len(df) / 2:, :]
    X_train = train_set[:, :56]
    Y_train = train_set[:, 57]
    X_test = test_set[:, :56]
    Y_test = test_set[:, 57]
    classifier = [RandomForestClassifier(max_leaf_nodes=20, random_state=0)]
    name = "Random Forest"
    for name, clf in zip(name, classifier):
        clf.fit(X_train, Y_train)
        # Predict
        Y_pred = clf.predict(X_test)
        # print y_pred
        f_measure.append(f1_score(Y_test, Y_pred, average='weighted'))
        # print sum(int(f1))
        class_precision.append(average_precision_score(Y_test, Y_pred))
        class_recall.append(recall_score(Y_test, Y_pred, average='weighted'))

    objects = ('SVM', 'Decision Tree(GINI)', 'Decision Tree(IG)','LDA','Random Forest')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, f_measure, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('F-Measure')
    plt.title('Classifiers V/S F-Measure')
    plt.show()

    plt.bar(y_pos, class_precision, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Precision')
    plt.title('Classifiers V/S Precision')
    plt.show()

    plt.bar(y_pos, class_recall, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Recall')
    plt.title('Classifiers V/S Recall')
    plt.show()
    return f_measure, class_recall, class_precision

random_forest(f_measure,class_precision,class_recall)