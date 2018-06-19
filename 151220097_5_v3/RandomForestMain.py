from sklearn import tree
from sklearn import metrics
import numpy as np
from random import randrange
from sklearn.model_selection import StratifiedKFold

class RandomForest:

    def __init__(self, n_estimators = 10):
        self.h = list()
        self.n_estimators = n_estimators

    def sample(self, length):
        index = np.zeros(length, dtype=np.int32)
        for i in range(length):
            r = randrange(0, length)
            index[i] = r
        return index

    def fit(self, XTrain, yTrain):
        for t in range(self.n_estimators):
            clf = tree.ExtraTreeClassifier(max_features="log2")
            index = self.sample(XTrain.shape[0])
            #print(index)
            XSub = XTrain[index]
            ySub = yTrain[index]

            f = clf.fit(XSub, ySub)
            self.h.append(f)

    def predict(self, XTest):
        length = XTest.shape[0]
        result = np.zeros(length)
        prob = np.zeros(length)
        T = len(self.h)
        for t in range(T):
            pred = self.h[t].predict(XTest)
            prob += pred

        for i in range(length):
            if prob[i] >= 0:
                result[i] = 1
            else:
                result[i] = -1
        return result, prob



def loadData():
    XTrain = np.genfromtxt("adult_dataset/adult_train_feature.txt")
    XTest = np.genfromtxt("adult_dataset/adult_test_feature.txt")
    yTrain = np.genfromtxt("adult_dataset/adult_train_label.txt")
    yTest = np.genfromtxt("adult_dataset/adult_test_label.txt")
    return XTrain, XTest, yTrain, yTest


def crossValidation(XTrain, yTrain):
    skf = StratifiedKFold(n_splits=5)
    maxT = 2
    maxAuc = 0.0

    for t in range(1, 51):
        Auc = 0.0
        for trIdx, cvIdx in skf.split(XTrain, yTrain):
            # print(trIdx, cvIdx)
            XTr, yTr = XTrain[trIdx], yTrain[trIdx]
            XCV, yCV = XTrain[cvIdx], yTrain[cvIdx]
            randomforest = RandomForest(n_estimators = t)
            randomforest.fit(XTr, yTr)
            result, yProb = randomforest.predict(XCV)
            Auc += metrics.roc_auc_score(yCV, yProb)
        Auc = Auc / 5
        if Auc > maxAuc:
            maxT = t
            maxAuc = Auc
        print(t, Auc)

    return maxT


if __name__ == '__main__':
    print('RandomForest algorithm start...')
    print('loading data...')
    XTrain, XTest, yTrain, yTest = loadData()
    yTrain = (yTrain - 0.5)*2
    yTest = (yTest - 0.5)*2

    T = 49
    #T = crossValidation(XTrain, yTrain)
    #print(T)

    print('training...')
    randomforest = RandomForest(n_estimators=T)
    randomforest.fit(XTrain, yTrain)
    yPred, yProb = randomforest.predict(XTest)

    print(metrics.classification_report(y_true=yTest, y_pred=yPred))
    print("Random Forest AUC: ", metrics.roc_auc_score(yTest, yProb))



