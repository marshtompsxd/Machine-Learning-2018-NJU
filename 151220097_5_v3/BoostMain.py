from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np

class AdaBoost:

    def __init__(self, n_estimators = 10):
        self.h = list()
        self.alpha = np.zeros(n_estimators)
        self.n_estimators = n_estimators

    def fit(self, XTrain, yTrain):
        length = XTrain.shape[0]
        D = np.ones(length) / length

        for t in range(self.n_estimators):

            clf = tree.DecisionTreeClassifier()
            f = clf.fit(XTrain, yTrain, sample_weight=D)
            pred = f.predict(XTrain)

            error = np.average(pred != yTrain, weights=D)

            #print(error)

            if error > 0.5:
                break
            elif error <= 0:
                self.h.append(f)
                self.alpha[t] = 1
                break

            self.h.append(f)

            self.alpha[t] = 0.5 * np.log((1 - error) / error)
            D = D*np.exp(-1*pred*yTrain*self.alpha[t])
            D = D/np.sum(D)


    def predict(self, XTest):
        length = len(self.h)
        pred = np.zeros((length, XTest.shape[0]))
        result = np.zeros(XTest.shape[0])
        prob = np.zeros(XTest.shape[0])

        for t in range(0, length):
            pred[t] = self.h[t].predict(XTest)*self.alpha[t]
            prob += pred[t]



        for i in range(XTest.shape[0]):
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
            adaboost = AdaBoost(n_estimators = t)
            adaboost.fit(XTr, yTr)
            result, prob = adaboost.predict(XCV)
            Auc += metrics.roc_auc_score(yCV, prob)
        Auc = Auc / 5
        if Auc > maxAuc:
            maxT = t
            maxAuc = Auc
        print(t, Auc)

    return maxT

if __name__ == '__main__':

    print('AdaBoost algorithm start...')
    print('loading data...')
    XTrain, XTest, yTrain, yTest = loadData()

    yTrain = (yTrain - 0.5)*2
    yTest = (yTest - 0.5)*2

    T = 25
    #T = crossValidation(XTrain, yTrain)
    #print(T)

    print('training...')
    adaboost = AdaBoost(n_estimators=T)
    adaboost.fit(XTrain, yTrain)
    yPred, yProb = adaboost.predict(XTest)

    print(metrics.classification_report(y_true=yTest, y_pred=yPred))
    print("AdaBoost AUC: ", metrics.roc_auc_score(yTest, yProb))




