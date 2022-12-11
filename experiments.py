import csv
import os

from sklearn.feature_selection import SelectKBest, f_classif, SequentialFeatureSelector, chi2, mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np


def inputation(X, strategy):
    # inputancja brakujących danych
    if strategy == "knn":
        imp = KNNImputer(n_neighbors=2, weights="uniform")
    else:
        imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imp.fit(X)
    X = imp.transform(X)

    return X


def experiment(experiment_number, strategy, metric, selector=None, chi2_sel=False):
    global clfs, scores

    # wczytanie zestawu danych
    dataset = 'ctg_3'
    dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    # zdefiniowanie klasyfikatorów
    clfs = {
        'gnb': GaussianNB(),
        'knn': KNeighborsClassifier(),
        'svm': SVC(gamma=1),
    }

    file = open(f"{experiment_number}.txt", 'a')

    file.write(f"\nInput strategy: {strategy}\n")

    # walidacja krzyżowa
    n_splits = 5
    n_repeats = 2
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    scores = np.zeros((len(clfs), n_splits * n_repeats))
    if selector is not None:
        file.write(f"Selector: {metric}")
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        X[train] = inputation(X[train], strategy)
        X[test] = inputation(X[test], strategy)
        if chi2_sel:
            X[train] = normalize(X[train])
            X[test] = normalize(X[test])
        if selector is not None:
            X_train = selector.fit_transform(X[train], y[train])
            X_test = selector.fit_transform(X[test], y[test])
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            if selector is not None:
                clf.fit(X_train, y[train])
                y_pred = clf.predict(X_test)
            else:
                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])

            scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)
    mean = np.mean(scores, axis=1)
    std = np.std(scores, axis=1)
    with open(f"exp{experiment_number}.txt", 'a') as expFile:
        csvWriter = csv.writer(expFile)
        for clf_id, clf_name in enumerate(clfs):
            print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))
            csvWriter.writerow((metric, clf_name, mean[clf_id], std[clf_id]))
    print("\n")

    # wczytanie wyników
    print("Folds:\n", np.array2string(scores, separator=", "))
    print("\n")
    # test parowy
    # t-statistic i p-value

    alfa = .05
    t_statistic = np.zeros((len(clfs), len(clfs)))
    p_value = np.zeros((len(clfs), len(clfs)))
    for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])
    print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)
    # print("\n")
    # print("p-value:\n", p_value)
    print("\n")

    # wypisanie z uzyciem tabulate
    headers = ["GNB", "KNN", "SVM"]
    names_column = np.array([["GNB"], ["KNN"], ["SVM"]])
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

    # print()
    print("\n")

    # file.write()
    file.write("\n")


def normalize(X):
    X[:, 24] = (X[:, 24] + 1) / 2
    return X


def experiment1():
    if os.path.exists("exp1.txt"):
        os.remove("exp1.txt")
    if os.path.exists("1.txt"):
        os.remove("1.txt")
    input_strategy = [
        'mean',
        'knn',
        'most_frequent'
    ]
    for strategy in input_strategy:
        print(f"\nInput strategy: {strategy}\n")
        experiment('1', strategy, strategy)


def experiment2():
    if os.path.exists("2.txt"):
        os.remove("2.txt")
    if os.path.exists("exp2.txt"):
        os.remove("exp2.txt")
    selected_features = [
        7,
        14,
        21,
        28
    ]
    for number in selected_features:
        selectors = {
            'chi2': SelectKBest(chi2, k=number),
            'ANOVA': SelectKBest(f_classif, k=number),
            'Mutual': SelectKBest(mutual_info_classif, k=number),
        }
        for selector_name, selector in selectors.items():
            print(f"\n###################################\n\nSelector: {selector_name} - selected {number} features\n")
            if selector_name == "chi2":
                experiment('2', 'mean', f"{selector_name}-{number}", selector=selector, chi2_sel=True)
            else:
                experiment('2', 'mean', f"{selector_name}-{number}", selector=selector)


def experiment3():
    if os.path.exists("exp3.txt"):
        os.remove("exp3.txt")
    if os.path.exists("3.txt"):
        os.remove("3.txt")

    selected_features = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        10,
        15,
        20,
        25,
        30
    ]
    for number in selected_features:
        selectors = {
            'SFS - forward': SequentialFeatureSelector(GaussianNB(), n_features_to_select=number, direction='forward'),
            'SFS - backward': SequentialFeatureSelector(GaussianNB(), n_features_to_select=number,
                                                        direction='backward'),
        }
        for selector_name, selector in selectors.items():
            print(f"\n###################################\n\nSelector: {selector_name} - selected {number} features\n")
            experiment('3', 'mean', f"{selector_name}-{number}", selector=selector)


if __name__ == "__main__":
    experiment2()
