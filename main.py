import numpy as np

from svm.binary_classes import SVM
from svm.kernel import Kernel
from svm.plotter import Plotter


def main():
    example()


def example(num_samples=100, num_features=2, grid_size=200):
    # ToDo: zbiory danych powinny byc wyklikane z UI (+ opcja wybierz jakis losowy rozklad? i wtedy mozna wylosowac jednym z tych sposobow ponizej)
    # generowanie danych nr 1
    samples = np.array(np.random.normal(size=num_samples * num_features).reshape(num_samples, num_features))
    labels = 2 * (samples.sum(axis=1) > 0) - 1.0

    # generowanie danych nr 2
    # zmieniajac cluster_std zmieniasz rozklad kropek w zbiorach, im mniejsze cluster_std, np. 0.5 to zbiory sa od siebie bardziej odseparowane

    # samples, labels = make_blobs(n_samples=num_samples, centers=num_features, random_state=0, cluster_std=1)
    # labels[labels == 0] = -1
    # tmp = np.ones(len(samples))
    # labels = tmp * labels

    clf = SVM(Kernel.linear(), 0.1)
    clf.fit(samples, labels)
    print("final score {}".format(clf.score(samples, labels)))
    Plotter.plot(clf, samples, labels, grid_size)


main()
