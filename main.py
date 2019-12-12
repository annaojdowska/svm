import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import itertools
from svm.kernel import Kernel
#from iterative.classificator import SVM
from svm.binary_classes import SVM
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
#from svm.binary_classes import binary_classification_smo


def main():
    # example()
    example()


# def generateDataManual():
#     X, y = make_blobs(n_samples=250, centers=2, random_state=0, cluster_std=0.60)
#     y[y == 0] = -1
#     tmp = np.ones(len(X))
#     y = tmp * y
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')
#     plt.show()
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#     classifier = binary_classification_smo(Kernel.linear())
#     classifier.fit(X_train, y_train)
#     sup_vec = classifier.support_vectors_
#     plt.plot(sup_vec)
#     plt.show()

def generateDataCVXOPT():
    X, y = make_blobs(n_samples=250, centers=2, random_state=0, cluster_std=0.60)
    y[y == 0] = -1
    tmp = np.ones(len(X))
    y = tmp * y

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    svm = SVM()
    svm.fit(X_train, y_train)

    def f(x, w, b, c=0):
        return (-w[0] * x - b + c) / w[1]

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter')
    # w.x + b = 0
    a0 = -4
    a1 = f(a0, svm.w, svm.b)
    b0 = 4
    b1 = f(b0, svm.w, svm.b)
    plt.plot([a0, b0], [a1, b1], 'k')
    # w.x + b = 1
    a0 = -4
    a1 = f(a0, svm.w, svm.b, 1)
    b0 = 4
    b1 = f(b0, svm.w, svm.b, 1)
    plt.plot([a0, b0], [a1, b1], 'k--')
    # w.x + b = -1
    a0 = -4
    a1 = f(a0, svm.w, svm.b, -1)
    b0 = 4
    b1 = f(b0, svm.w, svm.b, -1)
    plt.plot([a0, b0], [a1, b1], 'k--')
    plt.show()


def example(num_samples=100, num_features=2, grid_size=200):
    samples = np.array(np.random.normal(size=num_samples * num_features).reshape(num_samples, num_features))
    labels = 2 * (samples.sum(axis=1) > 0) - 1.0

    # np.random.seed(0)
    # samples = np.r_[np.random.randn(50, 2) - [2, 2], np.random.randn(50, 2) + [2, 2]]
    # labels = [0] * 50 + [1] * 50

    # samples, labels = make_blobs(n_samples=num_samples, centers=num_features, random_state=0, cluster_std=0.60)
    # labels[labels == 0] = -1
    # tmp = np.ones(len(samples))
    # labels = tmp * labels

    clf = SVM(Kernel.linear(), 0.1)
    clf.fit(samples, labels)
    print(clf.score(samples, labels))
    plot(clf, samples, labels, grid_size)


def plot(predictor, X, y, grid_size):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    flatten = lambda m: np.array(m).reshape(-1,)

    result = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        result.append(predictor.predict(point))

    Z = np.array(result).reshape(xx.shape)
    plt.clf()

    plt.contourf(xx, yy, Z,
                 cmap=cm.Paired,
                 levels=[-0.001, 0.001],
                 extend='both',
                 alpha=0.8)
    plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]),
                c=flatten(y), cmap=cm.Paired)

    def f(x, w, b, c=0):
        return (-w[0] * x - b + c) / w[1]

    # # w.x + b = 0
    # a0 = -4
    # a1 = f(a0, predictor.w, predictor.intercept_)
    # b0 = 4
    # b1 = f(b0, predictor.w, predictor.intercept_)
    # plt.plot([a0, b0], [a1, b1], 'k')
    # # w.x + b = 1
    # a0 = -10
    # a1 = f(a0, predictor.w, predictor.intercept_, 1)
    # b0 = 10
    # b1 = f(b0, predictor.w, predictor.intercept_, 1)
    # plt.plot([a0, b0], [a1, b1], 'k--')
    # # w.x + b = -1
    # a0 = -10
    # a1 = f(a0, predictor.w, predictor.intercept_, -1)
    # b0 = 10
    # b1 = f(b0, predictor.w, predictor.intercept_, -1)
    # plt.plot([a0, b0], [a1, b1], 'k--')
    w = predictor.w
    a = -w[0] / w[1]
    xx = np.linspace(-10, 10)
    yy = a * xx - predictor.intercept_ / w[1]
    plt.plot(xx, yy, 'k-')
    b = predictor.chosen_support_vectors[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = predictor.chosen_support_vectors[1]
    yy_up = a * xx + (b[1] - a * b[0])

    # plot the line, the points, and the nearest vectors to the plane
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

main()
