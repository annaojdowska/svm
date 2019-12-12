import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import itertools


class Plotter:


    @staticmethod
    def plot(predictor, X, y, grid_size):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                             np.linspace(y_min, y_max, grid_size),
                             indexing='ij')
        flatten = lambda m: np.array(m).reshape(-1, )

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
                     alpha=0.4)
        plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]), c=flatten(y), cmap=cm.Paired)

        w = predictor.w
        a = -w[0] / w[1]
        xx = np.linspace(-10, 10)
        yy = a * xx - predictor.intercept_ / w[1]
        plt.plot(xx, yy, 'k-')

        if predictor.chosen_support_vectors[0] is not None:
            b = predictor.chosen_support_vectors[0]
            yy_down = a * xx + (b[1] - a * b[0])
            plt.plot(xx, yy_down, 'k--')

        if predictor.chosen_support_vectors[1] is not None:
            b = predictor.chosen_support_vectors[1]
            yy_up = a * xx + (b[1] - a * b[0])
            plt.plot(xx, yy_up, 'k--')

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()
