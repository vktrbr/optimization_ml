import numpy as np
import torch
from typing import Literal, Callable
from metrics import make_metrics_tab, best_threshold


class LogisticRegressionRBF(torch.nn.Module):

    def __init__(self, x_basis: torch.Tensor, rbf: Literal['linear', 'gaussian', 'multiquadratic'] = 'gaussian',
                 print_function: Callable = print):
        """
        :param x_basis: centers of basis functions
        :param rbf: type of rbf function. Available: ['linear', 'gaussian']
        :param print_function: e.g. print or streamlit.write
        """
        super(LogisticRegressionRBF, self).__init__()

        self.w = torch.nn.Linear(x_basis.shape[0], 1)
        self.rbf = rbf
        self.x_basis = x_basis
        self.print = print_function
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor = None, phi_matrix: torch.Tensor = None) -> torch.Tensor:
        """
        Returns a "probability" (confidence) of class 1

        :param x: 2D array
        :param phi_matrix: 2D array
        :return: 1D array
        """
        if phi_matrix is None:
            phi_matrix = self.make_phi_matrix(x)

        return self.sigmoid(self.w(phi_matrix))

    def make_phi_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns k x n array with calculated phi(x_i, x_basis_j)

        :param x: Array k x m dimensional. k different x_i and m features
        """
        n = self.x_basis.shape[0]
        k = x.shape[0]

        repeated_input_x = torch.tile(x, (n, 1))
        repeated_basis_x = torch.tile(self.x_basis, (1, k))
        repeated_basis_x = torch.reshape(repeated_basis_x, repeated_input_x.shape)

        phi = ((repeated_input_x - repeated_basis_x) ** 2).sum(dim=1)
        phi = torch.reshape(phi, (n, k)).T

        if self.rbf == 'linear':
            phi = phi ** 0.5
            phi = phi / phi.max()
        elif self.rbf == 'gaussian':
            phi = torch.exp(-phi)
        elif self.rbf == 'multiquadratic':
            phi = (1 + phi) ** 0.5
            phi = phi / phi.max()

        return phi

    def fit(self, x, y, epochs=1):

        print_epochs = np.unique(np.geomspace(1, epochs + 1, 15, dtype=int))

        phi_matrix = self.make_phi_matrix(x)
        optimizer = torch.optim.Adam(self.parameters())
        loss = torch.nn.BCELoss()

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            output = loss(self.forward(x, phi_matrix).flatten(), y)
            output.backward()
            optimizer.step()

            with torch.no_grad():
                if epoch + 1 in print_epochs:
                    self.print(f'Epoch: {epoch: 5d} | CrossEntropyLoss: {output.item(): 0.5f}')

        return self

    def metrics_tab(self, x, y, metric: Literal['f1', 'by_roc'] = 'f1'):
        threshold = best_threshold(x, y, self, metric=metric)
        return make_metrics_tab(self, x, y, threshold)


class LogisticRegression(torch.nn.Module):

    def __init__(self, n_features: int, kernel: Literal['linear', 'perceptron'] = 'linear',
                 print_function: Callable = print):
        """

        :param n_features: amount of features (columns)
        :param kernel: 'linear' or 'perceptron'. linear - basic logistic regression, perceptron - nn with 1
        hidden layer with dim = 512
        :param print_function: print or streamlit.write
        """
        super(LogisticRegression, self).__init__()

        self.print = print_function
        self.sigmoid = torch.nn.Sigmoid()
        if kernel == 'linear':
            self.weights = torch.nn.Linear(n_features, 1)
        elif kernel == 'perceptron':
            self.weights = torch.nn.Sequential(
                torch.nn.Linear(n_features, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 1)
            )
        else:
            raise TypeError('Invalid kernel. Choose "linear" or "perceptron"')

    def forward(self, x):
        """ Just some function Rn -> R, for example linear. After that, the sigmoid function is applied """
        return self.sigmoid(self.weights(x))

    def fit(self, x, y, epochs=1):

        print_epochs = np.unique(np.geomspace(1, epochs + 1, 15, dtype=int))

        optimizer = torch.optim.Adam(self.parameters())
        loss = torch.nn.BCELoss()

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            output = loss(self.forward(x).flatten(), y)
            output.backward()
            optimizer.step()

            with torch.no_grad():
                if epoch + 1 in print_epochs:
                    self.print(f'Epoch: {epoch: 5d} | CrossEntropyLoss: {output.item(): 0.5f}')

        return self

    def metrics_tab(self, x, y, metric: Literal['f1', 'by_roc'] = 'f1'):
        threshold = best_threshold(x, y, self, metric=metric)
        return make_metrics_tab(self, x, y, threshold)
