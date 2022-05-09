from __future__ import annotations


class ModelCl:
    """
    Base model for classification
    """

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs) -> float:
        """ Method to call model """
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> float:
        pass

    def fit(self, *args, **kwargs) -> ModelCl:
        pass


class LogisticRegressionTorch(ModelCl):

    def __init__(self, w_size: int, l1_reg: bool = True, ):
        super(LogisticRegressionTorch, self).__init__()
