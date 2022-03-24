Mathematical model
=============================================
We have analyzed their problem. The solution is to use the gradient methods.

We choose 2 methods:
    1. Gradient Descent
    2. Nonlinear conjugate gradient method

Problem statement
---------------------------------------------
1. :math:`f(\mathbf{x}): \mathbb{R}^{n} \rightarrow \mathbb{R}`
2. :math:`\mathbf{x} \in X \subseteq \mathbb{R}^{n}`
3. :math:`\displaystyle f \longrightarrow \min_{\mathbf{x} \in X}`
4. The :math:`f` is defined and differentiable on the :math:`X`
5. Convergence to a local minimum can be guaranteed. When the function :math:`f` is convex, gradient descent can converge to the global minima.


Gradient descent equations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Set:

.. math::
    \displaystyle \mathbf{x} = \left[ \, x_1 \enspace x_2 \enspace \dots \enspace x_n \, \right]^\intercal
    :label: x-vec

2. Set:

.. math::
    \displaystyle \nabla f = \left[\frac{\partial f}{\partial x_1} \enspace \frac{\partial f}{\partial x_2} \enspace \dots \enspace \frac{\partial f}{\partial x_n}\right]^\intercal
    :label: grad

3. Gradient step:

.. math::
    \displaystyle \mathbf{x}^{i + 1} = \mathbf{x}^{i} - \gamma^{i} \cdot \nabla f(\mathbf{x}^{i})
    :label: grad-step

4. Terminate condition:

.. math::
    \displaystyle \lVert \nabla f(\mathbf{x}^{i}) \rVert_{2} < \varepsilon
    :label: terminate-cond


Nonlinear conjugate gradient method equations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Hessian:

.. math::
    \operatorname{H} =
    \begin{bmatrix} \displaystyle  \frac{\partial^2 f}{\partial x_1^2} & \displaystyle \frac{\partial^2 f}{\partial x_1 \partial x_2} &\displaystyle \dots & \displaystyle \frac{\partial^2 f}{\partial x_1 \partial x_n}  \\
                    \displaystyle  \frac{\partial^2 f}{\partial x_2 \partial x_1} & \displaystyle \frac{\partial^2 f}{\partial x_2^2} &\displaystyle \dots & \displaystyle \frac{\partial^2 f}{\partial x_2 \partial x_n}  \\
                    \displaystyle \vdots & \displaystyle \vdots & \displaystyle \ddots & \displaystyle \vdots \\
                    \displaystyle  \frac{\partial^2 f}{\partial x_n \partial x_1} & \displaystyle \frac{\partial^2 f}{\partial x_n \partial x_2} &\displaystyle \dots & \displaystyle \frac{\partial^2 f}{\partial x_n ^ 2}
    \end{bmatrix}

2. Set:

.. math::
    \displaystyle S_0, S_1, \dots, S_n \in X \subseteq \mathbb{R}^{n} \\

3. :math:`S_0, S_1, S_n` are conjugate vectors if:

.. math::
    \begin{cases}
    \displaystyle S_i^\intercal \operatorname{H} S_j = 0, \quad i \neq j, \quad i, j = 1, \dots, n \\
    \displaystyle S_i^\intercal \operatorname{H} S_i \leq 0, \quad i = 1, \dots, n
    \end{cases}

