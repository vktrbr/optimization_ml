Mathematical model
=============================================
We have analyzed their problem. The solution is to use the gradient methods.

We have chosen 2 methods:
    1. Gradient Descent
    2. Nonlinear conjugate gradient method

Problem statement
---------------------------------------------
1. :math:`f(\mathbf{x}): \mathbb{R}^{n} \rightarrow \mathbb{R}`
2. :math:`\mathbf{x} \in X \subseteq \mathbb{R}^{n}`
3. :math:`\displaystyle f \longrightarrow \min_{\mathbf{x} \in X}`
4. The :math:`f` is defined and differentiable on the :math:`X`
5. Convergence to a local minimum can be guaranteed. When the function :math:`f` is convex, gradient descent can converge to the global minima.


Gradient descent
---------------------------------------------

Equations
~~~~~~~~~~
1. Function argument:

.. math::
    \displaystyle x = \left[ \, x_1 \enspace x_2 \enspace \dots \enspace x_n \, \right]^\top
    :label: x-vec

2. Gradient:

.. math::
    \displaystyle \nabla f = \left[\frac{\partial f}{\partial x_1} \enspace \frac{\partial f}{\partial x_2} \enspace \dots \enspace \frac{\partial f}{\partial x_n}\right]^\top
    :label: grad

3. Gradient step:

.. math::
    \displaystyle \mathbf{x}_{i + 1} = \mathbf{x}_{i} - \gamma_{i} \cdot \nabla f(\mathbf{x}_{i})
    :label: grad-step

4. Terminate condition:

.. math::
    \displaystyle \lVert \nabla f(\mathbf{x}_{i}) \rVert_{2} < \varepsilon
    :label: terminate-cond


Algorithm with constant step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The gradient of the function shows us the direction of increasing the function. The idea is to move in the opposite
direction to :math:`\mathbf{x}_{k+1}` where :math:`f(\mathbf{x}_{k+1}) < f(\mathbf{x}_k)`.

But, if we add a gradient to :math:`\mathbf{x}_k` without changes, our method will often diverge. So we need to add a gradient
with some weight :math:`\gamma`.

.. include:: AlgorithmFlowcharts\FlowchartGradConstStep.rst

Algorithm with descent step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Requirements: :math:`0 < \lambda < 1` is the step multiplier,  :math:`0 < \delta < 1`.

.. include:: AlgorithmFlowcharts\FlowchartGradFracStep.rst

Algorithm with optimal step size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Another good idea is to find a :math:`\gamma`  that minimizes :math:`\phi(\gamma) = f(\mathrm{x}_k + \gamma \cdot \nabla f(\mathrm{x}_k))`

So we have a task to find the :math:`\displaystyle \gamma_{\min} = \operatorname*{arg\,min}_{\gamma \in (0, 1)}\phi(\gamma)`.
We will use Brent's algorithm to search :math:`\displaystyle \gamma_{\min}`.

.. include:: AlgorithmFlowcharts\FlowchartOptimalStep.rst

Strong Wolfe conditions
---------------------------------------------
The conditions necessary to minimize :math:`\displaystyle \phi(\gamma) = f(\mathrm{x}_k + \gamma p_k)` and find :math:`\gamma_k = \displaystyle \mathrm{arg}\min_{\gamma}\phi`

.. math::
    f(\mathrm{x}_k + \gamma_k p_k) \leq f(\mathrm{x}_k) + c_1 \gamma_k \nabla f_k^{\top} p_k
    :label: conditionWolfe1

.. math::
    |\nabla f(\mathrm{x}_k + \gamma_k p_k)^{\top}p_k| \leq -c_2 \nabla f_k^{\top}p_k
    :label: conditionWolfe2


Nonlinear conjugate gradient method
---------------------------------------------
The Fletcher–Reeves method.
:math:`p_k` is the direction to evaluate :math:`\mathrm{x}_{k+1}`.

    1. :math:`p_0 = -\nabla f_0`
    2. :math:`p_{k+1} = \nabla f_{k+1} + \beta^{FR}_{k+1} p_k`

In the RF method, :math:`\gamma` is searched using Line Search (Nocedal, Wright (2006) *Numerical Optimization* pp.60-61)

Our modification is that if Line Search does not converge, use Brent's algorithm to search for :math:`\displaystyle \gamma_{\min} = \mathrm{arg}\min_{\gamma}\phi`

.. math::
    \beta^{FR}_{k+1} = \frac{\nabla f_{k+1} ^ {\top} \nabla f_{k+1}}{\nabla f_{k}^{\top} \nabla f_{k}}
    :label: beta-fletcher-reeves

Algorithm Nonlinear conjugate gradient method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. include:: AlgorithmFlowcharts\FlowchartRFAlg.rst
