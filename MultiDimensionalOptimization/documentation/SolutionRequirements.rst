Requirements
=============================================

Required input fields
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Type of algorithm. (Gradient with constant step, fractional step, optimal step, nonlinear conjugate gradient method and Newton-CG)
#. Field with function input :math:`f(\mathrm{x})`
#. Field with the analytical gradient flag. If analytical gradient flag is True then request the gradient in analytical form. :math:`\nabla f(\mathrm{x})`
#. Field with the start point (The default value is random numbers from a uniform distribution on (-1, 1))
#. Search precision :math:`\varepsilon`
#. Field with a history saving flag
#. Required parameters for each method

Visualization Required
~~~~~~~~~~~~~~~~~~~~~~~~~~
#. If the function depends on 1 variable, then it is necessary to draw a curve and animate the movement of the point and draw the previous steps.
#. If the function depends on 2 variables, then you need to draw contour lines and animate the movement of the point and draw the previous steps.
#. For functions of larger dimensions, output a graph of the decreasing gradient norm and the function values for each iteration.

#. For 1 and 2 dimensional output a graph of the decreasing gradient norm and the function values for each iteration too.

Requirements for methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Each method must keep a history of all iterations. Be sure to save every :math:`\mathrm{x}_k, f_k, \| \nabla f_k\|`
#. The solution must contain the final point, the value of the function, the number of iterations, and the history.
#. The solution should cause a minimum number of calculations of the function values
