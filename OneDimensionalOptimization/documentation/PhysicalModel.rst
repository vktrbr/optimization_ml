Problem statement
=============================================
1. Problem initializing:
    | We have function :math:`f(x)` and interval :math:`[\, a, b \, ]`.
    | It's necessary to create a function that finds the minimum of function on interval. In addition need to create an application for work with function.

2. Minimizing need to doing by 4 methods:
    1. Golden section search
    2. Successive parabolic interpolation
    3. Brent's method
    4. Broyden–Fletcher–Goldfarb–Shanno algorithm

3. Requirements:
    1. Program needs to return :math:`x_{\min}, f_{\min}, ` information about algorithm steps.
    2. Application needs to get a function, bounds, etc., to print work success message, to animate every steps of algorithm's work.


Mathematical model
=============================================
1. Requirements to :math:`f(x)`:
    1. :math:`f: \mathbb{R} \rightarrow \mathbb{R}`
    2. :math:`f` is uni-modal function on :math:`[\, a, b \, ]`
    3. :math:`f \in \mathbf{C}[\, a, b \, ]`
    4. :math:`f(x)` has :math:`\min` on the interval :math:`[\, a, b \, ]`

2. Algorithms:
    1. Golden section search
        1. Set a :math:`f(x), a, b, e` - function, left and right bounds, precision

        2. :math:`\displaystyle x_1 = \frac{b - (b - a)}{\varphi}`
           :math:`\displaystyle x_2 = \frac{a + (b - a)}{\varphi}`

        3. | if :math:`\displaystyle f(x_1) > f(x_2)` (for min)
                :math:`\displaystyle [ f(x_1) < f(x_2)` (for max) :math:`]`
           | then :math:`a = x_1` else  :math:`b = x_2`

        4. Repeat  :math:`2, 3` steps while :math:`|a - b| \geq e`

    2. Successive parabolic interpolation
        1. 