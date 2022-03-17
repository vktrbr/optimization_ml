Mathematical model
=============================================
1. Requirements to :math:`f(x)`:
    1. :math:`f: \mathbb{R} arrow \mathbb{R}`
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
        1. Set :math:`x_0, x_2, x_1` and calculate :math:`f_0 = f(x_0), f_1 = f(x_1), f_2 = f(x_2)`
        2. Arrange :math:`x_0, x_1, x_2` so that :math:`f_2 \leq f_1 \leq f_0`
        3. | Calculate :math:`x_{i + 1}` with the formula below
           | :math:`\displaystyle x_{i+1}=x_{i} + \frac{1}{2}[\frac{ \displaystyle (x_{i-1}-x_{i})^{2}(f_{i}-f_{i-2})+ (x_{i-2}-x_{i})^{2}(f_{i-1}-f_{i})}{ \displaystyle (x_{i-1}-x_{i}) (f_{i}-f_{i-2})+(x_{i-2}-x_{i})(f_{i-1}-f_{i})}]`
        4. Repeat step 2-3 until then :math:`|x_{i+1}-x_{i}| \geq e` or :math:`|f(x_{i+1})-f(x_{i})| \geq e`
