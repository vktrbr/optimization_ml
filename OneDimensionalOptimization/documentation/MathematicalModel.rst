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

        2. :math:`\displaystyle x_1 = \frac{b - (b - a)}{\varphi} \quad`
           :math:`\displaystyle x_2 = \frac{a + (b - a)}{\varphi}`

        3. | if :math:`\displaystyle f(x_1) > f(x_2)` (for min)
                :math:`\displaystyle [ \, f(x_1) < f(x_2)` (for max) :math:`]`
           | then :math:`a = x_1` else  :math:`b = x_2`

        4. Repeat  :math:`2, 3` steps while :math:`|a - b| \geq e`

    2. Successive parabolic interpolation
        1. Set :math:`x_0, x_2, x_1, f = f(x), e` and calculate :math:`f_0 = f(x_0), f_1 = f(x_1), f_2 = f(x_2)`
        2. Arrange :math:`x_0, x_1, x_2` so that :math:`f_2 \leq f_1 \leq f_0`
        3. | Calculate :math:`x_{i + 1}` with the formula below
           | :math:`\displaystyle x_{i+1}=x_{i} + \frac{1}{2}\left[\frac{ \displaystyle (x_{i-1}-x_{i})^{2}(f_{i}-f_{i-2})+ (x_{i-2}-x_{i})^{2}(f_{i-1}-f_{i})}{ \displaystyle (x_{i-1}-x_{i}) (f_{i}-f_{i-2})+(x_{i-2}-x_{i})(f_{i-1}-f_{i})}\right]`
        4. Repeat step 2-3 until then :math:`|x_{i+1}-x_{i}| \geq e` or :math:`|f(x_{i+1})-f(x_{i})| \geq e`

    3. Brent's algorithm. Minimizer(func, a, b, e, t=10^{-9})
        0 Set:

            1. "Parabolic step":
            :math:`\displaystyle u = x + \frac{p}{q} = \frac{\displaystyle (x - u)^2 \cdot (f(x) - f(w)) - (x - w)^2 \cdot (f(x) - f(v))}{\displaystyle 2 \cdot ((x - v) \cdot (f(x) - f(w)) - (x - w) \cdot (f(x) - f(v)))}`

            2. "Golden step":
            if :math:`\displaystyle x < \frac{a + b}{2}`: :math:`\displaystyle u = x + \frac{\varphi - 1}{\varphi}  \cdot (b - x)`
            else: :math:`\displaystyle u = x + \frac{\varphi - 1}{\varphi}  \cdot (a - x)`

            3. Set tolerance :math:`= e * |x| + t`

        1. Set :math:`f(x), a, b, e` - function, left and right bounds, precision
        2. There are three variables :math:`x, w, v: x` is the point, where :math:`f(x)` is the least of all 3 points, :math:`w` and :math:`f(w)` has a middle value and :math:`v` and :math:`f(v)` has the largest value.
        3. Set :math:`\displaystyle x = w = v = a + \frac{\varphi - 1}{\varphi} \cdot (b - a)`
        4. Let r be the previous remainder. (The remainder is the value we add to x step by step).
        5. Check **4 conditions**

            1. :math:`|r| >` tolerance
            2. :math:`q \neq 0`
            3. :math:`\displaystyle x + \frac{p}{q} \in [\, a, b \,]`
            4. :math:`\displaystyle \frac{p}{q} < \frac{r}{2}`

        6. If 4 conditions are satisfied do **"Parabolic step"** else **"Golden step"**
        7. Rearrange :math:`u, x, w, v` to :math:`x, w, v` by rule in step 2.
        8. Repeat 4-7 until :math:`\displaystyle |x - \frac{a + b}{2}| < 2 \cdot` tolerance :math:`- \displaystyle \frac{b - a}{2}`

    4. BFGS
