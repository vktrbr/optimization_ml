Conceptual model
=============================================
Imagine situation:
---------------------------------------------
Factory's management has a function that describes the income from the sale of products.
The function depends on many variables, e.g. the salary by each employee, the cost of each product and so on.

The management gave us the function :math:`g(\mathbf{x})`, the running costs, the gradient of the functions in analytical form (Their mathematicians did a good job).
But it will better if our solution automatically calculates gradient.

And our goal is to ensure the **best cost distribution**.

\* For unifying let's introduce :math:`f(\mathbf{x}) = -g(\mathbf{x})`.
