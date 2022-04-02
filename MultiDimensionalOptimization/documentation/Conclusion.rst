Recommendations
================

We conducted a perceptron training test. The training took place on a dataset with diabetic patients and it was necessary to predict the progress of the disease in a year.
You can read more about the dataset here: `link <https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html>`_.

After the time test, we got the following results

+---+---------+----------+------------+-------------+
|   | method  | time, s  | iteration  | last loss   |
+===+=========+==========+============+=============+
| 0 | GDOS    | 2.3986   | 100        | 0.713682    |
+---+---------+----------+------------+-------------+
| 1 | GDCS    | 6.1530   | 499        | 0.723168    |
+---+---------+----------+------------+-------------+
| 2 | CDFS    | 5.9186   | 499        | 0.723168    |
+---+---------+----------+------------+-------------+
| 3 | NCGM    | 1.1079   | 21         | 0.712832    |
+---+---------+----------+------------+-------------+

And while working, nonlinear conjugate gradient method showed the best results, the advice is to use it.
