## Statistics of Evaluation Data
<a name="appendix:statistics-of-evaluation-data"></a>

**Table 3** presents the statistics of the evaluation data for each benchmark.  
For each benchmark, we generate a total of 1000 problems for evaluation.  
The total variable and letter occurrence ratios are computed by counting the total occurrences of variables and letters across all problems in a benchmark and dividing by the total number of terms.  

The column **"Single equation"** reports the length, number of variables, number of letters, and the maximum occurrences of a single variable for individual equations across all problems in the benchmarks.  
The column **"Problem level"** measures the number of equations, number of variables, number of letters, and maximum occurrences of a single variable for each conjunctive word equation in the benchmarks.  
For these columns, the metrics reported are the minimum, maximum, average, and standard deviation values.

For our `DragonLi` and `Z3`, the difficulties of these benchmarks are proportional to the total variable occurrence ratio and the length of the equations, as an increase in these values leads to a decrease in the number of solved problems, as shown in **Table 2**.

The number of equations in problems for **Benchmark B** is set to 50, differing from other benchmarks. This is because its average maximum single variable occurrence is high (53.5), meaning that, on average, a single variable appears 53.5 times across all equations in a problem. Intuitively, this can be interpreted as a formula with more constraints. Consequently, when the number of equations is set to 100, nearly all problems become **UNSAT**. When training a data-driven model, the model tends to predict **UNSAT** to achieve optimal performance. To mitigate this, the number of equations is reduced to 50.

<!-- Insert Table 1 here -->
<!-- Replace the line below with a direct Markdown table or an image if applicable -->

![Evaluation Data Statistics Table](tables/eval-data-statistics.png)
