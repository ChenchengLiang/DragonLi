
## Workflow
<a name="appendix:workflow"></a>

The workflow of our framework is illustrated in **Figure 1**. For a benchmark, we begin by randomly splitting the dataset into training and evaluation subsets.

### Training Phase

During the training phase, we first input the word equation problems into the split algorithm ranking option **RE1** to obtain their satisfiabilities and separate them into **SAT**, **UNSAT**, and **UNKNOWN** sets. For the **SAT** and **UNSAT** sets, we discard them since our algorithm already knows how to solve them.

For the **UNKNOWN** set, the problems are passed to other solvers, such as `z3` and `cvc5`. If the solver concludes **UNSAT**, we systematically identify **Minimal Unsatisfiable Subsets (MUSes)** of conjunctive word equations by exhaustively checking the satisfiability of subsets, starting with individual equations and stopping upon finding the first MUS.

We use the MUSes to rank and sort conjunctive word equations unsolvable by the split algorithm, then reprocess the sorted equations with it. This allows the split algorithm to solve some problems and construct proof trees.

Then, we can extract the labeled data from both the AND-OR tree and MUSes. The labeling process is described in Section 4.

Next, we convert the labeled conjunctive word equations from textual to graph format, enabling the model with GNN layers to process them. The model takes a **rank point** (i.e., a conjunction of word equations) as input and outputs corresponding scores that indicate the priority of the conjuncts.

### Prediction Phase

In the prediction phase, during the step where inference rules are applied, the trained model ranks and sorts the conjuncts. The equation with the highest score is then selected to apply the branching inference rules.

---

### Figure 1: Workflow Diagram

![Workflow Diagram](https://github.com/ChenchengLiang/DragonLi/blob/rank/Appendix/figures/workflow-diagram.pdf)

> *The workflow diagram for the training and prediction phase.*
