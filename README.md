# Distributed Delay-Tolerant Bregman Descent

This repo implements a distributed and asynchronous optimisation algorithm. This algorithm is an adaptation of the non-distributed Bregman descent algorithm [^1].

The convergence properties of this algorithm are described in our paper [^2].

To start using the code `cd` to the project directory and call

```shell
(v1.0) pkg> activate .

(DistributedBregman) pkg> instantiate
```

To reploduce the plots seen in the paper, 

```shell
$ julia main_fig.jl 
```

[^1]: Bolte, Jérôme & Bauschke, Heinz & Teboulle, Marc. (2016). A Descent Lemma Beyond Lipschitz Gradient Continuity: First-Order Methods Revisited and Applications. Mathematics of Operations Research. 42. 10.1287/moor.2016.0817. 

[^2]: S. Chraibi, F. Iutzelera, J. Malick and A. Rogozin. (2022). Delay-tolerant Distributed Bregman Proximal Algorithms.
