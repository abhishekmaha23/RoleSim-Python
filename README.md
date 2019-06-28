# RoleSim
A RoleSim implementation creating using networkX for processing

The theory and implementation details have been obtianed from the paper - https://arxiv.org/abs/1102.3937 - Axiomatic Ranking of Network Role Similarity by Jin et al. (2011).

The RoleSim algorithm is a convergin algorithm that attempts to approximately solve the NP-complete problem of role discovery problem within a reasonable number of iterations, within suitable computational complexity. In spite of operating this way, there is still a massive number of computations involved in creating an n * n matrix, and the process often times out.

The Iceberg RoleSim algorithm is a layer on top of the RoleSim algorithm that prunes out unimportant and unrelated nodepairs to identify nodes that seem to have the most interesting connections, thus reducing the order of magnitude of the processing to a manageable extent.

This implementation of RoleSim attempts to keep all the processing of the code within networkX, and thus still runs upon the CPU. On a graph with 5100 nodes, pruned by hyperparametric tuning to 665 nodes, the algorithm still takes approximately 11 minutes per iteration, and thus it is recommended that pruning be done to an extent that keeps the pruned graph within 500 nodes.

# Further

-A GPU implementation of the algorithm might help speed up computations.
-Dynamic programming and memoization are also aternatives expected to have significant benefits.
-A Numpy based processing approach was attempted, but the results were comparatively worse than through the current published implementation in NetworkX.
