# game-tree: A quantum algorithm to solve game trees

This can be applied to familiar games like chess, go, tic-tac-toe, checkers, or any combinatorial game with perfect information. Using the
rules of the game, a superposition of all possible game histories is created and then evaluated, giving all the nodes at the base of the game tree a value. The minmax algorithm is then applied to all the nodes in the tree, ending in a value for the top of the tree, and an associated game history.

This algorithm will return the 1st forced win deterministically. It can also be configured to pick the Nth forced win. (Part of the algorithm orders all the possible games.) Remarkably, the phase of the qubits is never used: all qubits have a positive phase the entire time.

I suspect that the complexity of the algorithm is linear in the depth of the game tree, which means some versions of Go allowing games with 10^40 moves will not be accessible to this algorithm unless pruning is developed.

## System Requirements

## Dependencies
* **Recommended**: virtualenv
* python3.7
    * ```sudo apt install virtualenv```
* numpy
* itertools_more
* matplotlib
