# Mutated AI Prolog Planning System
This repository is an AI planning system, which proposes classical search algorithms as well as *mutated* configurations, i.e., planners designed to be non-optimal. All the mutants rely on a forward *A\** search that does not reopen the closed set (states that have already been visited). Different mutated versions of the *f* function (used to prioritise the open set) and the *h* function (used to approximate the distance of a state to the objective) are available; a mutant being thus any $(f_i, h_j)$ combination:
- $f_{1}(s)=h(s)$ Expands the best states first.
- $f_{2}(s)=-h(s)$ Expands the worst states first.
- $f_{3}(s)=-g(s)+h(s)$ Encourages the search to expand the deepest nodes. It uses - $h(s)$ to mitigate its naive trend to always search for a longer solution, thus leading to a timeout signal.
- $f_{4}(s)=(-1)^{g(s)}h(s)$ Scores the discovered state with - $h$ but "discards" it by signing it with the parity of its cost.
- $f_{5}(s)=-g(s) + (-1)^{g(s)}h(s)$ Combination of $f_{3}$ and $f_{4}$.
- $h_{1}(s)=|s|$ Ranks a state by its length (i.e, its number of true facts).
- $h_{2}(s)=|g \setminus (s \cap g)|$ Ranks a state with the number of facts of the goal that are not true yet.
- $h_{3}(s)= s \Delta i$ Ranks a state with its distance to the initial state. The distance used is the symmetric difference.
- $h_{4}(s)=s \Delta g$ Ranks a state with its distance to the goal state.
- $h_{5}(s)=|succ(s)|$ Ranks a state with its number of applicable actions. If we suppose that every action leads to a distinct state, then - $h_{5}$ can be defined as the number of successors of the current state.
- $h_{6}(s)=h_{max}$ Ranks a state with the index of the first fact layer of the relaxed planning graph which contains all the facts of the goal. In our planning setting, where all action costs equal 1, it corresponds to the classical heuristic $h_{max}$.

In order to run a planner, the machine should use a Windows OS and have SICStus installed (we used version 4.7.0). A setting can be run with the configured command `sicstus -l main.pl --goal "start, halt." -- $1 $2 $3 $4 $5` where:
- `$1` is the search algorithm (e.g., `bfs`; see *searches.pl* for more information).
- `$2` is the heuristic (e.g., `h_add`; see *heuristics.pl* for more information).
- `$3` is the domain filename.
- `$4` is the problem filename.
- `$5` is the output filename. If not provided, the solution is printed in the standard output.

To run a mutant $(f_i, h_j)$, replace `$1` by `mutated_astar-`$f_i$. The parameters for the mutated heuristics are (from $h_1$ to $h_6$): *h_state_length*, *h_diff*, *h_distance_with_i*, *h_distance_with_g*, *h_nb_actions*, *h_max*. For example, running the mutated *A\** $(f_2, h_3)$ on the first problem of the *block* domain can be done with `sicstus -l main.pl --goal "start, halt." -- mutated_astar-f2 h_distance_with_i benchmarks/blocks/domain.pddl benchmarks/blocks/task01.pddl output.txt`.
