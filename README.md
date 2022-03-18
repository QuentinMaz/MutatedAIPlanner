# Mutated AI Prolog Planning System
This repository is an AI planning system, which proposes classical search algorithms as well as *mutated* configurations, i.e., hard-coded mutations of well known algorithms. Three mutants are implemented, all meant to be used with the forward search space:
- **a_star_mutant1** Mutated implementation of a forward A* search. It does not reopen closed nodes (states that have already been visited) and the *f* function (used to prioritise the open set) is given by *f(state)=-g(state)-h(state)*.
- **a_star_mutant2** Similar to the first A* based mutant, but uses a mutated prioritisation function *f(state)=-g(state)+h(state)*.
- **a_star_mutant3** Weighted A* version of **a_star_mutant2**. Precisely, the weight parameter is set to 10, leading to *f(state)=-g(state)+10\*h(state)*.


## Running a planner
In order to run a planner, the machine should use a Windows OS and have SICStus installed (we used version 4.7.0). A configuration can be run with the command `sicstus -l main.pl --goal "start, halt." -a $1 $2 $3 $4 $5 $6` where:
- `$1` is the search space (either `forward` or `backward`).
- `$2` is the search algorithm (e.g., `bfs`; see *main.pl* for more information).
- `$3` is the heuristic (e.g., `h_add`; see *heuristics.pl* for more information).
- `$4` is the domain filename.
- `$5` is the problem filename.
- `$6` is the output filename.

As a side note, the backward-based searches deliver poor performance and are thus not recommended. For that matter, the behavior of the unsupported configurations (like the mutated searches with the backward) is unknown.

## Building the planning system
SICStus provides a simple and easy way to build an all-in-one executable from the source code. Open a terminal in the folder containing this repository and do the following:
- Run the `sicstus` command and successively execute `compile(main).`,  `save_program('main.sav').` and `halt.`. The traces should look like:
```
sicstus
SICStus [...]
| ?- compile(main).
% [...]
yes
| ?- save_program('main.sav').
% [...]
yes
| ?- halt.
```
- Build the executable with `spld --output=main.exe --static main.sav`:
```
spld --output=main.exe --static main.sav
[...]
spldgen_s_14540_1647527992_restore_main.c
spldgen_s_14540_1647527992_main_wrapper.c
spldgen_s_14540_1647527992_prolog_rtable.c
   Creating library main.lib and object main.exp
Created "main.exe"
```
At this point, an executable file *main.exe* should have been created. When built, it uses the same inputs described above.
