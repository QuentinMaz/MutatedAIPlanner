:- use_module(library(plunit)).
:- use_module(library(timeout), [time_out/3]).

:- ensure_loaded(main).
:- ensure_loaded(blackboard_data).
:- ensure_loaded(utils).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INPUTS PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

complex_problem('test/blocks/domain.pddl', 'test/blocks/blocks4.pddl').
complex_problem('test/monkey/domain.pddl', 'test/monkey/monkey2.pddl').
complex_problem('test/complex_rover/domain.pddl', 'test/complex_rover/complex_rover1.pddl').
complex_problem('test/hanoi/domain.pddl', 'test/hanoi/hanoi3.pddl').
complex_problem('test/logistics/domain.pddl', 'test/logistics/logistics1.pddl').

light_problem('test/blocks/domain.pddl', 'test/blocks/blocks2.pddl').
light_problem('test/blocks/domain.pddl', 'test/blocks/blocks3.pddl').

light_problem('test/hanoi/domain.pddl', 'test/hanoi/hanoi1.pddl').
light_problem('test/hanoi/domain.pddl', 'test/hanoi/hanoi2.pddl').

light_problem('test/gripper/domain.pddl', 'test/gripper/gripper1.pddl').
light_problem('test/gripper/domain.pddl', 'test/gripper/gripper2.pddl').

light_problem('test/typed_gripper/domain.pddl', 'test/typed_gripper/typed_gripper1.pddl').
light_problem('test/typed_gripper/domain.pddl', 'test/typed_gripper/typed_gripper2.pddl').

light_problem('test/monkey/domain.pddl', 'test/monkey/monkey1.pddl').
light_problem('test/monkey/domain.pddl', 'test/monkey/monkey2.pddl').

light_problem('test/simple_rover/domain.pddl', 'test/simple_rover/simple_rover1.pddl').

light_problem('test/airport/domain1.pddl', 'test/airport/airport1.pddl').
light_problem('test/airport/domain2.pddl', 'test/airport/airport2.pddl').

heuristic(h_distance_with_i).
heuristic(h_distance_with_g).
heuristic(h_diff).
heuristic(h_state_length).
heuristic(h_max).
heuristic(h_nb_actions).

mutated_search(f1).
mutated_search(f2).
mutated_search(f3).
mutated_search(f4).
mutated_search(f5).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TESTING HELPERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

validate_blind_search(DomainFile, ProblemFile, BlindSearchAlgorithm) :-
    make_input(DomainFile, ProblemFile, _Domain, Problem),
    !,
    time_out(solve_problem(Problem, BlindSearchAlgorithm, Plan), 30000, _),
    validate_plan(Problem, Plan).

validate_informed_search(DomainFile, ProblemFile, SearchAlgorithm, Heuristic) :-
    make_input(DomainFile, ProblemFile, _Domain, Problem),
    !,
    set_heuristic(Heuristic),
    time_out(solve_problem(Problem, SearchAlgorithm, Plan), 30000, _),
    validate_plan(Problem, Plan).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BFS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(bfs).

test(bfs_light_problems, [nondet, forall(light_problem(DomainFile, ProblemFile))]) :-
    validate_blind_search(DomainFile, ProblemFile, bfs).

test(bfs_complex_problems, [nondet, forall(complex_problem(DomainFile, ProblemFile))]) :-
    validate_blind_search(DomainFile, ProblemFile, bfs).

:- end_tests(bfs).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A_STAR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(astar).

test(astar_light_problems, [nondet, forall(light_problem(DomainFile, ProblemFile))]) :-
    validate_informed_search(DomainFile, ProblemFile, astar, h_zero).

test(astar_complex_problems, [nondet, forall(complex_problem(DomainFile, ProblemFile))]) :-
    validate_informed_search(DomainFile, ProblemFile, astar, h_zero).

:- end_tests(astar).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MUTANTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(mutants).

test(mutant_light_problems, [nondet, forall((light_problem(DomainFile, ProblemFile), mutated_search(MutatedSearch), heuristic(Heuristic)))]) :-
    atom_concat('mutated_astar-', MutatedSearch, Search),
    validate_informed_search(DomainFile, ProblemFile, Search, Heuristic).

test(mutant_complex_problems, [nondet, forall((complex_problem(DomainFile, ProblemFile), mutated_search(MutatedSearch), heuristic(Heuristic)))]) :-
    atom_concat('mutated_astar-', MutatedSearch, Search),
    validate_informed_search(DomainFile, ProblemFile, Search, Heuristic).

:- end_tests(mutants).
