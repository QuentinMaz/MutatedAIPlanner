:- use_module(library(plunit)).
:- use_module(library(timeout), [time_out/3]).

:- [readFile, parseDomain, parseProblem].
:- [state_space_searches, search_algorithms, utils, blackboard_data].

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TESTING HELPERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

test_fwd_dfs_first_solution(DomainFile, ProblemFile, Plan) :-
    test_problem(DomainFile, ProblemFile, forward, dfs_first_solution, Plan).

test_fwd_dfs_longer_solution(DomainFile, ProblemFile, Plan) :-
    test_problem(DomainFile, ProblemFile, forward, dfs_longer_solution, Plan).

test_fwd_a_star_h_add(DomainFile, ProblemFile, Plan) :-
    set_heuristic(h_add),
    test_problem(DomainFile, ProblemFile, forward, a_star, Plan).

test_fwd_a_star_h_diff(DomainFile, ProblemFile, Plan) :-
    set_heuristic(h_diff),
    test_problem(DomainFile, ProblemFile, forward, a_star, Plan).

test_problem(DomainFile, ProblemFile, StateSpaceSearch, SearchAlgorithm, Plan) :-
    make_input(DomainFile, ProblemFile, Domain-Problem),
    time_out(solve(Domain, Problem, StateSpaceSearch, SearchAlgorithm, Plan), 30000, _),
    !.

%% solve(+Domain, +Problem, +StateSpaceSearch, +SearchAlgorithm, -Solution).
solve(D, P, StateSpaceSearch, SearchAlgorithm, Solution) :-
    set_blackboard(D, P, StateSpaceSearch, SearchAlgorithm),
    initialise_start_state(StateSpaceSearch, StartState),
    search(SearchAlgorithm, StartState, Solution).

heuristic(h_diff).
heuristic(h_add).
heuristic(h_plus).
heuristic(h_length).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLUNIT TESTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DFS_FIRST_SOLUTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(forward_dfs_first_solution).

test(blocks-04-0, [true(L > 6)]) :-
    test_fwd_dfs_first_solution('test/blocks/domain.pddl', 'test/blocks/blocks2.pddl', P),
    length(P, L).

test(blocks-06-2, [blocked('too long to run'), true(L > 20)]) :-
    test_fwd_dfs_first_solution('test/blocks/domain.pddl', 'test/blocks/blocks5.pddl', P),
    length(P, L).

test(hanoi-easy, [true(L > 1)]) :-
    test_fwd_dfs_first_solution('test/hanoi/domain.pddl', 'test/hanoi/hanoi1.pddl', P),
    length(P, L).

test(hanoi3, [true(L > 7)]) :-
    test_fwd_dfs_first_solution('test/hanoi/domain.pddl', 'test/hanoi/hanoi2.pddl', P),
    length(P, L).

test(gripper2, [true(L > 5)]) :-
    test_fwd_dfs_first_solution('test/gripper/domain.pddl', 'test/gripper/gripper1.pddl', P),
    length(P, L).

test(gripper4, [true(L > 11)]) :-
    test_fwd_dfs_first_solution('test/gripper/domain.pddl', 'test/gripper/gripper2.pddl', P),
    length(P, L).

test(typed_gripper2, [true(L > 5)]) :-
    test_fwd_dfs_first_solution('test/typed_gripper/domain.pddl', 'test/typed_gripper/typed_gripper1.pddl', P),
    length(P, L).

test(typed_gripper4, [true(L > 11)]) :-
    test_fwd_dfs_first_solution('test/typed_gripper/domain.pddl', 'test/typed_gripper/typed_gripper2.pddl', P),
    length(P, L).

test(monkey1, [true(L > 6)]) :-
    test_fwd_dfs_first_solution('test/monkey/domain.pddl', 'test/monkey/monkey1.pddl', P),
    length(P, L).

test(monkey2, [blocked('too long to run'), true(L > 12)]) :-
    test_fwd_dfs_first_solution('test/monkey/domain.pddl', 'test/monkey/monkey2.pddl', P),
    length(P, L).

test(simple_rover, [true(L > 8)]) :-
    test_fwd_dfs_first_solution('test/simple_rover/domain.pddl', 'test/simple_rover/simple_rover1.pddl', P),
    length(P, L).

:- end_tests(forward_dfs_first_solution).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DFS_LONGER_SOLUTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(forward_dfs_longer_solution).

test(blocks-04-0, [true(L > 6)]) :-
    test_fwd_dfs_longer_solution('test/blocks/domain.pddl', 'test/blocks/blocks2.pddl', P),
    length(P, L).

test(blocks-06-2, [blocked('too long to run'), true(L > 20)]) :-
    test_fwd_dfs_longer_solution('test/blocks/domain.pddl', 'test/blocks/blocks5.pddl', P),
    length(P, L).

test(hanoi-easy, [true(L > 1)]) :-
    test_fwd_dfs_longer_solution('test/hanoi/domain.pddl', 'test/hanoi/hanoi1.pddl', P),
    length(P, L).

test(hanoi3, [true(L > 7)]) :-
    test_fwd_dfs_longer_solution('test/hanoi/domain.pddl', 'test/hanoi/hanoi2.pddl', P),
    length(P, L).

test(gripper2, [true(L > 5)]) :-
    test_fwd_dfs_longer_solution('test/gripper/domain.pddl', 'test/gripper/gripper1.pddl', P),
    length(P, L).

test(gripper4, [true(L > 11)]) :-
    test_fwd_dfs_longer_solution('test/gripper/domain.pddl', 'test/gripper/gripper2.pddl', P),
    length(P, L).

test(typed_gripper2, [true(L > 5)]) :-
    test_fwd_dfs_longer_solution('test/typed_gripper/domain.pddl', 'test/typed_gripper/typed_gripper1.pddl', P),
    length(P, L).

test(typed_gripper4, [true(L > 11)]) :-
    test_fwd_dfs_longer_solution('test/typed_gripper/domain.pddl', 'test/typed_gripper/typed_gripper2.pddl', P),
    length(P, L).

test(monkey1, [true(L > 6)]) :-
    test_fwd_dfs_longer_solution('test/monkey/domain.pddl', 'test/monkey/monkey1.pddl', P),
    length(P, L).

test(monkey2, [blocked('too long to run'), true(L > 12)]) :-
    test_fwd_dfs_longer_solution('test/monkey/domain.pddl', 'test/monkey/monkey2.pddl', P),
    length(P, L).

test(simple_rover, [true(L > 8)]) :-
    test_fwd_dfs_longer_solution('test/simple_rover/domain.pddl', 'test/simple_rover/simple_rover1.pddl', P),
    length(P, L).

:- end_tests(forward_dfs_longer_solution).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A_STAR_H_ADD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(forward_a_star_h_add).

test(blocks-04-0, [true(L > 6)]) :-
    test_fwd_a_star_h_add('test/blocks/domain.pddl', 'test/blocks/blocks2.pddl', P),
    length(P, L).

test(blocks-06-2, [true(L > 20)]) :-
    test_fwd_a_star_h_add('test/blocks/domain.pddl', 'test/blocks/blocks5.pddl', P),
    length(P, L).

test(hanoi-easy, [true(L > 1)]) :-
    test_fwd_a_star_h_add('test/hanoi/domain.pddl', 'test/hanoi/hanoi1.pddl', P),
    length(P, L).

test(hanoi3, [true(L > 7)]) :-
    test_fwd_a_star_h_add('test/hanoi/domain.pddl', 'test/hanoi/hanoi2.pddl', P),
    length(P, L).

test(hanoi6, [true(L > 63)]) :-
    test_fwd_a_star_h_add('test/hanoi/domain.pddl', 'test/hanoi/hanoi3.pddl', P),
    length(P, L).

test(gripper2, [true(L > 5)]) :-
    test_fwd_a_star_h_add('test/gripper/domain.pddl', 'test/gripper/gripper1.pddl', P),
    length(P, L).

test(gripper4, [true(L > 11)]) :-
    test_fwd_a_star_h_add('test/gripper/domain.pddl', 'test/gripper/gripper2.pddl', P),
    length(P, L).

test(typed_gripper2, [true(L > 5)]) :-
    test_fwd_a_star_h_add('test/typed_gripper/domain.pddl', 'test/typed_gripper/typed_gripper1.pddl', P),
    length(P, L).

test(typed_gripper4, [true(L > 11)]) :-
    test_fwd_a_star_h_add('test/typed_gripper/domain.pddl', 'test/typed_gripper/typed_gripper2.pddl', P),
    length(P, L).

test(monkey1, [true(L > 6)]) :-
    test_fwd_a_star_h_add('test/monkey/domain.pddl', 'test/monkey/monkey1.pddl', P),
    length(P, L).

test(monkey2, [true(L > 12)]) :-
    test_fwd_a_star_h_add('test/monkey/domain.pddl', 'test/monkey/monkey2.pddl', P),
    length(P, L).

test(simple_rover, [true(L > 8)]) :-
    test_fwd_a_star_h_add('test/simple_rover/domain.pddl', 'test/simple_rover/simple_rover1.pddl', P),
    length(P, L).

test(complex_rover, [true(L > 20)]) :-
    test_fwd_a_star_h_add('test/complex_rover/domain.pddl', 'test/complex_rover/complex_rover1.pddl', P),
    length(P, L).

:- end_tests(forward_a_star_h_add).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A_STAR_H_DIFF
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(forward_a_star_h_diff).

test(blocks-04-0, [true(L > 6)]) :-
    test_fwd_a_star_h_diff('test/blocks/domain.pddl', 'test/blocks/blocks2.pddl', P),
    length(P, L).

test(blocks-06-2, [true(L > 20)]) :-
    test_fwd_a_star_h_diff('test/blocks/domain.pddl', 'test/blocks/blocks5.pddl', P),
    length(P, L).

test(hanoi-easy, [true(L > 1)]) :-
    test_fwd_a_star_h_diff('test/hanoi/domain.pddl', 'test/hanoi/hanoi1.pddl', P),
    length(P, L).

test(hanoi3, [true(L > 7)]) :-
    test_fwd_a_star_h_diff('test/hanoi/domain.pddl', 'test/hanoi/hanoi2.pddl', P),
    length(P, L).

test(hanoi6, [true(L > 63)]) :-
    test_fwd_a_star_h_diff('test/hanoi/domain.pddl', 'test/hanoi/hanoi3.pddl', P),
    length(P, L).

test(gripper2, [true(L > 5)]) :-
    test_fwd_a_star_h_diff('test/gripper/domain.pddl', 'test/gripper/gripper1.pddl', P),
    length(P, L).

test(gripper4, [true(L > 11)]) :-
    test_fwd_a_star_h_diff('test/gripper/domain.pddl', 'test/gripper/gripper2.pddl', P),
    length(P, L).

test(typed_gripper2, [true(L > 5)]) :-
    test_fwd_a_star_h_diff('test/typed_gripper/domain.pddl', 'test/typed_gripper/typed_gripper1.pddl', P),
    length(P, L).

test(typed_gripper4, [true(L > 11)]) :-
    test_fwd_a_star_h_diff('test/typed_gripper/domain.pddl', 'test/typed_gripper/typed_gripper2.pddl', P),
    length(P, L).

test(monkey1, [true(L > 6)]) :-
    test_fwd_a_star_h_diff('test/monkey/domain.pddl', 'test/monkey/monkey1.pddl', P),
    length(P, L).

test(monkey2, [true(L > 12)]) :-
    test_fwd_a_star_h_diff('test/monkey/domain.pddl', 'test/monkey/monkey2.pddl', P),
    length(P, L).

test(simple_rover, [true(L > 8)]) :-
    test_fwd_a_star_h_diff('test/simple_rover/domain.pddl', 'test/simple_rover/simple_rover1.pddl', P),
    length(P, L).

test(complex_rover, [true(L > 20)]) :-
    test_fwd_a_star_h_diff('test/complex_rover/domain.pddl', 'test/complex_rover/complex_rover1.pddl', P),
    length(P, L).

:- end_tests(forward_a_star_h_diff).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A_STAR_MUTANT1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(forward_a_star_mutant1).

test(blocks-04-0, [forall(heuristic(Heuristic)), true(L > 6)]) :-
    set_heuristic(Heuristic),
    test_problem('test/blocks/domain.pddl', 'test/blocks/blocks2.pddl', forward, a_star_mutant1, Plan),
    length(Plan, L).

test(blocks-06-2, [forall(heuristic(Heuristic)), true(L > 20)]) :-
    set_heuristic(Heuristic),
    test_problem('test/blocks/domain.pddl', 'test/blocks/blocks5.pddl', forward, a_star_mutant1, Plan),
    length(Plan, L).

test(hanoi-easy, [forall(heuristic(Heuristic)), true(L > 1)]) :-
    set_heuristic(Heuristic),
    test_problem('test/hanoi/domain.pddl', 'test/hanoi/hanoi1.pddl', forward, a_star_mutant1, Plan),
    length(Plan, L).

test(hanoi3, [forall(heuristic(Heuristic)), true(L > 7)]) :-
    set_heuristic(Heuristic),
    test_problem('test/hanoi/domain.pddl', 'test/hanoi/hanoi2.pddl', forward, a_star_mutant1, Plan),
    length(Plan, L).

test(hanoi6, [forall(heuristic(Heuristic)), true(L > 63)]) :-
    set_heuristic(Heuristic),
    test_problem('test/hanoi/domain.pddl', 'test/hanoi/hanoi3.pddl', forward, a_star_mutant1, Plan),
    length(Plan, L).

test(gripper2, [forall(heuristic(Heuristic)), true(L > 5)]) :-
    set_heuristic(Heuristic),
    test_problem('test/gripper/domain.pddl', 'test/gripper/gripper1.pddl', forward, a_star_mutant1, Plan),
    length(Plan, L).

test(gripper4, [forall(heuristic(Heuristic)), true(L > 11)]) :-
    set_heuristic(Heuristic),
    test_problem('test/gripper/domain.pddl', 'test/gripper/gripper2.pddl', forward, a_star_mutant1, Plan),
    length(Plan, L).

test(typed_gripper2, [forall(heuristic(Heuristic)), true(L > 5)]) :-
    set_heuristic(Heuristic),
    test_problem('test/typed_gripper/domain.pddl', 'test/typed_gripper/typed_gripper1.pddl', forward, a_star_mutant1, Plan),
    length(Plan, L).

test(typed_gripper4, [forall(heuristic(Heuristic)), true(L > 11)]) :-
    set_heuristic(Heuristic),
    test_problem('test/typed_gripper/domain.pddl', 'test/typed_gripper/typed_gripper2.pddl', forward, a_star_mutant1, Plan),
    length(Plan, L).

test(monkey1, [forall(heuristic(Heuristic)), true(L > 6)]) :-
    set_heuristic(Heuristic),
    test_problem('test/monkey/domain.pddl', 'test/monkey/monkey1.pddl', forward, a_star_mutant1, Plan),
    length(Plan, L).

test(monkey2, [forall(heuristic(Heuristic)), true(L > 12)]) :-
    set_heuristic(Heuristic),
    test_problem('test/monkey/domain.pddl', 'test/monkey/monkey2.pddl', forward, a_star_mutant1, Plan),
    length(Plan, L).

test(simple_rover, [forall(heuristic(Heuristic)), true(L > 8)]) :-
    set_heuristic(Heuristic),
    test_problem('test/simple_rover/domain.pddl', 'test/simple_rover/simple_rover1.pddl', forward, a_star_mutant1, Plan),
    length(Plan, L).

test(complex_rover, [forall(heuristic(Heuristic)), true(L > 20)]) :-
    set_heuristic(Heuristic),
    test_problem('test/complex_rover/domain.pddl', 'test/complex_rover/complex_rover1.pddl', forward, a_star_mutant1, Plan),
    length(Plan, L).

test(logistics-5-2, [forall(heuristic(Heuristic)), true(L > 8)]) :-
    set_heuristic(Heuristic),
    test_problem('test/logistics/domain.pddl', 'test/logistics/logistics1.pddl', forward, a_star_mutant1, Plan),
    length(Plan, L).

:- end_tests(forward_a_star_mutant1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A_STAR_MUTANT2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(forward_a_star_mutant2).

test(blocks-04-0, [forall(heuristic(Heuristic)), true(L > 6)]) :-
    set_heuristic(Heuristic),
    test_problem('test/blocks/domain.pddl', 'test/blocks/blocks2.pddl', forward, a_star_mutant2, Plan),
    length(Plan, L).

test(blocks-06-2, [forall(heuristic(Heuristic)), true(L > 20)]) :-
    set_heuristic(Heuristic),
    test_problem('test/blocks/domain.pddl', 'test/blocks/blocks5.pddl', forward, a_star_mutant2, Plan),
    length(Plan, L).

test(hanoi-easy, [forall(heuristic(Heuristic)), true(L > 1)]) :-
    set_heuristic(Heuristic),
    test_problem('test/hanoi/domain.pddl', 'test/hanoi/hanoi1.pddl', forward, a_star_mutant2, Plan),
    length(Plan, L).

test(hanoi3, [forall(heuristic(Heuristic)), true(L > 7)]) :-
    set_heuristic(Heuristic),
    test_problem('test/hanoi/domain.pddl', 'test/hanoi/hanoi2.pddl', forward, a_star_mutant2, Plan),
    length(Plan, L).

test(hanoi6, [forall(heuristic(Heuristic)), true(L > 63)]) :-
    set_heuristic(Heuristic),
    test_problem('test/hanoi/domain.pddl', 'test/hanoi/hanoi3.pddl', forward, a_star_mutant2, Plan),
    length(Plan, L).

test(gripper2, [forall(heuristic(Heuristic)), true(L > 5)]) :-
    set_heuristic(Heuristic),
    test_problem('test/gripper/domain.pddl', 'test/gripper/gripper1.pddl', forward, a_star_mutant2, Plan),
    length(Plan, L).

test(gripper4, [forall(heuristic(Heuristic)), true(L > 11)]) :-
    set_heuristic(Heuristic),
    test_problem('test/gripper/domain.pddl', 'test/gripper/gripper2.pddl', forward, a_star_mutant2, Plan),
    length(Plan, L).

test(typed_gripper2, [forall(heuristic(Heuristic)), true(L > 5)]) :-
    set_heuristic(Heuristic),
    test_problem('test/typed_gripper/domain.pddl', 'test/typed_gripper/typed_gripper1.pddl', forward, a_star_mutant2, Plan),
    length(Plan, L).

test(typed_gripper4, [forall(heuristic(Heuristic)), true(L > 11)]) :-
    set_heuristic(Heuristic),
    test_problem('test/typed_gripper/domain.pddl', 'test/typed_gripper/typed_gripper2.pddl', forward, a_star_mutant2, Plan),
    length(Plan, L).

test(monkey1, [forall(heuristic(Heuristic)), true(L > 6)]) :-
    set_heuristic(Heuristic),
    test_problem('test/monkey/domain.pddl', 'test/monkey/monkey1.pddl', forward, a_star_mutant2, Plan),
    length(Plan, L).

test(monkey2, [forall(heuristic(Heuristic)), true(L > 12)]) :-
    set_heuristic(Heuristic),
    test_problem('test/monkey/domain.pddl', 'test/monkey/monkey2.pddl', forward, a_star_mutant2, Plan),
    length(Plan, L).

test(simple_rover, [forall(heuristic(Heuristic)), true(L > 8)]) :-
    set_heuristic(Heuristic),
    test_problem('test/simple_rover/domain.pddl', 'test/simple_rover/simple_rover1.pddl', forward, a_star_mutant2, Plan),
    length(Plan, L).

test(complex_rover, [forall(heuristic(Heuristic)), true(L > 20)]) :-
    set_heuristic(Heuristic),
    test_problem('test/complex_rover/domain.pddl', 'test/complex_rover/complex_rover1.pddl', forward, a_star_mutant2, Plan),
    length(Plan, L).

test(logistics-5-2, [forall(heuristic(Heuristic)), true(L > 8)]) :-
    set_heuristic(Heuristic),
    test_problem('test/logistics/domain.pddl', 'test/logistics/logistics1.pddl', forward, a_star_mutant2, Plan),
    length(Plan, L).

:- end_tests(forward_a_star_mutant2).