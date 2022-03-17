:- use_module(library(plunit)).
:- use_module(library(timeout), [time_out/3]).

:- [readFile, parseDomain, parseProblem].
:- [state_space_searches, search_algorithms, utils, blackboard_data].

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TESTING HELPERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% forward testing
test_fwd_bfs(DomainFile, ProblemFile, Plan) :-
    test_problem(DomainFile, ProblemFile, forward, bfs, Plan).

test_fwd_dfs(DomainFile, ProblemFile, Plan) :-
    test_problem(DomainFile, ProblemFile, forward, dfs, Plan).

test_fwd_iddfs(DomainFile, ProblemFile, Plan) :-
    test_problem(DomainFile, ProblemFile, forward, iddfs, Plan).

test_fwd_a_star(DomainFile, ProblemFile, Plan) :-
    set_heuristic(h_0),
    test_problem(DomainFile, ProblemFile, forward, a_star, Plan).

test_fwd_a_star_h_plus(DomainFile, ProblemFile, Plan) :-
    set_heuristic(h_plus),
    test_problem(DomainFile, ProblemFile, forward, a_star, Plan).

test_problem(DomainFile, ProblemFile, StateSpaceSearch, SearchAlgorithm, Plan) :-
    make_input(DomainFile, ProblemFile, Domain-Problem),
    time_out(solve(Domain, Problem, StateSpaceSearch, SearchAlgorithm, Plan), 60000, _), 
    !.

%% solve(+Domain, +Problem, +StateSpaceSearch, +SearchAlgorithm, -Solution).
solve(D, P, StateSpaceSearch, SearchAlgorithm, Solution) :-
    set_blackboard(D, P, StateSpaceSearch, SearchAlgorithm),
    initialise_start_state(StateSpaceSearch, StartState),
    search(SearchAlgorithm, StartState, Solution).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLUNIT TESTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BFS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% FORWARD BFS
:- begin_tests(forward_bfs).

test(blocks-04-0, [true(L =:= 6)]) :-
    test_fwd_bfs('test/blocks/domain.pddl', 'test/blocks/blocks2.pddl', P),
    length(P, L).

test(blocks-06-2, [blocked('too long to run'), true(L =:= 20)]) :-
    test_fwd_bfs('test/blocks/domain.pddl', 'test/blocks/blocks5.pddl', P),
    length(P, L).

test(hanoi-easy, [true(L =:= 1)]) :-
    test_fwd_bfs('test/hanoi/domain.pddl', 'test/hanoi/hanoi1.pddl', P),
    length(P, L).

test(hanoi3, [true(L =:= 7)]) :-
    test_fwd_bfs('test/hanoi/domain.pddl', 'test/hanoi/hanoi2.pddl', P),
    length(P, L).

test(hanoi6, [true(L =:= 63)]) :-
    test_fwd_bfs('test/hanoi/domain.pddl', 'test/hanoi/hanoi3.pddl', P),
    length(P, L).

test(gripper2, [true(L =:= 5)]) :-
    test_fwd_bfs('test/gripper/domain.pddl', 'test/gripper/gripper1.pddl', P),
    length(P, L).

test(gripper4, [true(L =:= 11)]) :-
    test_fwd_bfs('test/gripper/domain.pddl', 'test/gripper/gripper2.pddl', P),
    length(P, L).

test(monkey1, [true(L =:= 6)]) :-
    test_fwd_bfs('test/monkey/domain.pddl', 'test/monkey/monkey1.pddl', P),
    length(P, L).

test(monkey2, [true(L =:= 12)]) :-
    test_fwd_bfs('test/monkey/domain.pddl', 'test/monkey/monkey2.pddl', P),
    length(P, L).

test(simple_rover, [true(L =:= 8)]) :-
    test_fwd_bfs('test/simple_rover/domain.pddl', 'test/simple_rover/simple_rover1.pddl', P),
    length(P, L).

test(complex_rover, [true(L =:= 20)]) :-
    test_fwd_bfs('test/complex_rover/domain.pddl', 'test/complex_rover/complex_rover1.pddl', P),
    length(P, L).

:- end_tests(forward_bfs).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DFS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% FORWARD DFS
:- begin_tests(forward_dfs).

test(blocks-04-0, [true(L =:= 6)]) :-
    test_fwd_dfs('test/blocks/domain.pddl', 'test/blocks/blocks2.pddl', P),
    length(P, L).

test(blocks-06-2, [blocked('too long to run'), true(L =:= 20)]) :-
    test_fwd_dfs('test/blocks/domain.pddl', 'test/blocks/blocks5.pddl', P),
    length(P, L).

test(hanoi-easy, [true(L =:= 1)]) :-
    test_fwd_dfs('test/hanoi/domain.pddl', 'test/hanoi/hanoi1.pddl', P),
    length(P, L).

test(hanoi3, [true(L =:= 7)]) :-
    test_fwd_dfs('test/hanoi/domain.pddl', 'test/hanoi/hanoi2.pddl', P),
    length(P, L).

test(gripper2, [true(L =:= 5)]) :-
    test_fwd_dfs('test/gripper/domain.pddl', 'test/gripper/gripper1.pddl', P),
    length(P, L).

test(gripper4, [true(L =:= 11)]) :-
    test_fwd_dfs('test/gripper/domain.pddl', 'test/gripper/gripper2.pddl', P),
    length(P, L).

test(monkey1, [true(L =:= 6)]) :-
    test_fwd_dfs('test/monkey/domain.pddl', 'test/monkey/monkey1.pddl', P),
    length(P, L).

test(monkey2, [blocked('too long to run'), true(L =:= 12)]) :-
    test_fwd_dfs('test/monkey/domain.pddl', 'test/monkey/monkey2.pddl', P),
    length(P, L).

test(simple_rover, [true(L =:= 8)]) :-
    test_fwd_dfs('test/simple_rover/domain.pddl', 'test/simple_rover/simple_rover1.pddl', P),
    length(P, L).

:- end_tests(forward_dfs).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% IDDFS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% FORWARD IDDFS
:- begin_tests(forward_iddfs).

test(blocks-04-0, [true(L =:= 6)]) :-
    test_fwd_iddfs('test/blocks/domain.pddl', 'test/blocks/blocks2.pddl', P),
    length(P, L).

test(blocks-06-2, [blocked('too long to run'), true(L =:= 20)]) :-
    test_fwd_iddfs('test/blocks/domain.pddl', 'test/blocks/blocks5.pddl', P),
    length(P, L).

test(hanoi-easy, [true(L =:= 1)]) :-
    test_fwd_iddfs('test/hanoi/domain.pddl', 'test/hanoi/hanoi1.pddl', P),
    length(P, L).

test(hanoi3, [true(L =:= 7)]) :-
    test_fwd_iddfs('test/hanoi/domain.pddl', 'test/hanoi/hanoi2.pddl', P),
    length(P, L).

test(gripper2, [true(L =:= 5)]) :-
    test_fwd_iddfs('test/gripper/domain.pddl', 'test/gripper/gripper1.pddl', P),
    length(P, L).

test(gripper4, [true(L =:= 11)]) :-
    test_fwd_iddfs('test/gripper/domain.pddl', 'test/gripper/gripper2.pddl', P),
    length(P, L).

test(monkey1, [true(L =:= 6)]) :-
    test_fwd_iddfs('test/monkey/domain.pddl', 'test/monkey/monkey1.pddl', P),
    length(P, L).

test(monkey2, [blocked('too long to run'), true(L =:= 12)]) :-
    test_fwd_iddfs('test/monkey/domain.pddl', 'test/monkey/monkey2.pddl', P),
    length(P, L).

test(simple_rover, [true(L =:= 8)]) :-
    test_fwd_iddfs('test/simple_rover/domain.pddl', 'test/simple_rover/simple_rover1.pddl', P),
    length(P, L).

:- end_tests(forward_iddfs).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A_STAR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% FORWARD A_STAR
:- begin_tests(forward_a_star).

test(blocks-04-0, [true(L =:= 6)]) :-
    test_fwd_a_star('test/blocks/domain.pddl', 'test/blocks/blocks2.pddl', P),
    length(P, L).

test(blocks-06-2, [blocked('too long to run'), true(L =:= 20)]) :-
    test_fwd_a_star('test/blocks/domain.pddl', 'test/blocks/blocks5.pddl', P),
    length(P, L).

test(hanoi-easy, [true(L =:= 1)]) :-
    test_fwd_a_star('test/hanoi/domain.pddl', 'test/hanoi/hanoi1.pddl', P),
    length(P, L).

test(hanoi3, [true(L =:= 7)]) :-
    test_fwd_a_star('test/hanoi/domain.pddl', 'test/hanoi/hanoi2.pddl', P),
    length(P, L).

test(hanoi6, [true(L =:= 63)]) :-
    test_fwd_a_star('test/hanoi/domain.pddl', 'test/hanoi/hanoi3.pddl', P),
    length(P, L).

test(gripper2, [true(L =:= 5)]) :-
    test_fwd_a_star('test/gripper/domain.pddl', 'test/gripper/gripper1.pddl', P),
    length(P, L).

test(gripper4, [true(L =:= 11)]) :-
    test_fwd_a_star('test/gripper/domain.pddl', 'test/gripper/gripper2.pddl', P),
    length(P, L).

test(monkey1, [true(L =:= 6)]) :-
    test_fwd_a_star('test/monkey/domain.pddl', 'test/monkey/monkey1.pddl', P),
    length(P, L).

test(monkey2, [true(L =:= 12)]) :-
    test_fwd_a_star('test/monkey/domain.pddl', 'test/monkey/monkey2.pddl', P),
    length(P, L).

test(simple_rover, [true(L =:= 8)]) :-
    test_fwd_a_star('test/simple_rover/domain.pddl', 'test/simple_rover/simple_rover1.pddl', P),
    length(P, L).

test(complex_rover, [true(L =:= 20)]) :-
    test_fwd_a_star('test/complex_rover/domain.pddl', 'test/complex_rover/complex_rover1.pddl', P),
    length(P, L).

:- end_tests(forward_a_star).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A_STAR_H_PLUS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% FORWARD A_STAR_H_PLUS
:- begin_tests(forward_a_star_h_plus).

test(blocks-04-0, [true(L =:= 6)]) :-
    test_fwd_a_star_h_plus('test/blocks/domain.pddl', 'test/blocks/blocks2.pddl', P),
    length(P, L).

test(blocks-06-2, [blocked('too long to run'), true(L =:= 20)]) :-
    test_fwd_a_star_h_plus('test/blocks/domain.pddl', 'test/blocks/blocks5.pddl', P),
    length(P, L).

test(hanoi-easy, [true(L =:= 1)]) :-
    test_fwd_a_star_h_plus('test/hanoi/domain.pddl', 'test/hanoi/hanoi1.pddl', P),
    length(P, L).

test(hanoi3, [true(L =:= 7)]) :-
    test_fwd_a_star_h_plus('test/hanoi/domain.pddl', 'test/hanoi/hanoi2.pddl', P),
    length(P, L).

test(hanoi6, [true(L =:= 63)]) :-
    test_fwd_a_star_h_plus('test/hanoi/domain.pddl', 'test/hanoi/hanoi3.pddl', P),
    length(P, L).

test(gripper2, [true(L =:= 5)]) :-
    test_fwd_a_star_h_plus('test/gripper/domain.pddl', 'test/gripper/gripper1.pddl', P),
    length(P, L).

test(gripper4, [true(L =:= 11)]) :-
    test_fwd_a_star_h_plus('test/gripper/domain.pddl', 'test/gripper/gripper2.pddl', P),
    length(P, L).

test(monkey1, [true(L =:= 6)]) :-
    test_fwd_a_star_h_plus('test/monkey/domain.pddl', 'test/monkey/monkey1.pddl', P),
    length(P, L).

test(monkey2, [true(L =:= 12)]) :-
    test_fwd_a_star_h_plus('test/monkey/domain.pddl', 'test/monkey/monkey2.pddl', P),
    length(P, L).

test(simple_rover, [true(L =:= 8)]) :-
    test_fwd_a_star_h_plus('test/simple_rover/domain.pddl', 'test/simple_rover/simple_rover1.pddl', P),
    length(P, L).

test(complex_rover, [blocked('too long to run'), true(L =:= 20)]) :-
    test_fwd_a_star_h_plus('test/complex_rover/domain.pddl', 'test/complex_rover/complex_rover1.pddl', P),
    length(P, L).

:- end_tests(forward_a_star_h_plus).