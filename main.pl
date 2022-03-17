:- use_module(library(timeout), [time_out/3]).

:- [blackboard_data, pddl_serialiser, search_algorithms, utils].

test :-
    prolog_flag(argv, [D, P]),
    write('BFS                  :\n'), solve_fwd_bfs(D, P),
    write('A*_PLUS              :\n'), solve_fwd_a_star_h_plus(D, P),
    write('DFS_FIRST_SOLUTION   :\n'), solve_fwd_dfs_first_solution(D, P),
    write('DFS                  :\n'), solve_fwd_dfs(D, P),
    write('IDDFS                :\n'), solve_fwd_iddfs(D, P),
    write('DFS_LONGER_SOLUTION  :\n'), solve_fwd_dfs_longer_solution(D, P),
    write('A*_0                 :\n'), solve_fwd_a_star(D, P),
    write('A*_DIFF              :\n'), solve_fwd_a_star_h_diff(D, P),
    write('A*_ADD               :\n'), solve_fwd_a_star_h_add(D, P),
    write('A*_MUTANT1             :\n'), solve_fwd_a_star_mutant1(D, P),
    write('A*_MUTANT2             :\n'), solve_fwd_a_star_mutant2(D, P),
    write('A*_MUTANT3             :\n'), solve_fwd_a_star_mutant3(D, P),
    halt.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SOLVING PREDICATES (runs a configuration and writes statistics)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

solve_fwd_bfs(DomainFilename, ProblemFilename) :-
    solve_problem(DomainFilename, ProblemFilename, forward, bfs).

solve_fwd_dfs(DomainFilename, ProblemFilename) :-
    solve_problem(DomainFilename, ProblemFilename, forward, dfs).

solve_fwd_iddfs(DomainFilename, ProblemFilename) :-
    solve_problem(DomainFilename, ProblemFilename, forward, iddfs).

solve_fwd_dfs_first_solution(DomainFilename, ProblemFilename) :-
    solve_problem(DomainFilename, ProblemFilename, forward, dfs_first_solution).

solve_fwd_dfs_longer_solution(DomainFilename, ProblemFilename) :-
    solve_problem(DomainFilename, ProblemFilename, forward, dfs_longer_solution).

solve_fwd_a_star(DomainFilename, ProblemFilename) :-
    set_heuristic(h_0),
    solve_problem(DomainFilename, ProblemFilename, forward, a_star).

solve_fwd_a_star_h_plus(DomainFilename, ProblemFilename) :-
    set_heuristic(h_plus),
    solve_problem(DomainFilename, ProblemFilename, forward, a_star).

solve_fwd_a_star_h_diff(DomainFilename, ProblemFilename) :-
    set_heuristic(h_diff),
    solve_problem(DomainFilename, ProblemFilename, forward, a_star).

solve_fwd_a_star_h_add(DomainFilename, ProblemFilename) :-
    set_heuristic(h_add),
    solve_problem(DomainFilename, ProblemFilename, forward, a_star).

solve_fwd_a_star_mutant1(DomainFilename, ProblemFilename) :-
    set_heuristic(h_add),
    solve_problem(DomainFilename, ProblemFilename, forward, a_star_mutant1).

solve_fwd_a_star_mutant2(DomainFilename, ProblemFilename) :-
    set_heuristic(h_diff),
    solve_problem(DomainFilename, ProblemFilename, forward, a_star_mutant2).

solve_fwd_a_star_mutant3(DomainFilename, ProblemFilename) :-
    set_heuristic(h_diff),
    solve_problem(DomainFilename, ProblemFilename, forward, a_star_mutant3).

solve_bwd_bfs(DomainFilename, ProblemFilename) :-
    solve_problem(DomainFilename, ProblemFilename, backward, bfs).

solve_bwd_dfs(DomainFilename, ProblemFilename) :-
    solve_problem(DomainFilename, ProblemFilename, backward, dfs).

%% solve_problem(+DomainFilename, +ProblemFilename, +StateSpaceSearch, +SearchAlgorithm).
% reads files and sets timelimit for planner
solve_problem(DomainFilename, ProblemFilename, StateSpaceSearch, SearchAlgorithm) :-
    make_input(DomainFilename, ProblemFilename, Domain-Problem),
    start_time,
    !,
    time_out(solve(Domain, Problem, StateSpaceSearch, SearchAlgorithm, Plan), 60000, _),
    print_statistic(Problem, Plan),
    writeq(Plan), nl,
    (check_plan_validity(Problem, Plan) -> true ; write('plan not valid\n')).

%% solve(+Domain, +Problem, +StateSpaceSearch, +SearchAlgorithm, -Solution).
solve(Domain, Problem, StateSpaceSearch, SearchAlgorithm, Solution) :-
    set_blackboard(Domain, Problem, StateSpaceSearch, SearchAlgorithm),
    initialise_start_state(StateSpaceSearch, StartState),
    search(SearchAlgorithm, StartState, Solution).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STATISTICS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

start_time :-
    statistics(runtime, [Time, _]),
    bb_put(start_time, Time).

get_start_time(Time) :-
    bb_get(start_time, Time).

%% print_statistic(+Problem, +Plan).
print_statistic(Problem, Plan) :-
    problem_name(Problem, Name),
    get_start_time(StartTime),
    statistics(runtime, [CurrentTime, _]),
    statistics(memory, [Memory, _]),
    Time is CurrentTime - StartTime,
    length(Plan, PlanLength),
    format('~a ~3d ~d ~d\n', [Name, Time, Memory, PlanLength]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RUNNING PREDICATE (runs a configuration and writes the solution)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

run_problem(DomainFilename, ProblemFilename, ResultFilename, StateSpaceSearch, SearchAlgorithm, Heuristic) :-
    make_input(DomainFilename, ProblemFilename, Domain-Problem),
    !,
    set_heuristic(Heuristic),
    time_out(solve(Domain, Problem, StateSpaceSearch, SearchAlgorithm, Plan), 30000, _),
    (check_plan_validity(Problem, Plan) ; true),
    serialise_plan(Plan, ResultFilename).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STARTING PREDICATE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

user:runtime_entry(start) :-
    start.

start :-
    prolog_flag(argv, [StateSpaceSearch, SearchAlgorithm, Heuristic, DomainFilename, ProblemFilename, ResultFilename]),
    %% TODO : check the inputs... (# security concerns)
    run_problem(DomainFilename, ProblemFilename, ResultFilename, StateSpaceSearch, SearchAlgorithm, Heuristic).