:- module(main, [run_problem/5, make_input/4, solve_problem/4, solve_problem/3, start/0]).

:- use_module(library(timeout), [time_out/3]).
:- use_module(library(lists), [maplist/3]).

:- ensure_loaded(blackboard_data).
:- ensure_loaded(domain).
:- ensure_loaded(problem).
:- ensure_loaded(pddl_parser).
:- ensure_loaded(pddl_serialiser).
:- ensure_loaded(searches).
:- ensure_loaded(utils).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STARTING PREDICATE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

user:runtime_entry(start) :-
    start.

start :-
    prolog_flag(argv, [SearchAlgorithm, Heuristic, DomainFilename, ProblemFilename, ResultFilename]),
    !,
    run_problem(DomainFilename, ProblemFilename, ResultFilename, SearchAlgorithm, Heuristic).
start :-
    prolog_flag(argv, [SearchAlgorithm, Heuristic, DomainFilename, ProblemFilename]),
    solve_problem(DomainFilename, ProblemFilename, SearchAlgorithm, Heuristic).

%% run_problem(+DomainFilename, +ProblemFilename, +ResultFilename, +SearchAlgorithm, +Heuristic).
% reads and parses input files; solves the planning problem, checks the validity of the result and exports it anyway.
run_problem(DomainFilename, ProblemFilename, ResultFilename, SearchAlgorithm, Heuristic) :-
    make_input(DomainFilename, ProblemFilename, _Domain, Problem),
    !,
    set_heuristic(Heuristic),
    time_out(solve_problem(Problem, SearchAlgorithm, Plan), 120000, _),
    (validate_plan(Problem, Plan) ; true),
    maplist(untyped_action, Plan, UntypedPlan),
    serialise_plan(UntypedPlan, ResultFilename).

%% solve_problem(+DomainFilename, +ProblemFilename, +SearchAlgorithm, +Heuristic).
% reads and parses input files; solves the planning problem and checks the validity of the result.
solve_problem(DomainFilename, ProblemFilename, SearchAlgorithm, Heuristic) :-
    start_time,
    make_input(DomainFilename, ProblemFilename, _Domain, Problem),
    save_input_process_time,
    !,
    set_heuristic(Heuristic),
    time_out(solve_problem(Problem, SearchAlgorithm, Plan), 120000, _),
    print_statistic,
    (validate_plan(Problem, Plan) -> true ; write('plan not valid\n')).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HELPERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% make_input(+DomainFilename, +ProblemFilename, -Domain, -Problem).
make_input(DomainFilename, ProblemFilename, Domain, Problem) :-
    parse_domain(DomainFilename, Domain),
    parse_problem(ProblemFilename, TmpProblem),
    sort_problem(TmpProblem, Problem),
    set_blackboard(Domain, Problem),
    ground_predicates(Domain, Problem, RigidPredicatesNames, RigidFacts),
    % length(RigidFacts, LRF), format('~d ground rigid facts found.\n', [LRF]),
    ground_actions(RigidPredicatesNames, RigidFacts, Operators),
    % length(Operators, LO), format('~d ground actions found.\n', [LO]),
    set_operators(Operators).

%% solve_problem(+Problem, +SearchAlgorithm, -Solution).
solve_problem(Problem, SearchAlgorithm, Solution) :-
    problem_initial_state(Problem, InitialState),
    search(SearchAlgorithm, InitialState, Solution).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STATISTICS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

start_time :-
    statistics(runtime, [Time, _]),
    bb_put(start_time, Time).

save_input_process_time :-
    statistics(runtime, [Time, _]),
    get_start_time(StartTime),
    InputProcessTime is Time - StartTime,
    bb_put(input_process_time, InputProcessTime),
    start_time.

get_start_time(Time) :-
    bb_get(start_time, Time).

print_statistic :-
    statistics(runtime, [CurrentTime, _]),
    get_start_time(StartTime),
    SearchTime is CurrentTime - StartTime,
    bb_get(input_process_time, InputProcessTime),
    statistics(memory, [M, _]),
    Memory is M / 1048576,
    format('search time: ~3ds (input processing time: ~3ds); memory used: ~dMB.\n', [SearchTime, InputProcessTime, Memory]).