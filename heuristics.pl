:- module(heuristics,
    [
        compute_heuristic/3,

        h_zero/2, h_state_length/2,
        h_distance_with_i/2, h_distance_with_g/2,
        h_max/2, h_diff/2, h_nb_actions/2,

        write_heuristics/0
    ]).

:- use_module(library(ordsets), [ord_intersection/3, ord_subtract/3, ord_subset/2, ord_union/2]).

:- ensure_loaded(blackboard_data).
:- ensure_loaded(problem).
:- ensure_loaded(domain).
:- ensure_loaded(utils).

write_heuristics :-
    Heuristics =
        [
            h_zero, h_state_length, h_diff,
            h_distance_with_i, h_distance_with_g,
            h_max, h_nb_actions
        ],
    format('\nheuristics available :\n~@\n', [write_list(Heuristics)]).

write_list([]).
write_list([H|T]) :-
    write(H),
    nl,
    write_list(T).

compute_heuristic(HeuristicName, State, HeuristicResult) :-
    HeuristicPredicate =.. [HeuristicName, State, HeuristicResult],
    HeuristicPredicate.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HEURISTICS PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% h_zero(+State, -Heuristic).
h_zero(_State, 0).

%% h_diff(+State, -Heuristic).
h_diff(State, Heuristic) :-
    get_problem(Problem),
    problem_goal_state(Problem, GoalState),
    ord_subtract(GoalState, State, Intersection),
    length(Intersection, Heuristic).

%% h_state_length(+State, -Heuristic).
h_state_length(State, StateLength) :-
    length(State, StateLength).

%% h_distance_with_i(+State, -Heuristic).
% The heuristic value is the sum of distinct literals of the initial and the current state.
h_distance_with_i(State, DistinctElements) :-
    get_problem(Problem),
    problem_initial_state(Problem, InitialState),
    compute_distance_between_states(InitialState, State, DistinctElements).

%% h_distance_with_g(+State, -Heuristic).
% The heuristic value is the sum of distinct literals of the initial and the current state.
h_distance_with_g(State, DistinctElements) :-
    get_problem(Problem),
    problem_goal_state(Problem, GoalState),
    compute_distance_between_states(GoalState, State, DistinctElements).

% unifies 0 if the current state S contains the goal G
h_max(State, 0) :-
    get_problem(Problem),
    problem_goal_state(Problem, GoalState),
    ord_subset(GoalState, State),
    !.
h_max(State, NewDepth) :-
    setof(PE, relaxed_step(State, PE), SetOfPE),
    ord_union([State|SetOfPE], TmpState),
    sort(TmpState, NewState),
    NewState \= State,
    !,
    h_max(NewState, Depth),
    NewDepth is Depth + 1.
% limits the search in case of unreachable goal state
h_max(_, 0). % (only seen in problems based on the airport domain...)

% this heuristic ranks a state by its number of successors
h_nb_actions(State, Heuristic) :-
    bagof(Operator, progress(State, Operator, _), Operators),
    !,
    length(Operators, Heuristic).
h_nb_actions(_, 0).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HELPERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% compute_distance_between_states(+State1, +State2, -DistinctElements).
compute_distance_between_states(State1, State2, DistinctElements) :-
    ord_intersection(State1, State2, Intersection),
    length(Intersection, IntersectionLength),
    length(State1, Length1),
    length(State2, Length2),
    DistinctElements is (Length1 - IntersectionLength) + (Length2 - IntersectionLength).

%% relaxed_step(+State, -PositiveEffectsOfAnOperator).
relaxed_step(State, PE) :-
    % chooses a possible (ground) action
    get_operators(Operators),
    member(Operator, Operators),
    action_preconditions(Operator, TmpPreconditions),
    sort(TmpPreconditions, Preconditions),
    ord_subset(Preconditions, State),
    % retrieves its positive effects
    action_positive_effects(Operator, TmpPE),
    sort(TmpPE, PE).