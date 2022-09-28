:- module(problem,
    [
        problem_name/2, problem_initial_state/2, problem_goal_state/2, problem_objects/2,
        sort_problem/2
    ]).

:- use_module(library(ordsets), [is_ordset/1]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PROBLEM STRUCTURE AND ACCESSORS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% problem(_Name, _Domain, _Requirements, _ObjectsDeclaration, _I, _G, _C, _MS, _LS).

problem_name(problem(Name, _, _, _, _, _, _, _, _), Name).
problem_initial_state(problem(_, _, _, _, Init, _, _, _, _), Init).
problem_goal_state(problem(_, _, _, _, _, Goal, _, _, _), Goal).
problem_objects(problem(_, _, _, Objects, _, _, _, _, _), Objects).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PROBLEM-RELATED PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sort_problem(problem(N, D, R, OD, I, G, C, MS, LS), problem(N, D, R, OD, SI, SG, C, MS, LS)) :-
    sort(I, SI),
    is_ordset(SI),
    sort(G, SG),
    is_ordset(SG).