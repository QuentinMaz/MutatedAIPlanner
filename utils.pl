:- module(utils, [predicates_names/2, filter_predicates/3, filter_predicates_with_names/3, validate_plan/2, progress/3]).

:- use_module(library(ordsets), [ord_subset/2, ord_subtract/3, ord_union/3]).

:- ensure_loaded(domain).
:- ensure_loaded(blackboard_data).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FILTERING PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% retrieves all the names of the terms in Predicates (/!\ may have duplicates)
%% predicates_names(+Predicates, -PredicatesNames).
predicates_names([], []).
predicates_names([H|T1], [Name|T2]) :-
    H =.. [Name|_],
    predicates_names(T1, T2).

% accumulates the predicates whose name matches Name
%% filter_predicates(+Pred, +Name, -Res).
filter_predicates([], _, []).
filter_predicates([H|T1], Name, [H|T2]) :-
    H =.. [Name|_],
    !,
    filter_predicates(T1, Name, T2).
filter_predicates([_|T1], Name, Results) :-
    filter_predicates(T1, Name, Results).

% accumulates the result of filter_predicates/3 in Res (/!\ not not flattened)
%% filter_predicates_with_names(-Pred, +Names, -Res).
filter_predicates_with_names(_, [], []).
filter_predicates_with_names(Predicates, [Name|T1], [MatchedPredicates|T2]) :-
    filter_predicates(Predicates, Name, MP),
    sort(MP, MatchedPredicates),
    filter_predicates_with_names(Predicates, T1, T2).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLAN & ACTIONS RELATED PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% check_plan(+Problem, +Plan).
validate_plan(_, P) :-
    \+ ground(P),
    !,
    write('suspected time out\n'),
    fail.
validate_plan(_, []) :-
    !,
    write('empty plan\n'),
    fail.
validate_plan(problem(_, _, _, _, I, G, _, _, _), Plan) :-
    apply_plan(I, Plan, Result),
    ord_subset(G, Result),
    length(Plan, L),
    format('valid plan of length ~d\n', [L]).

%% apply_plan(+StartState, +Plan, -FinalState).
apply_plan(FinalState, [], FinalState).
apply_plan(State, [Operator|Plan], Result) :-
    progress(State, Operator, NextState),
    % format('~p -- ~p --> ~p\n', [State, Operator, NewState]),
    apply_plan(NextState, Plan, Result).

%% progress(+State, ?Operator, -NextState).
progress(State, Operator, NextState) :-
    % retrieves from the ground actions a possible operator
    get_operators(Operators),
    member(Operator, Operators),
    action_preconditions(Operator, TmpPreconditions),
    sort(TmpPreconditions, Preconditions),
    ord_subset(Preconditions, State),
    % applies the operator
    action_positive_effects(Operator, TmpPE),
    sort(TmpPE, PE),
    action_negative_effects(Operator, TmpNE),
    sort(TmpNE, NE),
    ord_subtract(State, NE, TmpState),
    ord_union(TmpState, PE, NextState).