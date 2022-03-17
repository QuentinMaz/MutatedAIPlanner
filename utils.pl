:- use_module(library(ordsets), [ord_subtract/3, ord_union/3, is_ordset/1, ord_subset/2]).

:- [blackboard_data, parseProblem].

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ACTIONS-RELATED PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% generate_action(-Action).
% used to build an action
generate_action(Action) :-
    get_actions(As),
    member(ActionPDDL, As),
    copy_pddl_terms(ActionPDDL, Action).

copy_pddl_terms(A, B) :- copy_pddl_terms(A, [], B, _).

copy_pddl_terms(A, Vars, A, Vars) :-
    atomic(A),
    A \= ?(_). % A does NOT represent a term to turn into a variable
copy_pddl_terms(?(V), Vars, NV, NVars) :-
    atomic(V), % A does represent a term to turn into a variable
    register_variable(V, Vars, NV, NVars). % ... and so we either get the associated variable or register a new one
copy_pddl_terms(Term, Vars, NTerm, NVars):-
    compound(Term),
    Term \= ?(_),
    Term =.. [F|Args],
    copy_pddl_arguments(Args, Vars, NArgs, NVars),
    NTerm =.. [F|NArgs].

copy_pddl_arguments([H|T], Vars, [NH|NT], NVars) :-
    copy_pddl_terms(H, Vars, NH, SVars),
    copy_pddl_arguments(T, SVars, NT, NVars).
copy_pddl_arguments([], Vars, [], Vars).

%% register_variable(+T, +L, -N, -NL).
% browses the list of couples term/var L to retrieve the variable N associated to the term T.
% If there is no such association yet, then it registers a new variable (ie, a new couple is added to NL).
register_variable(V, [X/H|T], N, [X/H|NT]) :-
    V \== X , % different variables
    register_variable(V, T, N, NT).
register_variable(V, [X/H|T], H, [X/H|T]) :-
    V == X. % same variables
% registers a new variable N to the term V
register_variable(V, [], N, [V/N]).

%% instantiate_parameters(-GroundParameters).
% makes a ground instance of a list of parameters
instantiate_parameters([]).
% already ground case (the parameter is... a constant ?)
instantiate_parameters([Parameter|Ps]) :-
    ground(Parameter),
    !,
    % to do : check whether Parameter is declared in the constants of the problem (/!\ constants is a typed list !)
    instantiate_parameters(Ps).
% untyped case : it unifies Parameter with one of the untyped parameters
instantiate_parameters([Parameter|Ps]) :-
    var(Parameter),
    !,
    get_untyped_parameters(UntypedParameters),
    member(Parameter, UntypedParameters),
    instantiate_parameters(Ps).
% typed case : it unifies Parameter with one of the matching typed parameters
instantiate_parameters([Parameter|Ps]) :-
    \+ ground(Parameter),
    !,
    Parameter =.. [TypeName, Var], % type(var)
    TypedParameter =.. [TypeName, Var], % type(parameter)
    get_typed_parameters(TypedParameters),
    member(TypedParameter, TypedParameters),
    instantiate_parameters(Ps).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLAN-RELATED PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% check_plan(+Problem, +Plan).
check_plan_validity(_, P) :-
    \+ ground(P),
    !,
    write('suspected time out\n'),
    fail.
check_plan_validity(_, []) :-
    !,
    write('empty plan\n'),
    fail.
check_plan_validity(problem(_, _, _, _, I, G, _, _, _), Plan) :-
    apply_plan(I, Plan, Result),
    ord_subset(G, Result),
    length(Plan, L),
    format('valid plan of length ~d\n', [L]).

%% apply_plan(+StartState, +Plan, -FinalState).
apply_plan(FinalState, [], FinalState).
apply_plan(State, [ActionDefinition|Plan], Result) :-
    apply_action_definition(State, ActionDefinition, NextState),
    % format('~p -- ~p --> ~p\n', [State, ActionDefinition, NewState]),
    apply_plan(NextState, Plan, Result).

%% make_path_from_plan(+StartState, +Plan, -Path).
make_path_from_plan(FinalState, [], [FinalState]).
make_path_from_plan(State, [ActionDef|T1], [State|T2]) :-
    apply_action_definition(State, ActionDef, NextState),
    make_path_from_plan(NextState, T1, T2).

%% execute_actions_from_plan(+StartState, +Plan, +NumberOfActions, -FinalState, -NextActionDefinition).
execute_actions_from_plan(ResultState, [NextActionDef|_], 0, ResultState, NextActionDef).
execute_actions_from_plan(State, [ActionDef|T], NumberOfActions, ResultState, NextActionDef) :-
    apply_action_definition(State, ActionDef, NextState),
    NewNumberOfActions is NumberOfActions - 1,
    execute_actions_from_plan(NextState, T, NewNumberOfActions, ResultState, NextActionDef).

%% apply_action(+State, +ActionDefinition, -NextState).
apply_action_definition(State, ActionDefinition, NextState) :-
    generate_action(Action),
    action_definition(Action, ActionDefinition),
    action_preconditions(Action, TmpPreconditions),
    sort(TmpPreconditions, Preconditions),
    ord_subset(Preconditions, State),
    % applies the action
    action_positive_effects(Action, TmpPE),
    sort(TmpPE, PE),
    action_negative_effects(Action, TmpNE),
    sort(TmpNE, NE),
    ord_subtract(State, NE, TmpState),
    ord_union(TmpState, PE, NextState).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PDDL-RELATED PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% make_input(+DomainFilename, +ProblemFilename, -Input).
make_input(DomainFilename, ProblemFilename, Domain-Problem) :-
    parse_domain(DomainFilename, Domain),
    parse_problem(ProblemFilename, TmpProblem),
    sort_problem(TmpProblem, Problem).

%% sort_problem(+Problem, -Result).
sort_problem(problem(N, D, R, OD, I, G, C, MS, LS), problem(N, D, R, OD, SI, SG, C, MS, LS)) :-
    sort(I, SI),
    is_ordset(SI),
    sort(G, SG),
    is_ordset(SG).