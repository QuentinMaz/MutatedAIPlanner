:- module(blackboard_data,
    [
        set_blackboard/2,

        get_problem/1, get_domain/1, get_actions/1, get_objects/1, get_constants/1,
        get_rigid_predicates_names/1, get_rigid_facts/1, get_operators/1,
        get_untyped_variables/1, get_typed_variables/1, get_heuristic/1,

        set_operators/1, set_heuristic/1
    ]).

:- use_module(library(sets), [subtract/3]).

:- ensure_loaded(domain).
:- ensure_loaded(problem).
:- ensure_loaded(utils).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BLACKBOARD SETTERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% set_blackboard(+Domain, +Problem).
set_blackboard(Domain, Problem) :-
    bb_put(domain, Domain),
    bb_put(problem, Problem),
    domain_actions(Domain, Actions),
    bb_put(actions, Actions),
    domain_constants(Domain, Constants),
    bb_put(constants, Constants),
    problem_objects(Problem, Objects),
    bb_put(objects, Objects),
    domain_predicates(Domain, Predicates),
    bb_put(predicates, Predicates),
    compute_variables(Domain, Problem).

set_untyped_variables(UntypedVariables) :- bb_put(untyped_variables, UntypedVariables).

set_typed_variables(TypedVariables) :- bb_put(typed_variables, TypedVariables).

set_operators(Operators) :- bb_put(operators, Operators).

set_heuristic(Heuristic) :- bb_put(heuristic, Heuristic).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BLACKBOARD GETTERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

get_domain(Domain) :- bb_get(domain, Domain).

get_problem(Problem) :- bb_get(problem, Problem).

get_actions(Actions) :- bb_get(actions, Actions).

get_objects(Objects) :- bb_get(objects, Objects).

get_constants(Constants) :- bb_get(constants, Constants).

get_rigid_predicates_names(RigidPredicatesNames) :- bb_get(rigid_predicates_names, RigidPredicatesNames).

get_rigid_facts(RigidFacts) :- bb_get(rigid_facts, RigidFacts).

get_untyped_variables(UntypedVariables) :- bb_get(untyped_variables, UntypedVariables).

get_typed_variables(TypedVariables) :- bb_get(typed_variables, TypedVariables).

get_operators(Operators) :- bb_get(operators, Operators).

get_heuristic(Heuristic) :- bb_get(heuristic, Heuristic).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% VARIABLES PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

compute_variables(Domain, Problem) :-
    domain_constants(Domain, Constants),
    problem_objects(Problem, Objects),
    append(Constants, Objects, Variables),
    (
        foreach(Variable, Variables),
        fromto([], TypedIn, TypedOut, TypedVariables),
        fromto([], UntypedIn, UntypedOut, UntypedVariables)
    do
        (   atom(Variable) -> (UntypedOut = [Variable|UntypedIn], TypedOut = TypedIn)
        ;
            (UntypedOut = UntypedIn, TypedOut = [Variable|TypedIn])
        )
    ),
    set_untyped_variables(UntypedVariables),
    % format('untyped variables: ~p\n', [UntypedVariables]),
    % format('Typed variables: ~p\n', [TypedVariables]),
    domain_types(Domain, Types),
    % format('Types: ~p\n', [Types]),
    (
        foreach(TypedVariable, TypedVariables),
        fromto([], In, Out, NewTypedVariables),
        param(Types)
    do
        TypedVariable =.. [Type, Variable],
        % #TODO : inherited types are each time re-computed
        get_inherited_types([Type], Types, InheritedTypes),
        % mapping the list of inherited types with the current variable
        (
            foreach(InheritedType, InheritedTypes),
            foreach(InheritedTypedVariable, InheritedTypedVariables),
            param(Variable)
        do
            InheritedTypedVariable =.. [InheritedType, Variable]
        ),
        append(In, InheritedTypedVariables, Out)
    ),
    append(NewTypedVariables, TypedVariables, FinalTypedVariables),
    filter_predicates(FinalTypedVariables, object, VariablesOfTypeObject),
    subtract(FinalTypedVariables, VariablesOfTypeObject, Results),
    % format('Typed variables: ~p\n', [Results]),
    set_typed_variables(Results).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HELPERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% inherited_types(+Type, +Types, -InheritedTypes).
inherited_types(_, [], []).
inherited_types(Type, [H|T], [InheritedType|R]) :-
    H =.. [InheritedType, Type],
    !,
    inherited_types(Type, T, R).
inherited_types(Type, [_|T], InheritedTypes) :-
    inherited_types(Type, T, InheritedTypes).

get_inherited_types([], _, []).
get_inherited_types([Type|T], Types, InheritedTypes) :-
    inherited_types(Type, Types, IT1),
    append(T, IT1, NewT),
    get_inherited_types(NewT, Types, IT2),
    append(IT1, IT2, InheritedTypes).