%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BLACKBOARD SETTERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

set_parameters(Domain, Problem) :-
    domain_constants(Domain, Constants),
    problem_objects(Problem, Objects),
    append(Constants, Objects, Parameters),
    (
        foreach(Parameter, Parameters),
        fromto([], TypedIn, TypedOut, TypedParameters),
        fromto([], UntypedIn, UntypedOut, UntypedParameters)
    do
        (   atom(Parameter) -> (UntypedOut = [Parameter|UntypedIn], TypedOut = TypedIn)
        ;
            (UntypedOut = UntypedIn, TypedOut = [Parameter|TypedIn])
        )
    ),
    set_untyped_parameters(UntypedParameters),
    % format('untyped parameters : ~p\n', [UntypedParameters]),
    % format('Typed parameters : ~p\n', [TypedParameters]),
    domain_types(Domain, Types),
    % format('Types : ~p\n', [Types]),
    (
        foreach(TypedParameter, TypedParameters),
        fromto([], In, Out, NewTypedParameters),
        param(Types)
    do
        TypedParameter =.. [Type, Parameter],
        inherited_types(Type, Types, InheritedTypes), % not recursive : only single-level heritage supported
        % mapping the list of inherited types with the current parameter
        (
            foreach(InheritedType, InheritedTypes),
            foreach(InheritedTypedParameter, InheritedTypedParameters),
            param(Parameter)
        do
            InheritedTypedParameter =.. [InheritedType, Parameter]
        ),
        append(In, InheritedTypedParameters, Out)
    ),
    append(NewTypedParameters, TypedParameters, FinalTypedParameters),
    % format('Final typed parameters : ~p\n', [FinalTypedParameters]),
    set_typed_parameters(FinalTypedParameters).

%% inherited_types(+Type, +Types, -InheritedTypes).
inherited_types(_, [], []).
inherited_types(Type, [H|T], [InheritedType|R]) :-
    H =.. [InheritedType, Type],
    !,
    inherited_types(Type, T, R).
inherited_types(Type, [_|T], InheritedTypes) :-
    inherited_types(Type, T, InheritedTypes).

%% fill_blackboard(+Domain, +Problem).
set_blackboard(Domain, Problem) :-
    problem_initial_state(Problem, InitialState),
    bb_put(initial_state, InitialState),
    problem_goal_state(Problem, GoalState),
    bb_put(goal_state, GoalState),
    domain_actions(Domain, Actions),
    bb_put(actions, Actions),
    domain_constants(Domain, Constants),
    bb_put(constants, Constants),
    problem_objects(Problem, Objects),
    bb_put(objects, Objects),
    set_parameters(Domain, Problem).

%% set_blackboard(+Domain, +Problem, +StateSpaceSearch, +SearchAlgorithm).
set_blackboard(Domain, Problem, StateSpaceSearch, SearchAlgorithm) :-
    set_blackboard(Domain, Problem),
    bb_put(space_search, StateSpaceSearch),
    bb_put(search_algorithm, SearchAlgorithm).

set_heuristic(Heuristic) :- bb_put(heuristic, Heuristic).

set_space_search(StateSpaceSearch) :- bb_put(space_search, StateSpaceSearch).

set_search_algorithm(SearchAlgorithm) :- bb_put(search_algorithm, SearchAlgorithm).

set_heuristic_goal(State) :- bb_put(heuristic_goal, State).

set_untyped_parameters(UntypedParameters) :- bb_put(untyped_parameters, UntypedParameters).

set_typed_parameters(TypedParameters) :- bb_put(typed_parameters, TypedParameters).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BLACKBOARD GETTERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

get_initial_state(InitialState) :- bb_get(initial_state, InitialState).

get_goal_state(GoalState) :- bb_get(goal_state, GoalState).

get_actions(Actions) :- bb_get(actions, Actions).

get_objects(Objects) :- bb_get(objects, Objects).

get_constants(Constants) :- bb_get(constants, Constants).

get_space_search(StateSpaceSearch) :- bb_get(space_search, StateSpaceSearch).

get_search_algorithm(SearchAlgorithm) :- bb_get(search_algorithm, SearchAlgorithm).

get_heuristic(Heuristic) :- bb_get(heuristic, Heuristic).

get_heuristic_goal(InitialOrGoalState) :- bb_get(heuristic_goal, InitialOrGoalState).

get_untyped_parameters(UntypedParameters) :- bb_get(untyped_parameters, UntypedParameters).

get_typed_parameters(TypedParameters) :- bb_get(typed_parameters, TypedParameters).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DATA ACCESSORS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

domain_name(domain(Name, _, _, _, _, _, _, _), Name).
domain_actions(domain(_, _, _, _, _, _, _, Actions), Actions).
domain_constants(domain(_, _, _, Constants, _, _, _, _), Constants).
domain_types(domain(_, _, Types, _, _, _, _, _), Types).

problem_name(problem(Name, _, _, _, _, _, _, _, _), Name).
problem_initial_state(problem(_, _, _, _, Init, _, _, _, _), Init).
problem_goal_state(problem(_, _, _, _, _, Goal, _, _, _), Goal).
problem_objects(problem(_, _, _, Objects, _, _, _, _, _), Objects).
problem_metric(problem(_, _, _, _, _, _, _, Metric, _), Metric).

action_parameters(action(_, Parameters, _, _, _, _), Parameters).
action_preconditions(action(_, _, Preconditions, _, _, _), Preconditions).
action_positive_effects(action(_, _, _, PositiveEffects, _, _), PositiveEffects).
action_negative_effects(action(_, _, _, _, NegativeEffects, _), NegativeEffects).
action_definition(action(Name, Parameters, _, _, _, _), ActionDefinition) :-
    untype_parameters(Parameters, UntypedParameters),
    ActionDefinition =.. [Name|UntypedParameters].

%% untype_parameters(+Parameters, -UntypedParameters).
untype_parameters([], []).
untype_parameters([TypedHead|T1], [UntypedHead|T2]) :-
    compound(TypedHead),
    TypedHead =.. [_Type, UntypedHead], % reminder : type(x)
    !,
    untype_parameters(T1, T2).
untype_parameters([H|T1], [H|T2]) :-
    untype_parameters(T1, T2).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ADDITIONAL BLACKBOARD PREDIACTES FOR METAMORPHIC TESTING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

set_problem(Problem) :-
    problem_initial_state(Problem, InitialState),
    bb_put(initial_state, InitialState),
    problem_goal_state(Problem, GoalState),
    bb_put(goal_state, GoalState),
    problem_objects(Problem, Objects),
    bb_put(objects, Objects).

set_source_result(SourceResult) :-
    bb_put(source_result, SourceResult).

get_source_result(SourceResult) :-
    bb_get(source_result, SourceResult).