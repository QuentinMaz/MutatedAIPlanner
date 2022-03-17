:- use_module(library(ordsets), [ord_subset/2, ord_subtract/3, ord_union/3, ord_intersect/2, ord_disjoint/2, ord_add_element/3]).
:- use_module(library(sets), [is_set/1]).
:- use_module(library(lists), [reverse/2, some/3, some/2, select/3]).

:- [blackboard_data, utils].

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

initialise_start_state(forward, StartState) :-
    !,
    fwd_start_state(StartState).
initialise_start_state(backward, StartState) :-
    !,
    bwd_start_state(StartState).

is_goal_state(forward, State) :-
    !,
    is_fwd_goal_state(State).
is_goal_state(backward, State) :-
    !,
    is_bwd_goal_state(State).

step(forward, State, ActionDef, NewState) :-
    !,
    fwd_step(State, ActionDef, NewState).
step(backward, State, ActionDef, NewState) :-
    !,
    bwd_step(State, ActionDef, NewState).

are_states_equal(forward, S1, S2) :-
    !,
    fwd_are_states_equal(S1, S2).
are_states_equal(backward, S1, S2) :-
    !,
    bwd_are_states_equal(S1, S2).

final_solution(Solution, Solution) :-
    get_space_search(StateSpaceSearch),
    StateSpaceSearch = forward,
    !.
final_solution(TmpSolution, Solution) :-
    get_space_search(StateSpaceSearch),
    StateSpaceSearch = backward,
    reverse(TmpSolution, Solution),
    !.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FORWARD SEARCH
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% fwd_start_state(-StartState).
fwd_start_state(I) :-
    get_initial_state(I),
    get_goal_state(G),
    set_heuristic_goal(G).

%% fwd_step(+State, -ActionDef, -NewState).
% computes a progression step possible from State. ActionDef describes the action performed and NewState the new state reached.
% NewState is ordered.
fwd_step(State, ActionDef, NewState) :-
    % generates a free action...
    generate_action(Action),
    action_parameters(Action, Parameters),
    % makes it ground with every combinaison of the problem's objects possible
    instantiate_parameters(Parameters),
    is_set(Parameters),
    % and checks that its preconditions are included in the current state
    action_preconditions(Action, TmpP),
    sort(TmpP, Preconditions),
    ord_subset(Preconditions, State),
    % modifies the current state with the action's effects
    action_positive_effects(Action, TmpPE),
    sort(TmpPE, PE),
    action_negative_effects(Action, TmpNE),
    sort(TmpNE, NE),
    ord_subtract(State, NE, TmpState),
    ord_union(TmpState, PE, NewState),
    action_definition(Action, ActionDef). % gets the definition of the action (by untyping it)

is_fwd_goal_state(State) :-
    get_goal_state(GoalState),
    % every element of the ordered set G appears in the ordered set S
    ord_subset(GoalState, State).

%% fwd_are_states_equal(+State1, +State2).
%% Predicate used to determine if State1 equals State2 (when progressing).
fwd_are_states_equal(State1, State2) :-
    State1 = State2. % States have to be the same

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BACKWARD SEARCH
%% /!\ LEADS TO UNREACHABLE STATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% bwd_start_state(-StartState).
bwd_start_state(StartState) :-
    get_goal_state(StartState),
    get_initial_state(InitialState),
    set_heuristic_goal(InitialState).

%% bwd_step(+State, -ActionDef, -PreviousState).
bwd_step(State, ActionDef, PreviousState) :-
    % generates a free action...
    generate_action(Action),
    action_parameters(Action, Parameters),
    % makes it ground with every combinaison of the problem's objects possible
    instantiate_parameters(Parameters),
    is_set(Parameters),
    % gets its effects
    action_positive_effects(Action, TmpPE),
    sort(TmpPE, PE),
    action_negative_effects(Action, TmpNE),
    sort(TmpNE, NE),
    %% relevance checking of the given instancied action A
    % checks that the intersection between PE and G is not empty
    % (it means that the given action makes at least one the goal's literals true)
    ord_intersect(State, PE),
    % checks that the intersection between NE and G is empty
    % (it means that the given action does not make any of goal's literals false)
    ord_disjoint(State, NE),
    % retrieves the preconditions of the action once checking is done
    action_preconditions(Action, TmpP),
    sort(TmpP, Preconditions),
    % PreviousState is made of State (from which we subtract the own effects of A) with the preconditions of A
    ord_subtract(State, PE, TmpState),
    ord_union(TmpState, Preconditions, PreviousState),
    action_definition(Action, ActionDef). %,
    % format('regressed state : ~p\n', [PreviousState]).

is_bwd_goal_state(State) :-
    get_initial_state(InitialState),
    %% In regression, search states are sub-goals (we would need to achieve).
    %% So, we achieve our goal if the initial state satisfies all the needed sub-goals (contains the state S).
    ord_subset(State, InitialState).

%% bwd_are_states_equal(+State1, +State2).
%% Predicate used to determine if State1 has already been visited knowing that State2 has so (when regressing).
bwd_are_states_equal(State1, State2) :-
    %% If State2 has already been visited, then the initial state does not include it.
    %% So, if State1 contains State2 then for sure the initial state won't satisfy State1.
    %% State1 is thus considered as alreaby been visited !
    ord_subset(State2, State1). % true is State1 contains State2