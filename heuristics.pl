:- use_module(library(ordsets), [ord_subset/2, ord_subtract/3, ord_union/2]).
:- use_module(library(sets), [is_set/1]).

:- [blackboard_data, utils].

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h(h_0, State, Result) :-
    h_0(State, Result),
	!.
h(h_diff, State, Result) :-
    h_diff(State, Result),
    !.
h(h_plus, State, Result) :-
    h_plus(State, Result),
    !.
h(h_add, State, Result) :-
    h_add(State, Result),
    !.
h(h_length, State, Result) :-
    h_length(State, Result),
    !.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% h_0 heuristic :
%% It implements the simpliest heuristic function,
%% as it always returns 0.
%%
%% Admissible and consistent.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h_0(_, 0).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% h_plus heuristic :
%% It is an approximaiton of the classical (but too expansive) h+ heuristic.
%% Indeed, in the delete (relaxed) problem, the order off all possible actions
%% from a state doesn't matter (as we'll only add their positive effects without considering the negative ones).
%% So here we greatly reduce the cost of the h+ heuristic by considering all the possible actions at each state.
%%
%% Precisely, at each step, it adds to the current state all the positive effects of every possible action.
%% Between each step, the heuristic value is incremented.
%% Once the goal is contained by the ever-growing state, it unifies 0
%% and the heuristic value is finally computed.
%%
%% So it counts the ideal hypothetical minimum number of actions needed to reach the goal.
%% (It is linked to the reachability graph.)
%% Admissible.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% unifies 0 if the current state S contains the goal G
h_plus(State, 0) :-
    get_heuristic_goal(Goal),
    ord_subset(Goal, State),
    !.
h_plus(State, NewDepth) :-
    setof(PE, h_step(State, PE), SetOfPE),
    ord_union([State|SetOfPE], TmpState),
    sort(TmpState, NewState),
    NewState \= State,
    h_plus(NewState, Depth),
    NewDepth is Depth + 1.
% limits the search in case of unreachable goal state
h_plus(_, 0). % (only seen in problems based on the airport domain...)

h_step(State, PE) :-
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
    % gets only its positive effects
    action_positive_effects(Action, TmpPE),
    sort(TmpPE, PE).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% h_diff heuristic :
%% It is based on the difference between the goal and the current state.
%% It returns the number of literals
%% that needs to be true in order to satisfy the goal.
%%
%% Not admissible.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h_diff(S, E) :-
    get_heuristic_goal(G),
    ord_subtract(G, S, I),
    length(I, E).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% h_add heuristic function :
%% At each step, it adds to the current state all the positive effects
%% of every possible action.
%% Between each step, it subtracts from the goal all the facts already satisfied,
%% and adds the number of the remaining facts to the estimated cost.
%% Once the goal is empty, it unifies 0 and so the final estimated value
%% is the sum of all remaining facts to satisfy left (at each step).
%%
%% Not admissible.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h_add(State, Heuristic) :-
    get_heuristic_goal(Goal),
    h_add(State, Goal, Heuristic).

h_add(_, [], 0) :- !.
h_add(State, Goal, Heuristic) :-
    ord_subtract(Goal, State, Delta),
    (setof(PE, h_step(State, PE), SetOfPE) ; SetOfPE = []),
    ord_union([State|SetOfPE], TmpState),
    sort(TmpState, NewState),
    NewState \= State,
    h_add(NewState, Delta, NextHeuristic),
    length(Delta, DeltaLength),
    Heuristic is DeltaLength + NextHeuristic.
% limits the search in case of unreachable goal state
h_add(State, Goal, SizeOfDelta) :-
    ord_subtract(Goal, State, Delta),
    length(Delta, SizeOfDelta).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% h_length heuristic function :
%% Used for mutated a* algorithm. The heuristic is equal to the number
%% of literals of the current state.
%%
%% Not admissible.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h_length(State, Heuristic) :-
    length(State, Heuristic).