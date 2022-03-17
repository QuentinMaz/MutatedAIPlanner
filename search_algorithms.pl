:- use_module(library(ordsets), [ord_add_element/3, ord_subtract/3, ord_subset/2]).
:- use_module(library(queues), [queue_cons/3, empty_queue/1, list_queue/2, queue_append/3, queue_member/2, queue_length/2]).
:- use_module(library(heaps), [empty_heap/1, add_to_heap/4, list_to_heap/2, get_from_heap/4]).
:- use_module(library(lists), [min_member/2]).

:- [blackboard_data, state_space_searches, heuristics].

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

search(bfs, StartState, Solution) :-
    bfs_search(StartState, Solution),
    !.
search(dfs, StartState, Solution) :-
    dfs_search(StartState, Solution),
    !.
search(iddfs, StartState, Solution) :-
    iddfs_search(StartState, Solution),
    !.
search(dfs_first_solution, StartState, Solution) :-
    dfs_first_solution_search(StartState, Solution),
    !.
search(dfs_longer_solution, StartState, Solution) :-
    dfs_longer_solution_search(StartState, Solution),
    !.
search(a_star, StartState, Solution) :-
    a_star_search(StartState, Solution),
    !.
search(a_star_mutant1, StartState, Solution) :-
    a_star_mutant1_search(StartState, Solution),
    !.
search(a_star_mutant2, StartState, Solution) :-
    a_star_mutant2_search(StartState, Solution),
    !.
search(a_star_mutant3, StartState, Solution) :-
    a_star_mutant3_search(StartState, Solution),
    !.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% state_record(State, PreviousState, Action, Deep, StateRecord).
state_record(S, PS, A, D, [S, PS, A, D]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BFS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% bfs_search(+StartState, -Solution).
bfs_search(StartState, Solution) :-
    state_record(StartState, nil, nil, 0, SR),
    list_queue([SR], Q),
    % Q contains state_records
    bfs(Q, [], Solution).

%% bfs(+Queue, +Visited, -Solution).
% no solution found case : unifies an empty plan as the result
bfs(Q, _, []) :-
    empty_queue(Q),
    !.
% solution found case : solution is reached
bfs(Q, V, Solution) :-
    queue_cons(SR, _, Q),
    state_record(S, _, _, _, SR),
    % calls the goal checking predicate (wrt the planning approach)
    get_space_search(SpaceSearch),
    is_goal_state(SpaceSearch, S),
    build_solution(SR, V, TmpSolution),
    final_solution(TmpSolution, Solution).
% recursive case
bfs(Q, V, Solution) :-
    queue_cons(SR, RQ, Q),
    % state_record(S, PS, AD, D, SR),	write(S), write(' -- '), write(AD), write(' --> '), write(PS), nl,
    % state_record(S, _, _, _, SR), write('step from '), write(S), write(' : \n'),
    (bagof(NS, bfs_next_node(SR, Q, V, NS), NextNodes) ; NextNodes = []),
    % write(NextNodes), nl,
    % appends to the queue the next nodes after the tail (breath first)
    queue_append(RQ, NextNodes, NQ),
    % adds SR in the list of the visited states
    ord_add_element(V, SR, NV),
    bfs(NQ, NV, Solution).

%% build_solution(+StateRecord, +Visited, -ListOfActions).
%% It builds the solution by following the previous actions done
build_solution(SR, V, L) :-
    build_solution(SR, V, [], L).
% stop case of the recursive predicate below (when the initial state is reached)
build_solution(SR, _, L, L) :-
    state_record(_, nil, nil, _, SR),
    !.
% recursive predicate that buils the list of actions done to reach SR
build_solution(SR, V, R, L) :-
    state_record(_, PS, AD, _, SR),
    state_record(PS, _, _, _, Previous),
    memberchk(Previous, V),
    build_solution(Previous, V, [AD|R], L).

%% bfs_next_node(+StateRecord, +Queue, +Visited, -NewStateRecord).
bfs_next_node(SR, Q, V, NewSR) :-
    state_record(S, _, _, D, SR),
    % gets a reachable state (wrt the current space search) by its given ActionDef AD
    get_space_search(SpaceSearch),
    step(SpaceSearch, S, AD, NewS),
    % checks that we never visit NewS
    \+ state_member_state_records(NewS, V, SpaceSearch),
    % checks that the stateRecord associated to NewS is not already in the queue Q
    % state_record(NewS, _, _, _, Temp),
    % \+ queue_member(Temp, Q),
    list_queue(StateRecordsInQueue, Q),
    \+ state_member_state_records(NewS, StateRecordsInQueue, SpaceSearch),
    % write(S), write(' --> '), write(NewS), write(' with '), write(AD), nl,
    NewD is D + 1,
    state_record(NewS, S, AD, NewD, NewSR).

%% state_member_state_records(+State, +StateRecords, +SpaceSearch).
% checks if a state is member of a list of state records (wrt the current space search)
state_member_state_records(State, [SR|_], SpaceSearch) :-
    state_record(RecordedState, _, _, _, SR),
    are_states_equal(SpaceSearch, State, RecordedState),
    !.
state_member_state_records(State, [_|T], SpaceSearch) :-
    state_member_state_records(State, T, SpaceSearch).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DFS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% dfs_search(+InitState, -Solution).
dfs_search(I, Solution) :-
    bb_put(boundary, 21),
    bagof(SolutionDepth-Actions, dfs(I, [], 0, Actions, SolutionDepth), Results),
    min_member(_-S, Results),
    final_solution(S, Solution).
% no solution found case : unifies an empty plan as the result
dfs_search(_, []).

%% dfs(+State, +Visited, +Depth, -Solution, -SolutionDepth).
% solution found case : a solution is reached
dfs(S, _, D, [], D) :-
    % calls the goal checking predicate (wrt the planning approach)
    get_space_search(SpaceSearch),
    is_goal_state(SpaceSearch, S),
    % always updates the maximum depth value
    % as every new solution is shorter than the previous ones
    NewD is D - 1,
    bb_put(boundary, NewD),
    % format('Depth of solution : ~d ; now tries to find a solution of cost : ~p\n', [D, NewD]),
    !.
dfs(S, V, D, [ActionDef|Ls], SD) :-
    % gets a reachable state (wrt the current space search) by its given ActionDef AD
    get_space_search(SpaceSearch),
    bagof(NS-AD, step(SpaceSearch, S, AD, NS), NextStates),
    % iterates over the successors
    member(NextState-ActionDef, NextStates),
    % checks whether the given successor has already been visited
    \+ member_custom(NextState, V, SpaceSearch), % /!\ should take [S|V] !...
    % checks the boundary at the end to prevent searching other solutions of same length
    test_boundary(D),
    NewD is D + 1,
    dfs(NextState, [S|V], NewD, Ls, SD).

%% test_boundary(+Depth).
% true is Depth is lower than the boundary value (blackboard)
test_boundary(Depth) :-
    bb_get(boundary, Boundary),
    Depth < Boundary.

%% member_custom(+Element, +List, +SpaceSearch).
% checks if Element is member of List by using are_states_equal/3 (which depends on the current space search)
member_custom(E, [H|_], SpaceSearch) :-
    are_states_equal(SpaceSearch, E, H),
    !.
member_custom(E, [_|T], SpaceSearch) :-
    member_custom(E, T, SpaceSearch).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% IDDFS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

iddfs_search(StartState, Solution) :-
    iddfs(StartState, 1, TmpSolution),
    final_solution(TmpSolution, Solution).

%% iddfs(+StartState, +Boundary, -Solution).
iddfs(StartState, Boundary, Solution) :-
    bb_put(boundary, Boundary),
    dfs(StartState, [], 0, Solution, _),
    !.
iddfs(StartState, Boundary, Solution) :-
    NewBoundary is Boundary + 1,
    iddfs(StartState, NewBoundary, Solution).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DFS FIRST SOLUTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% dfs_first_solution_search(+StarState, -Solution).
dfs_first_solution_search(StartState, Solution) :-
    bb_put(boundary, 21),
    % gets the first plan found
    dfs(StartState, [], 0, ActionDefinitions, _SolutionDepth),
    !,
    final_solution(ActionDefinitions, Solution).
% no solution found case : unifies an empty plan as the result
dfs_first_solution_search(_, []).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DFS LONGER SOLUTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% dfs_longer_solution_search(+StarState, -Solution).
dfs_longer_solution_search(StartState, Solution) :-
    iddfs(StartState, 1, _OptimalSolution),
    bb_get(boundary, Boundary),
    bb_put(boundary, 100),
    dfs_with_minimum(StartState, [], 0, ActionDefinitions, Boundary),
    !,
    final_solution(ActionDefinitions, Solution).
% no found solution case : it might happen that all solutions are optimal (so it returns an empty plan)
dfs_longer_solution_search(_, []).

%% dfs_with_minimum(+State, +Visited, +Depth, -Solution, +Minimum).
% solution found case : a solution longer than Minimum is reached
dfs_with_minimum(State, _, Depth, [], Minimum) :-
    get_space_search(SpaceSearch),
    is_goal_state(SpaceSearch, State),
    Depth > Minimum,
    !.
dfs_with_minimum(State, Visited, Depth, [ActionDef|Ls], Minimum) :-
    test_boundary(Depth),
    NewDepth is Depth + 1,
    get_space_search(SpaceSearch),
    bagof(NS-AD, step(SpaceSearch, State, AD, NS), NextStates),
    member(NextState-ActionDef, NextStates),
    \+ member_custom(NextState, Visited, SpaceSearch),
    dfs_with_minimum(NextState, [State|Visited], NewDepth, Ls, Minimum).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A*
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% a_star_search(+StartState, -Solution).
a_star_search(StartState, Solution) :-
    state_record(StartState, nil, nil, 0, StateRecord),
    list_to_heap([0-StateRecord], PQ),
    get_space_search(SpaceSearch),
    % PQ is a heap which sorts the states to expand by their estimated cost f = c + h
    a_star(PQ, [], Solution, SpaceSearch).

%% a_star(+Queue, +Visited, -Solution, +SpaceSearch).
% no solution found case : unifies an empty plan as the result
a_star(PQ, _, [], _) :-
    empty_heap(PQ),
    !.
a_star(PQ, V, Solution, SpaceSearch) :-
    get_from_heap(PQ, _, SR, _),
    state_record(S, _, _, _, SR),
    is_goal_state(SpaceSearch, S),
    build_cheapest_solution(SR, V, TmpSolution),
    final_solution(TmpSolution, Solution).
% recursive case : the next cheapest state is worth being expanded
a_star(PQ, V, Solution, SpaceSearch) :-
    get_from_heap(PQ, _, SR, RPQ),
    state_record(S, _, _, Cost, SR),
    \+ member_and_higher_weight(S, V, Cost, SpaceSearch), % state is worth being expanded
    !,
    % expands the state (we know is not a goal state)
    ord_add_element(V, SR, NV),
    % format('~d ~p\n', [Cost, S]),
    (bagof(F-NextStateRecord, a_star_next_state(SR, NV, F, NextStateRecord, SpaceSearch), NextStateRecords) ; NextStateRecords = []),
    add_list_to_heap(RPQ, NextStateRecords, NPQ),
    a_star(NPQ, NV, Solution, SpaceSearch).
% last recursive case : the next cheapest state is not worth being expanded
a_star(PQ, V, Solution, SpaceSearch) :-
    get_from_heap(PQ, _, _, NPQ),
    a_star(NPQ, V, Solution, SpaceSearch).

%% a_star_next_state(+StateRecord, +Visited, -EstimatedCost, -NewStateRecord, +SpaceSearch).
a_star_next_state(SR, V, F, NewStateRecord, SpaceSearch) :-
    state_record(State, _, _, Depth, SR),
    get_heuristic(Heuristic),
    NewDepth is Depth + 1,
    % gets a reachable state (wrt the current space search)
    step(SpaceSearch, State, ActionDefinition, NewState),
    % checks that NewState has not been visited yet (or already visited but with a higher cost)
    \+ member_and_higher_weight(NewState, V, NewDepth, SpaceSearch),
    h(Heuristic, NewState, H),
    % format('h(~p) = ~d\n', [NewState, H]),
    F is NewDepth + H,
    % write(NewState), write(' costs : '), write(F), write(' ('), write(H), write(' + '), write(NewDepth), write(')'), nl,
    state_record(NewState, State, ActionDefinition, NewDepth, NewStateRecord).

%% add_list_to_heap(+OldHeap, +List, -NewHeap).
add_list_to_heap(OH, [], OH).
add_list_to_heap(OH, [K-D|T], NH) :-
    add_to_heap(OH, K, D, H),
    add_list_to_heap(H, T, NH).

%% member_and_higher_weight(+Element, +List, +Weight, +SpaceSearch).
% It checks if Element is in List with a higher weight.
% Element is a state (list).
% List is a list of state records.
% Weight is a numerical value.
% SpaceSearch is an atom to select the correct version of the predicate are_states_equal/3.
% This helper predicate is used to re-add in the open list states already visited but with a longer path.
% It has to be used when the heuristic is not monotone / consistent (... most cases).
member_and_higher_weight(S, [SR|_], K, SpaceSearch) :-
    state_record(S2, _, _, D2, SR),
    are_states_equal(SpaceSearch, S, S2),
    K >= D2,
    !.
member_and_higher_weight(S, [_|T], K, SpaceSearch) :-
    member_and_higher_weight(S, T, K, SpaceSearch).

%% build_cheapest_solution(+StateRecord, +Visited, -ListOfActions).
% It is similar to build_solution. However, it considers the cheapest list of actions.
% This predicate is mandatory when nodes have been revisited (ie, non-monotone heuristic is used).
build_cheapest_solution(SR, V, L) :-
    build_cheapest_solution(SR, V, [], L).
build_cheapest_solution(SR, _, L, L) :-
    state_record(_, nil, nil, _, SR),
    !.
build_cheapest_solution(SR, V, R, L) :-
    state_record(_, PS, AD, _, SR),
    % gets all the previous state leading to PS that have been visited...
    findall(Prev, (state_record(PS, _, _, _, Prev), member(Prev, V)), Trail),
    % ... and takes the cheapest state from them
    choose_min_prev(Trail, Min),
    % recursive call while stacking AD in the list of the actions
    build_cheapest_solution(Min, V, [AD|R], L).

%% choose_min_prev(+List, -Minimum).
% When there are multiple nodes to backtrack to, this chooses the one with lowest depth.
choose_min_prev([H|T], Min) :-
    choose_min_prev(T, H, Min).
% stops when the list is empty
choose_min_prev([], Min, Min) :-
    !.
% updates Current with H if H's depth is lower than Current's depth
choose_min_prev([H|T], Current, Min) :-
    state_record(_, _, _, D_Current, Current),
    state_record(_, _, _, D_New, H),
    D_New < D_Current,
    choose_min_prev(T, H, Min).
% otherwise browses the list
choose_min_prev([_|T], Current, Min) :-
    choose_min_prev(T, Current, Min).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A* MUTANT1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% a_star_mutant1_search(+StartState, -Solution).
a_star_mutant1_search(StartState, Solution) :-
    state_record(StartState, nil, nil, 0, StateRecord),
    list_to_heap([0-StateRecord], PQ),
    get_space_search(SpaceSearch),
    % PQ is a heap which sorts the states to expand by their estimated cost f = -(c + h)
    a_star_mutant1(PQ, [], Solution, SpaceSearch).

% no solution found case :unifies an empty plan as the result
a_star_mutant1(PQ, _, [], _) :-
    empty_heap(PQ),
    !.
% solution found case : the next state to expand (which is the cheapest one) is a goal state
a_star_mutant1(PQ, V, Solution, SpaceSearch) :-
    get_from_heap(PQ, _, SR, _),
    state_record(S, _, _, _, SR),
    is_goal_state(SpaceSearch, S),
    build_cheapest_solution(SR, V, TmpSolution),
    final_solution(TmpSolution, Solution).
% recursive case : the next cheapest state is worth being expanded
a_star_mutant1(PQ, V, Solution, SpaceSearch) :-
    get_from_heap(PQ, _, SR, RPQ),
    \+ member(SR, V), % state is worth being expanded
    !,
    % expands the state (we know is not a goal state)
    ord_add_element(V, SR, NV),
    (bagof(F-NextStateRecord, a_star_mutant1_next_state(SR, NV, F, NextStateRecord, SpaceSearch), NextStateRecords) ; NextStateRecords = []),
    add_list_to_heap(RPQ, NextStateRecords, NPQ),
    a_star_mutant1(NPQ, NV, Solution, SpaceSearch).
% last recursive case : the next cheapest state is not worth being expanded
a_star_mutant1(PQ, V, Solution, SpaceSearch) :-
    get_from_heap(PQ, _, _, NPQ),
    a_star_mutant1(NPQ, V, Solution, SpaceSearch).

%% a_star_mutant1_next_state(+StateRecord, +Visited, -EstimatedCost, -NewStateRecord, +SpaceSearch).
a_star_mutant1_next_state(SR, V, F, NewStateRecord, SpaceSearch) :-
    state_record(State, _, _, Depth, SR),
    NewDepth is Depth + 1,
    step(SpaceSearch, State, ActionDefinition, NewState),
    state_record(NewState, _, _, _, TmpSR),
    \+ member(TmpSR, V),
    get_heuristic(Heuristic),
    h(Heuristic, NewState, H),
    F is -1 * (NewDepth + H),
    state_record(NewState, State, ActionDefinition, NewDepth, NewStateRecord).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A* MUTANT2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% a_star_mutant2_search(+StartState, -Solution).
a_star_mutant2_search(StartState, Solution) :-
    state_record(StartState, nil, nil, 0, StateRecord),
    list_to_heap([0-StateRecord], PQ),
    get_space_search(SpaceSearch),
    a_star_mutant2(PQ, [], Solution, SpaceSearch).

% no solution found case :unifies an empty plan as the result
a_star_mutant2(PQ, _, [], _) :-
    empty_heap(PQ),
    !.
% solution found case : the next state to expand (which is the cheapest one) is a goal state
a_star_mutant2(PQ, V, Solution, SpaceSearch) :-
    get_from_heap(PQ, _, SR, _),
    state_record(S, _, _, _, SR),
    is_goal_state(SpaceSearch, S),
    build_cheapest_solution(SR, V, TmpSolution),
    final_solution(TmpSolution, Solution).
% recursive case : the next cheapest state is worth being expanded
a_star_mutant2(PQ, V, Solution, SpaceSearch) :-
    get_from_heap(PQ, _, SR, RPQ),
    \+ member(SR, V), % state is worth being expanded
    !,
    % expands the state (we know is not a goal state)
    ord_add_element(V, SR, NV),
    (bagof(F-NextStateRecord, a_star_mutant2_next_state(SR, NV, F, NextStateRecord, SpaceSearch), NextStateRecords) ; NextStateRecords = []),
    add_list_to_heap(RPQ, NextStateRecords, NPQ),
    a_star_mutant2(NPQ, NV, Solution, SpaceSearch).
% last recursive case : the next cheapest state is not worth being expanded
a_star_mutant2(PQ, V, Solution, SpaceSearch) :-
    get_from_heap(PQ, _, _, NPQ),
    a_star_mutant2(NPQ, V, Solution, SpaceSearch).

%% a_star_mutant2_next_state(+StateRecord, +Visited, -EstimatedCost, -NewStateRecord, +SpaceSearch).
a_star_mutant2_next_state(SR, V, F, NewStateRecord, SpaceSearch) :-
    state_record(State, _, _, Depth, SR),
    NewDepth is Depth + 1,
    step(SpaceSearch, State, ActionDefinition, NewState),
    state_record(NewState, _, _, _, TmpSR),
    \+ member(TmpSR, V),
    get_heuristic(Heuristic),
    h(Heuristic, NewState, H),
    F is H - NewDepth,
    state_record(NewState, State, ActionDefinition, NewDepth, NewStateRecord).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A* MUTANT3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% a_star_mutant3_search(+StartState, -Solution).
a_star_mutant3_search(StartState, Solution) :-
    state_record(StartState, nil, nil, 0, StateRecord),
    list_to_heap([0-StateRecord], PQ),
    get_space_search(SpaceSearch),
    a_star_mutant3(PQ, [], Solution, SpaceSearch).

% no solution found case :unifies an empty plan as the result
a_star_mutant3(PQ, _, [], _) :-
    empty_heap(PQ),
    !.
% solution found case : the next state to expand (which is the cheapest one) is a goal state
a_star_mutant3(PQ, V, Solution, SpaceSearch) :-
    get_from_heap(PQ, _, SR, _),
    state_record(S, _, _, _, SR),
    is_goal_state(SpaceSearch, S),
    build_cheapest_solution(SR, V, TmpSolution),
    final_solution(TmpSolution, Solution).
% recursive case : the next cheapest state is worth being expanded
a_star_mutant3(PQ, V, Solution, SpaceSearch) :-
    get_from_heap(PQ, _, SR, RPQ),
    \+ member(SR, V), % state is worth being expanded
    !,
    % expands the state (we know is not a goal state)
    ord_add_element(V, SR, NV),
    (bagof(F-NextStateRecord, a_star_mutant3_next_state(SR, NV, F, NextStateRecord, SpaceSearch), NextStateRecords) ; NextStateRecords = []),
    add_list_to_heap(RPQ, NextStateRecords, NPQ),
    a_star_mutant3(NPQ, NV, Solution, SpaceSearch).
% last recursive case : the next cheapest state is not worth being expanded
a_star_mutant3(PQ, V, Solution, SpaceSearch) :-
    get_from_heap(PQ, _, _, NPQ),
    a_star_mutant3(NPQ, V, Solution, SpaceSearch).

%% a_star_mutant3_next_state(+StateRecord, +Visited, -EstimatedCost, -NewStateRecord, +SpaceSearch).
a_star_mutant3_next_state(SR, V, F, NewStateRecord, SpaceSearch) :-
    state_record(State, _, _, Depth, SR),
    NewDepth is Depth + 1,
    step(SpaceSearch, State, ActionDefinition, NewState),
    state_record(NewState, _, _, _, TmpSR),
    \+ member(TmpSR, V),
    get_heuristic(Heuristic),
    h(Heuristic, NewState, H),
    F is -1 * NewDepth + 10 * H,
    state_record(NewState, State, ActionDefinition, NewDepth, NewStateRecord).