:- module(searches, [search/3, bfs_search/2, astar_search/2]).

:- use_module(library(ordsets), [ord_add_element/3, ord_subtract/3, ord_subset/2]).
:- use_module(library(queues), [queue_cons/3, empty_queue/1, list_queue/2, queue_append/3, queue_member/2, queue_length/2]).
:- use_module(library(heaps), [empty_heap/1, add_to_heap/4, list_to_heap/2, get_from_heap/4]).
:- use_module(library(lists), [min_member/2, maplist/3]).

:- ensure_loaded(blackboard_data).
:- ensure_loaded(heuristics).
:- ensure_loaded(problem).
:- ensure_loaded(domain).
:- ensure_loaded(utils).

%% state_record(+State, +PreviousState, +Operator, +Depth, -StateRecord).
state_record(S, PS, Op, D, [S, PS, Op, D]).

search(bfs, StartState, Solution) :-
    bfs_search(StartState, Solution),
    !.
search(astar, StartState, Solution) :-
    astar_search(StartState, Solution),
    !.
search(MA, StartState, Solution) :-
    atom_concat('mutated_astar-', FFunction, MA),
    mutated_astar_search(StartState, FFunction, Solution),
    !.

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
    state_record(State, _, _, _, SR),
    get_problem(Problem),
    problem_goal_state(Problem, GoalState),
    ord_subset(GoalState, State),
    % length(V, L), format('~d states visited\n', [L]),
    build_solution(SR, V, Solution).
% recursive case
bfs(Q, V, Solution) :-
    queue_cons(SR, RQ, Q),
    % state_record(S, PS, AD, D, SR),	write(PS), write(' -- '), write(AD), write(' --> '), write(S), nl,
    % state_record(S, _, _, _, SR), write('step from '), write(S), write(' : \n'),
    (bagof(NS, bfs_next_node(SR, Q, V, NS), NextNodes) ; NextNodes = []),
    % length(NextNodes, NNL), format('adds ~d nodes to the queue\n', [NNL]),
    % appends to the queue the next nodes after the tail (breath first)
    queue_append(RQ, NextNodes, NQ),
    % adds SR in the list of the visited states
    ord_add_element(V, SR, NV),
    bfs(NQ, NV, Solution).

%% bfs_next_node(+StateRecord, +Queue, +Visited, -NewStateRecord).
bfs_next_node(SR, Q, V, NewSR) :-
    state_record(S, _, _, D, SR),
    % progresses forward to NewS with the operator Op
    progress(S, Op, NewS),
    % checks that we never visit NewS
    \+ state_member_state_records(NewS, V),
    % checks that the stateRecord associated to NewS is not already in the queue Q
    % state_record(NewS, _, _, _, Temp),
    % \+ queue_member(Temp, Q),
    list_queue(StateRecordsInQueue, Q),
    \+ state_member_state_records(NewS, StateRecordsInQueue),
    % write(S), write(' --> '), write(NewS), write(' with '), write(Op), nl,
    NewD is D + 1,
    state_record(NewS, S, Op, NewD, NewSR).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A*
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% astar_search(+StartState, -Solution).
astar_search(StartState, Solution) :-
    state_record(StartState, nil, nil, 0, StateRecord),
    list_to_heap([0-StateRecord], PQ),
    % PQ is a heap which sorts the states to expand by their estimated cost f = c + h
    astar(PQ, [], Solution).

%% a_star(+Queue, +Visited, -Solution).
% no solution found case: unifies an empty plan as the result
astar(PQ, _, []) :-
    empty_heap(PQ),
    !.
astar(PQ, V, Solution) :-
    get_from_heap(PQ, _, SR, _),
    state_record(State, _, _, _, SR),
    get_problem(Problem),
    problem_goal_state(Problem, GoalState),
    ord_subset(GoalState, State),
    % length(V, L), format('~d states visited\n', [L]),
    build_cheapest_solution(SR, V, Solution).
% recursive case: the next cheapest state is worth being expanded
astar(PQ, V, Solution) :-
    get_from_heap(PQ, _, SR, RPQ),
    state_record(S, _, _, Cost, SR),
    \+ member_and_higher_weight(S, V, Cost), % state is worth being expanded
    !,
    % expands the state (we know is not a goal state)
    ord_add_element(V, SR, NV),
    % format('~d ~p\n', [Cost, S]),
    (bagof(F-NextStateRecord, astar_next_state(SR, NV, F, NextStateRecord), NextStateRecords) ; NextStateRecords = []),
    add_list_to_heap(RPQ, NextStateRecords, NPQ),
    astar(NPQ, NV, Solution).
% last recursive case: the next cheapest state is not worth being expanded
astar(PQ, V, Solution) :-
    get_from_heap(PQ, _, _, NPQ),
    astar(NPQ, V, Solution).

%% astar_next_state(+StateRecord, +Visited, -EstimatedCost, -NewStateRecord).
astar_next_state(SR, V, F, NewStateRecord) :-
    state_record(State, _, _, Depth, SR),
    get_heuristic(Heuristic),
    NewDepth is Depth + 1,
    % progresses to NewState with Operator
    progress(State, Operator, NewState),
    % checks that NewState has not been visited yet (or already visited but with a higher cost)
    \+ member_and_higher_weight(NewState, V, NewDepth),
    compute_heuristic(Heuristic, NewState, H),
    % format('h(~p) = ~d\n', [NewState, H]),
    F is NewDepth + H,
    % write(NewState), write(' costs : '), write(F), write(' ('), write(H), write(' + '), write(NewDepth), write(')'), nl,
    state_record(NewState, State, Operator, NewDepth, NewStateRecord).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MUTATED ASTAR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mutated_astar_search(StartState, FFunction, Solution) :-
    state_record(StartState, nil, nil, 0, StateRecord),
    list_to_heap([0-StateRecord], PQ),
    mutated_astar(PQ, [], FFunction, Solution).

mutated_astar(PQ, _, _, []) :-
    empty_heap(PQ),
    !.
mutated_astar(PQ, V, _, Solution) :-
    get_from_heap(PQ, _, SR, _),
    state_record(State, _, _, _, SR),
    get_problem(Problem),
    problem_goal_state(Problem, GoalState),
    ord_subset(GoalState, State),
    % length(V, L), format('~d states visited\n', [L]),
    build_solution(SR, V, Solution).
mutated_astar(PQ, V, FFunction, Solution) :-
    get_from_heap(PQ, _, SR, RPQ),
    % #MUTATION: a visited state is never revisited
    \+ member_state_record(SR, V),
    !,
    ord_add_element(V, SR, NV),
    (bagof(F-NextStateRecord, mutated_astar_next_state(SR, NV, FFunction, F, NextStateRecord), NextStateRecords) ; NextStateRecords = []),
    add_list_to_heap(RPQ, NextStateRecords, NPQ),
    mutated_astar(NPQ, NV, FFunction, Solution).
mutated_astar(PQ, V, FFunction, Solution) :-
    get_from_heap(PQ, _, _, NPQ),
    mutated_astar(NPQ, V, FFunction, Solution).

mutated_astar_next_state(SR, V, FFunction, F, NewStateRecord) :-
    state_record(State, _, _, Depth, SR),
    get_heuristic(Heuristic),
    NewDepth is Depth + 1,
    progress(State, Operator, NewState),
    \+ state_member_state_records(NewState, V),
    compute_heuristic(Heuristic, NewState, H),
    % #MUTATION: f is not g + h anymore
    call(f(FFunction, State, NewState, NewDepth, H, F)),
    state_record(NewState, State, Operator, NewDepth, NewStateRecord).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HELPERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% state_member_state_records(+State, +StateRecords).
% checks if a state is member of a list of state records.
state_member_state_records(State, [SR|_]) :-
    state_record(State, _, _, _, SR),
    !.
state_member_state_records(State, [_|T]) :-
    state_member_state_records(State, T).

%% build_solution(+StateRecord, +Visited, -ListOfActions).
%% It builds the solution by following the previous actions done
build_solution(SR, V, L) :-
    build_solution(SR, V, [], L).

build_solution(SR, _, L, L) :-
    state_record(_, nil, nil, _, SR),
    !.
build_solution(SR, V, R, L) :-
    state_record(_, PS, Operator, _, SR),
    state_record(PS, _, _, _, Previous),
    memberchk(Previous, V),
    build_solution(Previous, V, [Operator|R], L).

%% add_list_to_heap(+OldHeap, +List, -NewHeap).
add_list_to_heap(OH, [], OH).
add_list_to_heap(OH, [K-D|T], NH) :-
    add_to_heap(OH, K, D, H),
    add_list_to_heap(H, T, NH).

member_and_higher_weight(S, [SR|_], K) :-
    state_record(S, _, _, D2, SR),
    K >= D2,
    !.
member_and_higher_weight(S, [_|T], K) :-
    member_and_higher_weight(S, T, K).

%% build_cheapest_solution(+StateRecord, +Visited, -ListOfActions).
% It is similar to build_solution. However, it considers the cheapest list of actions.
build_cheapest_solution(SR, V, L) :-
    build_cheapest_solution(SR, V, [], L).

build_cheapest_solution(SR, _, L, L) :-
    state_record(_, nil, nil, _, SR),
    !.
build_cheapest_solution(SR, V, R, L) :-
    state_record(_, PS, Operator, _, SR),
    % gets all the previous state leading to PS that have been visited...
    findall(Prev, (state_record(PS, _, _, _, Prev), member(Prev, V)), Trail),
    % ... and takes the cheapest state from them
    choose_min_prev(Trail, Min),
    % recursive call while stacking Operator in the list of the actions
    build_cheapest_solution(Min, V, [Operator|R], L).

%% choose_min_prev(+List, -Minimum).
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

member_state_record(SR, [H|_]) :-
    state_record(S, _, _, _, SR),
    state_record(S, _, _, _, H).
member_state_record(SR, [_|T]) :-
    !,
    member_state_record(SR, T).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MUTATED PRIORITY FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% f(+FFunction, +State, +NewState, +NewDepth, +H, -F).
% implements the classical function found in WA* searches: f = g + 10 * h
f(f1, _, _, Depth, H, F) :-
    !,
    F is Depth + 10 * H.
% implements what we call the best first f function: f = h
f(f2, _, _, _, H, F) :-
    !,
    F is H.
% implements what we call the worst first f function: f = -h
f(f3, _, _, _, H, F) :-
    !,
    F is -H.
% encourages the search to expand the deepest nodes (with the mitigation of h): f = -g + h
f(f4, _, _, Depth, H, F) :-
    !,
    F is -Depth + H.
% scores the discovered state with h but 'discards' it by signing it with the parity of the current depth
f(f5, _, _, Depth, H, F) :-
    !,
    % #TODO: improve arithmetics
    F is integer((-1 ** (Depth mod 2)) * H).