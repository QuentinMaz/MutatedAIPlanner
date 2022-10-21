:- module(pddl_serialiser, [serialise_plan/2, write_plan/1]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SERIALISATION PREDICATE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% serialise_plan(+Plan, +Filename).
serialise_plan(Plan, Filename) :-
    open(Filename, write, FileStream),
    set_output(FileStream),
    write_plan(Plan),
    flush_output(FileStream),
    told.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% WRITE PREDICATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

write_plan([]).
write_plan([ActionDef|T]) :-
    ActionDef =.. [Action|Parameters],
    format('(~a~@)\n', [Action, write_list(Parameters)]),
    write_plan(T).

write_list([]).
write_list([H|T]) :-
    write(' '),
    write(H),
    write_list(T).