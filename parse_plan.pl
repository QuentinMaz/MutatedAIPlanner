:- [parseDomain].

%% deserialise_plan(+Filename, -Plan).
deserialise_plan(Filename, Plan) :- deserialise_plan(Filename, Plan, []).

%% deserialise_plan(+Filename, -Plan, -RestOfFile).
deserialise_plan(Filename, Plan, RestOfFile) :-
    read_file(Filename, List),
    ipc_style_plan(Plan, List, RestOfFile).

ipc_style_plan(Plan) --> oneOrMore(action, Plan).
action(Action) --> ['('], action_name(N), zeroOrMore(action_parameter, P), [')'], {Action =.. [N|P]}.
action_name(Name) --> name(Name).
action_parameter(Parameter) --> name(Parameter).