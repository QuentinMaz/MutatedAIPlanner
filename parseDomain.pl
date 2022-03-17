%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  parseDomain.pl
%%  Simple parser of PDDL domain file into prolog syntax.
%%  Author: Robert Sasak, Charles University in Prague
%%
%%  Example:
%%  ?-parseDomain('blocks_world.pddl', O).
%%      O = domain(blocks,
%%          [strips, typing, 'action-costs'],
%%          [block],
%%          _G4108,
%%          [on(block(?x), block(?y)), ontable(block(?x)),
%%	         clear(block(?x)), handempty, holding(block(?x))],
%%          [number(f('total-cost', []))],
%%          _G4108,
%%          [action('pick-up', [block(?x)],       %parameters
%%		        [clear(?x), ontable(?x), handempty], %preconditions
%%              [holding(?x)],                       %positiv effects
%%          [ontable(?x), clear(?x), handempty], %negativ effects
%%          [increase('total-cost', 2)]),        %numeric effects
%%          ...],
%%      ...)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% parseDomain(+File, -Output).
% parses PDDL domain File and turns it into re-written Prolog syntax
parse_domain(F, O) :- parseDomain(F, O, []).

%% parseDomain(+File, -Output, -RestOfFile).
% same as parseDomain/2 but also returns the rest of the file. It can be useful if the domain and problem are in one file
parseDomain(File, Domain, R) :-
    read_file(File, List),
    domainBNF(Output, List, R),
    Output =.. [domain|Data],
    (
        foreach(Term, Data),
        foreach(GroundTerm, GroundData)
    do
        (var(Term) -> GroundTerm = [] ; GroundTerm = Term)
    ),
    Domain =.. [domain|GroundData].

:- [readFile]. % loads dependancies

% defines operator ?. It is a syntax sugar to mark PDDL variables (example : ?x)
:- op(300, fy, ?).

% List of DCG rules describing structure of domain file in language PDDL.
% BNF description was obtain from http://www.cs.yale.edu/homes/dvm/papers/pddl-bnf.pdf
% This parser do not fully NOT support PDDL 3.0
% However you will find comment out lines ready for futher development.

domainBNF(domain(N, R, T, C, P, [], [], S))
    --> ['(', 'define', '(', 'domain'], name(N), [')'],
        (require_def(R)	; []),
        (types_def(T) ; []),
        (constants_def(C) ; []),
        (predicates_def(P) ; []),
        % (functions_def(F) ; []), % :fluents
        % (constraints(C) ; []), % :constraints
        zeroOrMore(structure_def, S),
        [')'].

require_def(R) --> ['(', ':', 'requirements'], oneOrMore(require_key, R), [')'].
require_key(strips) --> [':', strips].
require_key(typing) --> [':', typing].
% require_key('negative-preconditions') --> [':', 'negative-preconditions'].
% require_key('disjunctive-preconditions') --> [':', 'disjunctive-preconditions'].
require_key(equality) --> [':', 'equality'].
require_key('existential-preconditions') --> [':', 'existential-preconditions'].
require_key('universal-preconditions') --> [':', 'universal-preconditions'].
require_key('quantified-preconditions')	--> [':', 'quantified-preconditions'].
require_key('conditional-effects') --> [':', 'conditional-effects'].
require_key(fluents) --> [':', 'fluents'].
require_key(adl) --> [':', 'adl'].
require_key('durative-actions') --> [':', 'durative-actions'].
require_key('derived-predicates') --> [':', 'derived-predicates'].
require_key('timed-initial-literals') --> [':', 'timed-initial-literals'].
require_key(preferences) --> [':', 'preferences'].
require_key(constraints) --> [':', 'constraints'].

types_def(L) --> ['(', ':', types], typed_list(name, L), [')'].

constants_def(L) --> ['(', ':', constants], typed_list(name, L), [')'].

predicates_def(P) --> ['(', ':', predicates], oneOrMore(atomic_formula_skeleton, P), [')'].

atomic_formula_skeleton(F) --> ['('], predicate(P), typed_list(variable, L), [')'], {F =.. [P|L]}.

predicate(P) --> name(P).

variable(V) --> ['?'], name(N), {V =.. [?, N]}.

atomic_function_skeleton(f(S, L)) --> ['('], function_symbol(S), typed_list(variable, L), [')'].

function_symbol(S) --> name(S).

functions_def(F) --> ['(', ':', functions], function_typed_list(atomic_function_skeleton, F), [')']. %:fluents

% constraints(C) --> ['(', ':', constraints], con_GD(C), [')']. % :constraints

structure_def(A) --> action_def(A).
% structure_def(D) --> durative_action_def(D). % :durative actions
% structure_def(D) --> derived_def(D). % :derived predicates

typed_list(W, R) --> oneOrMore(W, N), ['-'], type(T), !, typed_list(W, Ns), {type_list(T, N, TN), append(TN, Ns, R)}.
typed_list(W, N) --> zeroOrMore(W, N).

primitive_type(N) --> name(N).

type(either(PT)) --> ['(', either], !, oneOrMore(primitive_type, PT), [')'].
type(PT) --> primitive_type(PT).

function_typed_list(W, [F|Ls]) --> oneOrMore(W, L), ['-'], function_type(T), !, function_typed_list(W, Ls), {F =.. [T|L]}. % :typing
function_typed_list(W, L) --> zeroOrMore(W, L).

function_type(number) --> [number].

emptyOr(_) --> ['(', ')'].
emptyOr(W) --> W.

%% action_def(-Action).
action_def(action(S, L, Precon, Pos, Neg, Assign))
    --> ['(', ':', action], action_symbol(S),
        [':', parameters, '('], typed_list(variable, L), [')'],
        action_def_body(Precon, Pos, Neg, Assign),
        [')'].

action_symbol(N) --> name(N).

action_def_body(P, Pos, Neg, Assign) --> [':', precondition], pre_GD(P), [':', effect], emptyOr(effect(Pos, Neg, Assign)).
action_def_body([], Pos, Neg, Assign) --> ([':', precondition, '(', ')'] ; []), [':', effect], emptyOr(effect(Pos, Neg, Assign)).

pre_GD([F]) --> atomic_formula(term, F), !.
pre_GD(P) --> pref_GD(P).
pre_GD(P) --> ['(', and], pre_GD(P), [')'].
% pre_GD(forall(L, P)) --> ['(', forall, '('], typed_list(variable, L), [')'], pre_GD(P), [')']. % :universal-preconditions

% pref_GD(preference(N, P))	--> ['(', preference], (pref_name(N) ; []), gd(P), [')']. % :preferences
pref_GD(P) --> gd(P).

pref_name(N) --> name(N).

gd(F) --> atomic_formula(term, F).	% this option is covered by gd(L)
% gd(L) --> literal(term, L). % :negative-preconditions
gd(P) --> ['(', and], zeroOrMore(gd, P), [')'].
% gd(or(P)) --> ['(', or], zeroOrMore(gd, P), [')']. % :disjuctive-preconditions
% gd(not(P)) --> ['(', not], gd(P), [')']. % :disjuctive-preconditions
% gd(imply(P1, P2)) --> ['(', imply], gd(P1), gd(P2), [')']. % :disjuctive-preconditions
% gd(exists(L, P)) --> ['(', exists, '('], typed_list(variable, L), [')'], gd(P), [')']. % :existential-preconditions
% gd(forall(L, P)) --> ['(', forall, '('], typed_list(variable, L), [')'], gd(P), [')']. % :universal-preconditions
gd(F) --> f_comp(F). % :fluents

f_comp(compare(C, E1, E2)) --> ['('], binary_comp(C), f_exp(E1), f_exp(E2), [')'].

literal(T, F) --> atomic_formula(T, F).
literal(T, not(F)) --> ['(', not], atomic_formula(T, F), [')'].

atomic_formula(_, F) --> ['('], predicate(P), zeroOrMore(term, T), [')'], {F =.. [P|T]}.

term(N)	--> name(N).
term(V)	--> variable(V).

f_exp(N) --> number(N).
f_exp(op(O, E1, E2)) --> ['('], binary_op(O), f_exp(E1), f_exp(E2), [')'].
f_exp('-'(E)) --> ['(', '-'], f_exp(E), [')'].
f_exp(H) --> f_head(H).

f_head(F) --> ['('], function_symbol(S), zeroOrMore(term, T), [')'], {F =.. [S|T]}.
f_head(S) --> function_symbol(S).

binary_op(O) --> multi_op(O).
binary_op(45) --> [45]. % 45 = minus = '-'
binary_op('/') --> ['/'].

multi_op('*') --> ['*'].
multi_op('+') --> ['+'].

binary_comp('>') --> ['>'].
binary_comp('<') --> ['<'].
binary_comp('=') --> ['='].
binary_comp('>=') --> ['>='].
binary_comp('<=') --> ['<='].

number(N) --> [N], {integer(N)}.
number(N) --> [N], {float(N)}.

effect(P, N, A) --> ['(', and], c_effect(P, N, A), [')'].
effect(P, N, A) --> c_effect(P, N, A).

c_effect(P, N, A) --> p_effect(P, N, A).

p_effect([], [], []) --> [].
p_effect(Ps, Ns, [F|As])
    --> ['('], assign_op(O), f_head(H), f_exp(E), [')'],
        p_effect(Ps, Ns, As), {F =.. [O, H, E]}.
p_effect(Ps, [F|Ns], As) --> ['(', not], atomic_formula(term, F), [')'], p_effect(Ps, Ns, As).
p_effect([F|Ps], Ns, As) --> atomic_formula(term, F), p_effect(Ps, Ns, As).

assign_op(assign) --> [assign].
assign_op(scale_up) --> [scale_up].
assign_op(scale_down) --> [scale_down].
assign_op(increase) --> [increase].
assign_op(decrease) --> [decrease].

%% oneOrMore(+Word, -Results, +Input, -Output).
% DCG extension that overcomes the <term>+ operator (cf the BNF description)
% mimics the behavior of the predicate 'C'/3 (which is part of the underlying DCG processing)
oneOrMore(Word, [R|Rs], Input, Output) :-
    F =.. [Word, R, Input, TmpOutput], % builds a functor with R and Input as first variables
    F, % calls the predicate
    (oneOrMore(Word, Rs, TmpOutput, Output) ; (Rs = [] , Output = TmpOutput)).

%% zeroOrMore(+Word, -Rest).
% DCG extension that overcomes the <term>* operator (cf the BNF description)
zeroOrMore(W, R) --> oneOrMore(W, R).
zeroOrMore(_, []) --> [].

%% type_list(+Type, +List, -TypedList).
type_list(_, [], []).
type_list(Type, [Head|T1], [TypedHead|T2]) :-
    TypedHead =.. [Type, Head],
    type_list(Type, T1, T2).

%% name(+Name).
% Name is everything that is not number, bracket or question mark.
% Those rules are not necessary, but rapidly speed up parsing process.
name(N) --> [N], {integer(N), !, fail}.
name(N) --> [N], {float(N), !, fail}.
name(N)	--> [N], {N = ')', !, fail}.
name(N)	--> [N], {N = '(', !, fail}.
name(N)	--> [N], {N = '?', !, fail}.
name(N)	--> [N], {N = '-', !, fail}.
name(N)	--> [N].