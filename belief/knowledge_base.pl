% ---------------------------
% 개념 정의
% ---------------------------
concept(set).
concept(element).
concept(subset).
concept(union).
concept(intersection).
concept(empty_set).

% ---------------------------
% 정의 (definition/2)
% ---------------------------
definition(element_of, "x ∈ A means x is a member of set A").
definition(empty_set, "A set with no elements, denoted ∅").
definition(subset, "A ⊆ B means every element of A is also in B").
definition(union, "A ∪ B is the set of all elements that are in A or B").
definition(intersection, "A ∩ B is the set of all elements that are in both A and B").

% ---------------------------
% 집합 예시 (fact 형태)
% ---------------------------
set(a, [1,2,3]).
set(b, [3,4]).
set(c, []).

% ---------------------------
% 규칙 (멤버십, 부분집합, 합집합, 교집합)
% ---------------------------
% 원소 관계
element_of(X, SetName) :-
    set(SetName, Elements),
    member(X, Elements).

% 부분집합
subset_of(A, B) :-
    set(A, ElementsA),
    set(B, ElementsB),
    forall(member(X, ElementsA), member(X, ElementsB)).

% 합집합
union_of(A, B, Union) :-
    set(A, ElementsA),
    set(B, ElementsB),
    append(ElementsA, ElementsB, Temp),
    sort(Temp, Union).

% 교집합
intersection_of(A, B, Intersection) :-
    set(A, ElementsA),
    set(B, ElementsB),
    findall(X, (member(X, ElementsA), member(X, ElementsB)), Temp),
    sort(Temp, Intersection).

% ---------------------------
% 예시 질의/답변 저장 (example/2)
% ---------------------------
example("Is 3 an element of set a?", "Yes, because 3 ∈ {1,2,3}").
example("Is 5 an element of set a?", "No, because 5 ∉ {1,2,3}").

example("Is set a a subset of set b?", "No, because 1 ∈ a but 1 ∉ b").
example("Is set c a subset of set a?", "Yes, ∅ ⊆ any set").

example("What is the union of sets a and b?", "{1,2,3,4}").
example("What is the intersection of sets a and b?", "{3}").
