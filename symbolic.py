"""
symbolic.py — Neuro-Symbolic Reasoning Layer for MiAi

Architecture:
    User message ───► SymbolicRouter
                         │ entity extraction & intent classification
             ┌───────────┼───────────┐
             ▼           ▼           ▼
      KnowledgeGraph  RuleEngine  WorkingMemory
      (entity store,  (Horn       (conversation
       typed edges)    clause      fact stack)
                        inference)
             └───────────┴───────────┘
                         │ symbolic context
                         ▼
                  LLM (neural side)
                         │ raw response
                         ▼
                SymbolicVerifier
                  • fact-check vs KG
                  • contradiction detection
                  • extract new facts → WM / KG
                         │ grounded response
                         ▼
                       User

Six core components plus supporting utilities.
"""

from __future__ import annotations

import json
import math
import operator
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ──────────────────────────────────────────────
#  Helper: entity mention extraction
# ──────────────────────────────────────────────

def _extract_entity_mentions(text: str) -> List[str]:
    """Extract candidate entity mentions from free text.

    Returns capitalized multi-word phrases (2-4 words) and single
    significant lowercase words (nouns that are not in a small stop-list).
    """
    mentions: List[str] = []

    # Multi-word capitalized phrases (proper nouns / titles)
    multi_pattern = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', text)
    mentions.extend(multi_pattern)

    # Single significant lowercase words (len > 3, not stop words)
    stop_words = {
        'the', 'this', 'that', 'with', 'from', 'have', 'been',
        'were', 'what', 'when', 'where', 'which', 'there', 'their',
        'about', 'would', 'could', 'should', 'into', 'over', 'also',
    }
    single_words = re.findall(r'\b([a-z]{4,})\b', text.lower())
    mentions.extend(w for w in single_words if w not in stop_words)

    return list(set(mentions))


# ──────────────────────────────────────────────
#  Helper: arithmetic expression evaluation (safe)
# ──────────────────────────────────────────────

_ALLOWED_ARITHMETIC_NAMES: Dict[str, Any] = {
    'sqrt': math.sqrt,
    'abs': abs,
    'log': math.log,
    'sin': math.sin,
    'cos': math.cos,
    'pi': math.pi,
    'e': math.e,
    'floor': math.floor,
    'ceil': math.ceil,
    'pow': pow,
    'min': min,
    'max': max,
}

_ARITHMETIC_PATTERN = re.compile(
    r'^[\d\s+\-*/().,%_a-zA-Z]+$'
)


def safe_eval_arithmetic(expression: str) -> Optional[float]:
    """Safely evaluate a mathematical expression.

    Supports: +, -, *, /, **, %, (), and the functions/constants in
    _ALLOWED_ARITHMETIC_NAMES.  Returns None when the expression is
    unsafe or cannot be parsed.
    """
    expr = expression.strip()
    if not expr:
        return None

    if not _ARITHMETIC_PATTERN.match(expr):
        return None

    # Check for dangerous builtins
    dangerous = {'__', 'import', 'exec', 'eval', 'open', 'os.', 'sys.'}
    if any(d in expr for d in dangerous):
        return None

    try:
        result = eval(expr, {'__builtins__': {}}, _ALLOWED_ARITHMETIC_NAMES)
        return float(result)
    except Exception:
        return None


# ──────────────────────────────────────────────
#  Helper: entity/triple extraction from text
# ──────────────────────────────────────────────

_RELATION_ALIASES: Dict[str, str] = {
    'is a': 'is_a',
    'is an': 'is_a',
    'are a': 'is_a',
    'is not a': 'is_not_a',
    'is not an': 'is_not_a',
    'is part of': 'part_of',
    'belongs to': 'member_of',
    'is a type of': 'subclass_of',
    'is a kind of': 'subclass_of',
    'leads to': 'causes',
    'results in': 'causes',
    'prevents': 'prevents',
    'stops': 'prevents',
    'blocks': 'prevents',
    'related to': 'related_to',
    'associated with': 'related_to',
    'opposite of': 'opposite_of',
    'located in': 'located_in',
    'located at': 'located_in',
    'lives in': 'located_in',
    'works at': 'works_for',
    'employed by': 'works_for',
    'created by': 'created_by',
    'invented by': 'created_by',
    'owned by': 'owned_by',
    'has property': 'has_property',
    'has feature': 'has_feature',
    'can cause': 'causes',
    'may cause': 'causes',
    'depends on': 'depends_on',
    'requires': 'depends_on',
    'uses': 'uses',
    'contains': 'contains',
    'has': 'has_property',
}

_TRIPLE_PATTERNS = [
    re.compile(
        r'(?P<subject>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        r'\s+(?:is|are)\s+'
        r'(?P<relation>a|an|not\s+a|not\s+an|part\s+of|a\s+type\s+of|a\s+kind\s+of)'
        r'\s+(?P<object>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    ),
    re.compile(
        r'(?P<subject>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        r'\s+(?P<relation>' + '|'.join(
            re.escape(alias) for alias in sorted(
                _RELATION_ALIASES.keys(), key=len, reverse=True
            )
        ) + r')'
        r'\s+(?P<object>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    ),
]

_RELATION_LOOKUP = {v: k for k, v in _RELATION_ALIASES.items()}


def _canonicalize_relation(raw: str) -> str:
    """Map a surface-form relation string to its canonical key."""
    return _RELATION_ALIASES.get(raw.strip().lower(), raw.strip().lower().replace(' ', '_'))


def extract_triples(text: str) -> List[Tuple[str, str, str]]:
    """Lightweight heuristic triple extraction from free text.

    Returns a list of (subject, canonical_relation, object) triples.
    """
    triples: List[Tuple[str, str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()

    for pattern in _TRIPLE_PATTERNS:
        for match in pattern.finditer(text):
            subject = match.group('subject').strip()
            relation_raw = match.group('relation').strip().lower()
            obj = match.group('object').strip()

            if relation_raw in ('a', 'an'):
                relation = 'is_a'
            elif relation_raw in ('not a', 'not an'):
                relation = 'is_not_a'
            else:
                relation = _canonicalize_relation(relation_raw)

            triple = (subject, relation, obj)
            if triple not in seen:
                seen.add(triple)
                triples.append(triple)

    return triples


# ══════════════════════════════════════════════
#  Component 1: KnowledgeGraph
# ══════════════════════════════════════════════

Triple = Tuple[str, str, str]  # (subject, relation, object)


class KnowledgeGraph:
    """In-memory directed labelled graph (entity → relation → entity).

    Persisted to JSON.  Indexed three ways for O(1) lookups:
      - by subject   → {relation → {objects}}
      - by object    → {relation → {subjects}}
      - by predicate → {(subject, object)}
    """

    def __init__(self, persistence_path: Optional[str] = None) -> None:
        self._persistence_path = persistence_path

        # Forward index: subject → {relation → set[objects]}
        self._by_subject: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        # Reverse index: object → {relation → set[subjects]}
        self._by_object: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        # Predicate index: (subject, relation, object) → True
        self._by_triple: Set[Tuple[str, str, str]] = set()

        if persistence_path and os.path.exists(persistence_path):
            self._load()

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        try:
            with open(self._persistence_path, 'r') as f:
                data = json.load(f)
            for triple in data.get('triples', []):
                s, r, o = triple['subject'], triple['relation'], triple['object']
                self._add_to_indexes(s, r, o)
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    def save(self) -> None:
        if self._persistence_path is None:
            return
        triples = [
            {'subject': s, 'relation': r, 'object': o}
            for s, r, o in self._by_triple
        ]
        with open(self._persistence_path, 'w') as f:
            json.dump({'triples': triples}, f, indent=2)

    def _add_to_indexes(self, s: str, r: str, o: str) -> None:
        self._by_subject[s][r].add(o)
        self._by_object[o][r].add(s)
        self._by_triple.add((s, r, o))

    def _remove_from_indexes(self, s: str, r: str, o: str) -> None:
        if s in self._by_subject and r in self._by_subject[s]:
            self._by_subject[s][r].discard(o)
            if not self._by_subject[s][r]:
                del self._by_subject[s][r]
        if o in self._by_object and r in self._by_object[o]:
            self._by_object[o][r].discard(s)
            if not self._by_object[o][r]:
                del self._by_object[o][r]
        self._by_triple.discard((s, r, o))

    # -- mutation ----------------------------------------------------------

    def add_triple(self, subject: str, relation: str, object: str) -> None:
        self._add_to_indexes(subject, relation, object)

    def add_triples(self, triples: List[Triple]) -> None:
        for s, r, o in triples:
            self._add_to_indexes(s, r, o)

    def remove_triple(self, subject: str, relation: str, object: str) -> None:
        self._remove_from_indexes(subject, relation, object)

    # -- query -------------------------------------------------------------

    def query_by_subject(self, subject: str) -> Dict[str, Set[str]]:
        return dict(self._by_subject.get(subject, {}))

    def query_by_object(self, object: str) -> Dict[str, Set[str]]:
        return dict(self._by_object.get(object, {}))

    def query_by_relation(self, relation: str) -> List[Triple]:
        results: List[Triple] = []
        for s, r, o in self._by_triple:
            if r == relation:
                results.append((s, r, o))
        return results

    def query(self,
              subject: Optional[str] = None,
              relation: Optional[str] = None,
              object: Optional[str] = None) -> List[Triple]:
        """SPARQL-lite pattern matching.

        Any combination of subject / relation / object may be specified;
        None fields act as wildcards.
        """
        results: List[Triple] = []

        # Best-index-first dispatch
        if subject is not None and subject in self._by_subject:
            rels = self._by_subject[subject]
            for rel, objects in rels.items():
                if relation is not None and rel != relation:
                    continue
                for obj in objects:
                    if object is not None and obj != object:
                        continue
                    results.append((subject, rel, obj))
            return results

        if object is not None and object in self._by_object:
            rels = self._by_object[object]
            for rel, subjects in rels.items():
                if relation is not None and rel != relation:
                    continue
                for subj in subjects:
                    if subject is not None and subj != subject:
                        continue
                    results.append((subj, rel, object))
            return results

        # Full scan (worst case)
        for s, r, o in self._by_triple:
            if subject is not None and s != subject:
                continue
            if relation is not None and r != relation:
                continue
            if object is not None and o != object:
                continue
            results.append((s, r, o))
        return results

    def bfs_paths(self,
                  start: str,
                  goal: str,
                  max_depth: int = 5) -> List[List[Triple]]:
        """BFS from *start* entity to *goal*, returning paths as triple lists.

        Each path is a list of (subject, relation, object) steps.
        Only the first *max_depth* steps are explored.
        """
        if start not in self._by_subject:
            return []

        visited: Set[Tuple[str, str, str]] = set()
        # queue entries: (current_entity, path_so_far)
        queue: deque = deque()
        queue.append((start, []))

        results: List[List[Triple]] = []

        while queue and len(results) < 10:
            current, path = queue.popleft()
            if len(path) >= max_depth:
                continue

            rels = self._by_subject.get(current, {})
            for rel, objects in rels.items():
                for obj in objects:
                    step = (current, rel, obj)
                    if step in visited:
                        continue
                    visited.add(step)
                    new_path = path + [step]
                    if obj == goal:
                        results.append(new_path)
                    else:
                        queue.append((obj, new_path))

        return results

    def get_entities(self) -> Set[str]:
        return set(self._by_subject.keys()) | set(self._by_object.keys())

    def contains(self, subject: str, relation: str, object: str) -> bool:
        return (subject, relation, object) in self._by_triple

    def __len__(self) -> int:
        return len(self._by_triple)

    def __repr__(self) -> str:
        return f'KnowledgeGraph({len(self)} triples)'


# ══════════════════════════════════════════════
#  Component 2: RuleEngine
# ══════════════════════════════════════════════

# A rule is a dict: { 'premises': [...], 'conclusion': (...), 'name': '...' }
# Premises and conclusion use variables prefixed with '?'.
Rule = Dict[str, Any]


def _substitute(term: str, binding: Dict[str, str]) -> str:
    """Replace ?variables in *term* using *binding*."""
    for var, val in binding.items():
        term = term.replace(var, val)
    return term


def _unify(pattern: str, fact: str,
           binding: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Attempt to unify *pattern* (which may contain ?vars) with *fact*.

    Returns an updated binding dict on success, or None on failure.
    """
    # If pattern is a variable, bind it
    if pattern.startswith('?'):
        if pattern in binding:
            return binding if binding[pattern] == fact else None
        new_binding = dict(binding)
        new_binding[pattern] = fact
        return new_binding

    # Direct equality
    return binding if pattern == fact else None


def _unify_triple(pattern_triple: Tuple[str, str, str],
                  fact_triple: Tuple[str, str, str],
                  binding: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Unify a pattern triple against a fact triple."""
    ps, pr, po = pattern_triple
    fs, fr, fo = fact_triple

    b = _unify(ps, fs, binding)
    if b is None:
        return None
    b = _unify(pr, fr, b)
    if b is None:
        return None
    b = _unify(po, fo, b)
    return b


class RuleEngine:
    """Forward-chaining Horn clause inference engine.

    Rules are Python dicts with ?-prefixed variables.
    The engine saturates the fact set on every query via recursive
    backtracking unification.
    """

    BUILTIN_RULES: List[Rule] = [
        # Transitivity: A is_a B ∧ B is_a C → A is_a C
        {
            'name': 'transitivity_is_a',
            'premises': [
                ('?X', 'is_a', '?Y'),
                ('?Y', 'is_a', '?Z'),
            ],
            'conclusion': ('?X', 'is_a', '?Z'),
        },
        # Inverse: A related_to B → B related_to A
        {
            'name': 'inverse_related_to',
            'premises': [
                ('?X', 'related_to', '?Y'),
            ],
            'conclusion': ('?Y', 'related_to', '?X'),
        },
        # Sibling: A is_a Z ∧ B is_a Z → A related_to B  (if A ≠ B)
        {
            'name': 'sibling',
            'premises': [
                ('?X', 'is_a', '?Z'),
                ('?Y', 'is_a', '?Z'),
            ],
            'conclusion': ('?X', 'related_to', '?Y'),
        },
        # Grandparent: A is_a B ∧ B is_a C → A is_a C (already covered
        # by transitivity, but explicit rule is clearer)
        {
            'name': 'grandparent',
            'premises': [
                ('?A', 'is_a', '?B'),
                ('?B', 'is_a', '?C'),
                ('?C', 'is_a', '?D'),
            ],
            'conclusion': ('?A', 'is_a', '?D'),
        },
        # Inverse location: A located_in B → B contains A
        {
            'name': 'inverse_located_in',
            'premises': [
                ('?X', 'located_in', '?Y'),
            ],
            'conclusion': ('?Y', 'contains', '?X'),
        },
        # Composition: A causes B ∧ B causes C → A causes C
        {
            'name': 'causal_chain',
            'premises': [
                ('?X', 'causes', '?Y'),
                ('?Y', 'causes', '?Z'),
            ],
            'conclusion': ('?X', 'causes', '?Z'),
        },
    ]

    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        self._kg = knowledge_graph
        self._custom_rules: List[Rule] = []

    def add_rule(self, rule: Rule) -> None:
        self._custom_rules.append(rule)

    def add_rules(self, rules: List[Rule]) -> None:
        self._custom_rules.extend(rules)

    def get_all_rules(self) -> List[Rule]:
        return self.BUILTIN_RULES + self._custom_rules

    def infer(self, max_iterations: int = 10) -> int:
        """Forward-chain all rules until saturation.

        Returns the number of new facts inferred.
        """
        all_rules = self.get_all_rules()
        new_facts: Set[Triple] = set()
        existing_triples: Set[Triple] = set(self._kg._by_triple)

        for _ in range(max_iterations):
            added_this_round = 0

            for rule in all_rules:
                premises = rule['premises']
                conclusion = rule['conclusion']

                # Recursive backtracking over premises
                inferred = self._apply_rule(
                    premises, conclusion, existing_triples | new_facts
                )
                for fact in inferred:
                    if fact not in existing_triples and fact not in new_facts:
                        new_facts.add(fact)
                        added_this_round += 1

            if added_this_round == 0:
                break  # saturated

        # Commit new facts to KG
        for fact in new_facts:
            self._kg._add_to_indexes(*fact)

        return len(new_facts)

    def _apply_rule(self,
                    premises: List[Tuple[str, str, str]],
                    conclusion: Tuple[str, str, str],
                    facts: Set[Triple]) -> Set[Triple]:
        """Recursive backtracking: find all variable bindings that satisfy
        the premises, then generate the instantiated conclusion."""
        return self._backtrack(premises, 0, {}, facts, conclusion)

    def _backtrack(self,
                   premises: List[Tuple[str, str, str]],
                   idx: int,
                   binding: Dict[str, str],
                   facts: Set[Triple],
                   conclusion: Tuple[str, str, str]) -> Set[Triple]:
        if idx >= len(premises):
            # All premises satisfied — instantiate conclusion
            concluded = tuple(
                _substitute(t, binding) for t in conclusion
            )
            return {concluded}

        results: Set[Triple] = set()
        pattern = premises[idx]

        for fact in facts:
            new_binding = _unify_triple(pattern, fact, binding)
            if new_binding is not None:
                sub_results = self._backtrack(
                    premises, idx + 1, new_binding, facts, conclusion
                )
                results.update(sub_results)

                # Guard against combinatorial explosion
                if len(results) > 1000:
                    return results

        return results

    def query(self,
              subject: Optional[str] = None,
              relation: Optional[str] = None,
              object: Optional[str] = None) -> List[Triple]:
        """Query the KG with inference.  Runs forward-chaining first,
        then delegates to KnowledgeGraph.query()."""
        self.infer()
        return self._kg.query(subject, relation, object)


# ══════════════════════════════════════════════
#  Component 3: WorkingMemory
# ══════════════════════════════════════════════

@dataclass
class TurnFacts:
    """Facts extracted from a single conversational turn."""
    turn_index: int
    triples: List[Triple] = field(default_factory=list)
    raw_text: str = ''


class WorkingMemory:
    """Conversation-scoped fact stack.

    Facts extracted from each turn are asserted here; facts seen
    >= *promotion_threshold* turns are promoted to the KG.  Bounded
    to the last *max_turns* turns.
    """

    def __init__(self,
                 knowledge_graph: KnowledgeGraph,
                 max_turns: int = 10,
                 promotion_threshold: int = 3) -> None:
        self._kg = knowledge_graph
        self._max_turns = max_turns
        self._promotion_threshold = promotion_threshold
        self._turns: List[TurnFacts] = []
        self._turn_counter = 0
        # Triple → set of turn indices where it appeared
        self._fact_occurrences: Dict[Triple, Set[int]] = defaultdict(set)

    def add_facts(self, triples: List[Triple], raw_text: str = '') -> None:
        self._turn_counter += 1
        turn = TurnFacts(
            turn_index=self._turn_counter,
            triples=triples,
            raw_text=raw_text,
        )
        self._turns.append(turn)

        # Record occurrences & check for promotion
        for triple in triples:
            self._fact_occurrences[triple].add(self._turn_counter)
            if len(self._fact_occurrences[triple]) >= self._promotion_threshold:
                self._promote_to_kg(triple)

        # Enforce bounded window
        self._prune()

    def _promote_to_kg(self, triple: Triple) -> None:
        s, r, o = triple
        if not self._kg.contains(s, r, o):
            self._kg.add_triple(s, r, o)

    def _prune(self) -> None:
        """Drop turns beyond the rolling window and clean up occurrence
        tracking."""
        while len(self._turns) > self._max_turns:
            expired = self._turns.pop(0)
            for triple in expired.triples:
                self._fact_occurrences[triple].discard(expired.turn_index)
                if not self._fact_occurrences[triple]:
                    del self._fact_occurrences[triple]

    def get_recent_facts(self, n: int = 5) -> List[Triple]:
        """Return the union of facts from the last *n* turns."""
        recent: Set[Triple] = set()
        for turn in self._turns[-n:]:
            recent.update(turn.triples)
        return list(recent)

    def get_all_facts(self) -> List[Triple]:
        """Return all facts currently in working memory."""
        all_facts: Set[Triple] = set()
        for turn in self._turns:
            all_facts.update(turn.triples)
        return list(all_facts)

    def clear(self) -> None:
        self._turns.clear()
        self._fact_occurrences.clear()

    def __len__(self) -> int:
        return sum(len(t.triples) for t in self._turns)

    def __repr__(self) -> str:
        return f'WorkingMemory({len(self)} facts across {len(self._turns)} turns)'


# ══════════════════════════════════════════════
#  Component 4: SymbolicRouter
# ══════════════════════════════════════════════

Intent = str  # 'neural' | 'symbolic_aug' | 'symbolic_exec'

_SYMBOLIC_EXEC_PATTERNS = [
    re.compile(r'^what\s+is\s+-?\d+(?:\s*[+\-*/]\s*-?\d+)*', re.IGNORECASE),
    re.compile(r'^calculate\b', re.IGNORECASE),
    re.compile(r'^compute\b', re.IGNORECASE),
    re.compile(r'^solve\b', re.IGNORECASE),
    re.compile(r'^evaluate\b', re.IGNORECASE),
    re.compile(r'^simplify\b', re.IGNORECASE),
]

_NEURAL_PATTERNS = [
    re.compile(r'^(hi|hello|hey|good\s+(morning|afternoon|evening))', re.IGNORECASE),
    re.compile(r'^(how\s+are\s+you|what\'?s?\s+up)', re.IGNORECASE),
    re.compile(r'\bcould\s+you\s+(explain|elaborate|tell|describe)\b', re.IGNORECASE),
    re.compile(r'\b(write|compose|draft|create)\s+(a|an|the)\s+(poem|story|essay|letter|email)\b', re.IGNORECASE),
    re.compile(r'\bopinion\b', re.IGNORECASE),
    re.compile(r'\bfeeling\b', re.IGNORECASE),
    re.compile(r'\bthink\s+(about|of)\b', re.IGNORECASE),
]

_SYMBOLIC_AUG_PATTERNS = [
    re.compile(r'\bwhat\s+is\b', re.IGNORECASE),
    re.compile(r'\bwho\s+is\b', re.IGNORECASE),
    re.compile(r'\bhow\s+(does|do|is|are|can|would)\b', re.IGNORECASE),
    re.compile(r'\bwhere\s+(is|are|does)\b', re.IGNORECASE),
    re.compile(r'\btell\s+me\s+about\b', re.IGNORECASE),
    re.compile(r'\bexplain\b', re.IGNORECASE),
    re.compile(r'\bcompare\b', re.IGNORECASE),
    re.compile(r'\brelation(ship)?\b', re.IGNORECASE),
    re.compile(r'\bconnection\b', re.IGNORECASE),
    re.compile(r'\bdefine\b', re.IGNORECASE),
]


class SymbolicRouter:
    """Classifies each user message into an intent.

    Intent values:
      - ``neural``        — pure LLM, no symbolic augmentation
      - ``symbolic_aug``  — LLM + symbolic context injection
      - ``symbolic_exec`` — handled entirely by the symbolic engine
    """

    def classify(self, message: str) -> Intent:
        """Return the predicted intent for *message*."""
        msg = message.strip()

        if not msg:
            return 'neural'

        # Check symbolic_exec first (deterministic math)
        for pattern in _SYMBOLIC_EXEC_PATTERNS:
            if pattern.search(msg):
                return 'symbolic_exec'

        # Check neural patterns (greetings, creative, opinion)
        for pattern in _NEURAL_PATTERNS:
            if pattern.search(msg):
                return 'neural'

        # Check symbolic_aug patterns (factual questions)
        for pattern in _SYMBOLIC_AUG_PATTERNS:
            if pattern.search(msg):
                return 'symbolic_aug'

        # Default: neural for short messages, symbolic_aug for longer ones
        if len(msg.split()) <= 3:
            return 'neural'
        return 'symbolic_aug'


# ══════════════════════════════════════════════
#  Component 5: SymbolicVerifier
# ══════════════════════════════════════════════

# Pairs of relations that are contradictory
_CONTRADICTION_PAIRS: List[Tuple[str, str]] = [
    ('is_a', 'is_not_a'),
    ('causes', 'prevents'),
    ('related_to', 'opposite_of'),
]


@dataclass
class VerificationResult:
    """Result of verifying an LLM response."""
    verified_text: str
    contradictions: List[Tuple[Triple, Triple, str]] = field(default_factory=list)
    extracted_triples: List[Triple] = field(default_factory=list)
    is_grounded: bool = True


class SymbolicVerifier:
    """Post-processes LLM output.

    - Detects contradictions with the KG using paired predicates
    - Extracts new entity/relation triples via heuristics
    - Appends grounding / contradiction annotations to the response
    """

    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        self._kg = knowledge_graph

    def verify(self,
               llm_response: str,
               original_message: str) -> VerificationResult:
        """Run full verification pipeline over an LLM response."""
        # Step 1: Extract candidate triples from the response
        candidate_triples = extract_triples(llm_response)

        # Step 2: Check each against the KG for contradictions
        contradictions: List[Tuple[Triple, Triple, str]] = []
        grounded_triples: List[Triple] = []
        new_triples: List[Triple] = []

        for triple in candidate_triples:
            s, r, o = triple

            # Check for contradictions
            contradicted = self._check_contradictions(s, r, o)
            for contra_triple, pair_name in contradicted:
                contradictions.append((triple, contra_triple, pair_name))

            # Check if already in KG
            if self._kg.contains(s, r, o):
                grounded_triples.append(triple)
            else:
                new_triples.append(triple)

        # Step 3: Annotate the response
        annotations: List[str] = []

        if contradictions:
            for cand, contra, pair_name in contradictions:
                annotations.append(
                    f'[Contradiction: "{cand[0]} {cand[1]} {cand[2]}" '
                    f'contradicts "{contra[0]} {contra[1]} {contra[2]}" '
                    f'({pair_name})]'
                )

        if new_triples:
            annotations.append(
                f'[New facts extracted: '
                + ', '.join(f'"{s} {r} {o}"' for s, r, o in new_triples)
                + ']'
            )

        verified = llm_response
        if annotations:
            verified += '\n\n' + '\n'.join(annotations)

        return VerificationResult(
            verified_text=verified,
            contradictions=contradictions,
            extracted_triples=new_triples,
            is_grounded=len(contradictions) == 0,
        )

    def _check_contradictions(
        self,
        subject: str,
        relation: str,
        object: str,
    ) -> List[Tuple[Triple, str]]:
        """Check if (subject, relation, object) contradicts any KG fact.

        Returns list of (contradicting_triple, pair_name).
        """
        results: List[Tuple[Triple, str]] = []

        for rel_a, rel_b in _CONTRADICTION_PAIRS:
            if relation == rel_a:
                contra_relation = rel_b
            elif relation == rel_b:
                contra_relation = rel_a
            else:
                continue

            pair_name = f'{rel_a} / {rel_b}'
            # Check exact existence of the contradictory triple
            if self._kg.contains(subject, contra_relation, object):
                results.append(((subject, contra_relation, object), pair_name))

            # Also check via query (in case inference added it)
            matches = self._kg.query(subject=subject, relation=contra_relation, object=object)
            for match in matches:
                results.append((match, pair_name))

        return results


# ══════════════════════════════════════════════
#  Component 6: NeuralSymbolicBridge
# ══════════════════════════════════════════════

@dataclass
class PreProcessResult:
    """Output of NeuralSymbolicBridge.pre_process()."""
    symbolic_context: Optional[str] = None
    intent: Intent = 'neural'
    symbolic_answer: Optional[str] = None
    extracted_entities: List[str] = field(default_factory=list)
    extracted_triples: List[Triple] = field(default_factory=list)


@dataclass
class PostProcessResult:
    """Output of NeuralSymbolicBridge.post_process()."""
    verified_response: str
    new_triples: List[Triple] = field(default_factory=list)
    contradictions_found: int = 0
    is_grounded: bool = True


class NeuralSymbolicBridge:
    """Top-level orchestrator for the neuro-symbolic layer.

    Usage::

        bridge = NeuralSymbolicBridge()
        pre = bridge.pre_process(user_message)
        if pre.symbolic_answer:
            reply = pre.symbolic_answer
        else:
            # Inject pre.symbolic_context into LLM prompt
            llm_raw = llm_chat(context=pre.symbolic_context, ...)
            post = bridge.post_process(llm_raw, user_message)
            reply = post.verified_response
    """

    def __init__(self,
                 knowledge_graph: Optional[KnowledgeGraph] = None,
                 rule_engine: Optional[RuleEngine] = None,
                 working_memory: Optional[WorkingMemory] = None,
                 router: Optional[SymbolicRouter] = None,
                 verifier: Optional[SymbolicVerifier] = None,
                 persistence_path: Optional[str] = None) -> None:
        self._kg = knowledge_graph or KnowledgeGraph(
            persistence_path=persistence_path or 'kg_store.json'
        )
        self._rule_engine = rule_engine or RuleEngine(self._kg)
        self._working_memory = working_memory or WorkingMemory(self._kg)
        self._router = router or SymbolicRouter()
        self._verifier = verifier or SymbolicVerifier(self._kg)

    # -- public API --------------------------------------------------------

    def pre_process(self, message: str) -> PreProcessResult:
        """Analyse *message* and return symbolic context or a fully-
        symbolic answer."""
        intent = self._router.classify(message)
        entities = _extract_entity_mentions(message)
        triples = extract_triples(message)

        result = PreProcessResult(
            intent=intent,
            extracted_entities=entities,
            extracted_triples=triples,
        )

        if intent == 'symbolic_exec':
            answer = self._handle_symbolic_exec(message)
            result.symbolic_answer = answer
            return result

        if intent == 'symbolic_aug':
            context = self._build_symbolic_context(entities, triples, message)
            result.symbolic_context = context
            return result

        # neural — no symbolic augmentation
        return result

    def post_process(self,
                     llm_response: str,
                     original_message: str) -> PostProcessResult:
        """Verify *llm_response* and extract new facts."""
        # Extract and persist new triples from the original message
        msg_triples = extract_triples(original_message)
        if msg_triples:
            self._working_memory.add_facts(msg_triples, original_message)

        # Run the verifier
        verification = self._verifier.verify(llm_response, original_message)

        # Persist newly discovered triples into working memory
        if verification.extracted_triples:
            self._working_memory.add_facts(
                verification.extracted_triples, llm_response
            )

        # Run inference to materialise derived facts
        self._rule_engine.infer()

        # Persist KG
        self._kg.save()

        return PostProcessResult(
            verified_response=verification.verified_text,
            new_triples=verification.extracted_triples,
            contradictions_found=len(verification.contradictions),
            is_grounded=verification.is_grounded,
        )

    # -- internals ---------------------------------------------------------

    def _handle_symbolic_exec(self, message: str) -> str:
        """Handle a fully-symbolic (mathematical) request."""
        # Try to extract an arithmetic expression
        expr_match = re.search(
            r'-?\d+(?:\s*[+\-*/().,%]\s*-?\d+)*(?:\.\d+)?', message
        )
        if expr_match:
            expr = expr_match.group(0).strip()
            result = safe_eval_arithmetic(expr)
            if result is not None:
                # Round to a reasonable precision
                if result == int(result):
                    return str(int(result))
                return f'{result:.6f}'.rstrip('0').rstrip('.')

        # KG query: "what is X?"
        what_match = re.match(
            r'what\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', message, re.IGNORECASE
        )
        if what_match:
            entity = what_match.group(1).strip()
            kg_info = self._kg.query_by_subject(entity)
            if kg_info:
                parts = []
                for rel, objects in kg_info.items():
                    for obj in objects:
                        parts.append(f'{entity} {rel} {obj}')
                return '\n'.join(parts)
            return f'I don\'t have information about "{entity}" in my knowledge graph.'

        return 'I cannot process that request symbolically.'

    def _build_symbolic_context(self,
                                 entities: List[str],
                                 triples: List[Triple],
                                 message: str) -> str:
        """Build a symbolic context string for LLM prompt injection."""
        parts: List[str] = []

        # Run inference
        self._rule_engine.infer()

        # KG facts matching extracted entities
        for entity in entities:
            subj_facts = self._kg.query_by_subject(entity)
            obj_facts = self._kg.query_by_object(entity)
            if subj_facts:
                for rel, objects in subj_facts.items():
                    for obj in objects:
                        parts.append(f'[KG] {entity} {rel} {obj}')
            if obj_facts:
                for rel, subjects in obj_facts.items():
                    for subj in subjects:
                        parts.append(f'[KG] {subj} {rel} {entity}')

        # Include inferred triples that match mentioned entities
        for s, r, o in triples:
            inferred = self._kg.query(subject=s, relation=r)
            for isub, irel, iobj in inferred:
                if (isub, irel, iobj) != (s, r, o):
                    parts.append(f'[INF] {isub} {irel} {iobj}')

        # Working memory — recent facts
        recent = self._working_memory.get_recent_facts(5)
        for triple in recent:
            s, r, o = triple
            parts.append(f'[WM] {s} {r} {o}')

        if not parts:
            return ''

        return (
            '### Symbolic Context\n'
            'The following facts are known from the knowledge graph, '
            'inference engine, and working memory:\n'
            + '\n'.join(parts)
        )

    # -- accessors ---------------------------------------------------------

    @property
    def knowledge_graph(self) -> KnowledgeGraph:
        return self._kg

    @property
    def rule_engine(self) -> RuleEngine:
        return self._rule_engine

    @property
    def working_memory(self) -> WorkingMemory:
        return self._working_memory

    @property
    def router(self) -> SymbolicRouter:
        return self._router

    @property
    def verifier(self) -> SymbolicVerifier:
        return self._verifier
