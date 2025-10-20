# world_model/__init__.py
"""
World Model - module unique, optimisé et auto-contenu.
Contient :
  - PhysicsEngine        : dynamique physique simplifiée, contraintes et simulations
  - SocialModel          : agents, normes, intentions, relations
  - TemporalReasoning    : temps, causalité, échéances, fenêtres temporelles
  - SpatialReasoning     : repères, cartes mentales, graphes spatiaux, itinéraires

Objectifs :
  - Aucun import de sous-fichiers (évite ModuleNotFoundError)
  - Compatibilité avec core.cognitive_architecture (from world_model import PhysicsEngine)
  - to_state()/from_state() pour persistance (core/persistence.py)
  - Auto-wiring doux via cognitive_architecture (getattr, sans import croisé)

Dépendances : standard library only.
"""

from __future__ import annotations
import math
import time
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 🧭 Utilitaires
# ============================================================

def _now() -> float:
    return time.time()

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _mean(xs: List[float], default: float = 0.0) -> float:
    return sum(xs) / len(xs) if xs else default


# ============================================================
# 🎯 Exploration & apprentissage en ligne
# ============================================================


class StructuredRewardBus:
    """Collecte centralisée de récompenses structurées.

    Permet de partager les signaux entre sous-modules (social, physique, planification)
    et, si présent, avec d'autres composants de l'architecture (mémoire, méta-contrôle).
    """

    _KNOWN_CATEGORIES = {
        "social_prediction",
        "physics_stability",
        "planning_utility",
    }

    def __init__(self):
        self.history: Dict[str, List[Tuple[float, float]]] = {
            key: [] for key in self._KNOWN_CATEGORIES
        }

    def emit(self, category: str, value: float, timestamp: Optional[float] = None):
        cat = category if category in self.history else None
        if cat is None:
            # on autorise des catégories ad-hoc tout en conservant la trace
            self.history.setdefault(category, [])
            cat = category
        ts = _now() if timestamp is None else float(timestamp)
        self.history[cat].append((ts, float(value)))
        if len(self.history[cat]) > 500:
            self.history[cat] = self.history[cat][-500:]

    def summary(self, category: str, window: Optional[int] = None) -> float:
        data = self.history.get(category)
        if not data:
            return 0.0
        if window is not None and window > 0:
            data = data[-window:]
        return _mean([val for _, val in data], 0.0)

    def to_state(self) -> Dict[str, Any]:
        return {
            cat: list(entries)
            for cat, entries in self.history.items()
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "StructuredRewardBus":
        obj = cls()
        for cat, entries in state.items():
            obj.history[cat] = [
                (float(ts), float(val)) for ts, val in entries if isinstance(ts, (int, float))
            ][-500:]
        return obj


class DiscreteThompsonSampler:
    """Bandit à échantillonnage de Thompson pour ensembles finis."""

    def __init__(
        self,
        choices: Iterable[Tuple[str, float]] | Iterable[float] | Mapping[str, float],
        prior_success: float = 1.0,
        prior_failure: float = 1.0,
    ):
        tmp: List[Tuple[str, float]] = []
        if isinstance(choices, Mapping):
            tmp = [(str(k), float(v)) for k, v in choices.items()]
        else:
            for idx, value in enumerate(choices):
                if isinstance(value, tuple) and len(value) == 2:
                    key, val = value
                else:
                    key, val = str(idx), float(value)  # type: ignore[arg-type]
                tmp.append((str(key), float(val)))
        if not tmp:
            raise ValueError("DiscreteThompsonSampler requires at least one choice")
        self.choices: List[Tuple[str, float]] = tmp

        self.success: Dict[str, float] = {
            key: float(prior_success) for key, _ in self.choices
        }
        self.failure: Dict[str, float] = {
            key: float(prior_failure) for key, _ in self.choices
        }
        self._last_key: Optional[str] = None

    def sample(self) -> str:
        best_key: Optional[str] = None
        best_score = -1.0
        for key, _ in self.choices:
            s = max(1e-3, self.success.get(key, 1.0))
            f = max(1e-3, self.failure.get(key, 1.0))
            score = random.betavariate(s, f)
            if score > best_score:
                best_key = key
                best_score = score
        assert best_key is not None
        self._last_key = best_key
        return best_key

    def sample_value(self) -> Tuple[float, str]:
        key = self.sample()
        value = next(val for k, val in self.choices if k == key)
        return value, key

    def update(self, key: Optional[str], reward: float):
        if key is None:
            key = self._last_key
        if key is None:
            return
        reward = _clamp(float(reward), 0.0, 1.0)
        self.success[key] = self.success.get(key, 1.0) + reward
        self.failure[key] = self.failure.get(key, 1.0) + (1.0 - reward)

    def to_state(self) -> Dict[str, Any]:
        return {
            "choices": list(self.choices),
            "success": dict(self.success),
            "failure": dict(self.failure),
            "last": self._last_key,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "DiscreteThompsonSampler":
        obj = cls(state.get("choices", []))
        obj.success.update({k: float(v) for k, v in state.get("success", {}).items()})
        obj.failure.update({k: float(v) for k, v in state.get("failure", {}).items()})
        obj._last_key = state.get("last")
        return obj


class BoundedOnlineLinear:
    """Régression logistique online à poids bornés."""

    def __init__(
        self,
        dim: int,
        base_lr: float = 0.1,
        bounds: Optional[List[Tuple[float, float]]] = None,
        max_step: float = 0.25,
    ):
        self.weights: List[float] = [0.0] * dim
        self.base_lr = float(base_lr)
        self.bounds = bounds or [(-1.0, 1.0)] * dim
        self.max_step = float(max_step)

    def predict(self, features: List[float]) -> float:
        z = sum(w * x for w, x in zip(self.weights, features))
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, features: List[float], target: float, lr: Optional[float] = None) -> Tuple[float, float]:
        target = _clamp(target, 0.0, 1.0)
        lr = self.base_lr if lr is None else float(lr)
        prediction = self.predict(features)
        error = target - prediction
        for i, (w, x) in enumerate(zip(self.weights, features)):
            step = _clamp(lr * error * x, -self.max_step, self.max_step)
            lo, hi = self.bounds[i] if i < len(self.bounds) else (-1.0, 1.0)
            self.weights[i] = _clamp(w + step, lo, hi)
        updated = self.predict(features)
        return prediction, updated

    def to_state(self) -> Dict[str, Any]:
        return {
            "weights": list(self.weights),
            "base_lr": self.base_lr,
            "bounds": list(self.bounds),
            "max_step": self.max_step,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "BoundedOnlineLinear":
        obj = cls(
            dim=len(state.get("weights", [])) or 1,
            base_lr=float(state.get("base_lr", 0.1)),
            bounds=[tuple(map(float, b)) for b in state.get("bounds", [])],
            max_step=float(state.get("max_step", 0.25)),
        )
        obj.weights = [float(w) for w in state.get("weights", obj.weights)]
        return obj

# ============================================================
# 🧩 SpatialReasoning
# ============================================================

@dataclass
class Node2D:
    id: str
    x: float
    y: float
    tags: List[str] = field(default_factory=list)


@dataclass
class Edge:
    a: str
    b: str
    cost: float


class SpatialReasoning:
    """
    Graphe spatial minimal : nœuds 2D + arêtes pondérées.
    - Construction de carte : add_node / add_edge
    - Itinéraire : shortest_path (Dijkstra simple)
    - Proximité : nearest_nodes
    - Repères : find_by_tag
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_architecture = cognitive_architecture
        self.nodes: Dict[str, Node2D] = {}
        self.edges: Dict[str, List[Edge]] = {}
        self.landmarks: List[str] = []

    # ----- Carte & repères -----

    def add_node(self, node_id: str, x: float, y: float, tags: Optional[List[str]] = None):
        self.nodes[node_id] = Node2D(node_id, float(x), float(y), tags or [])
        if "landmark" in (tags or []):
            self.landmarks.append(node_id)

    def add_edge(self, a: str, b: str, cost: Optional[float] = None, bidirectional: bool = True):
        if a not in self.nodes or b not in self.nodes:
            return
        cost = float(cost) if cost is not None else self._euclid(a, b)
        self.edges.setdefault(a, []).append(Edge(a, b, cost))
        if bidirectional:
            self.edges.setdefault(b, []).append(Edge(b, a, cost))

    def _euclid(self, a: str, b: str) -> float:
        na, nb = self.nodes[a], self.nodes[b]
        return math.hypot(na.x - nb.x, na.y - nb.y)

    # ----- Requêtes -----

    def nearest_nodes(self, x: float, y: float, k: int = 3) -> List[Tuple[str, float]]:
        pairs = [(nid, math.hypot(n.x - x, n.y - y)) for nid, n in self.nodes.items()]
        pairs.sort(key=lambda t: t[1])
        return pairs[:max(0, k)]

    def find_by_tag(self, tag: str, k: int = 5) -> List[str]:
        return [nid for nid, n in self.nodes.items() if tag in n.tags][:k]

    # ----- Dijkstra minimal -----

    def shortest_path(self, start: str, goal: str) -> Tuple[float, List[str]]:
        if start not in self.nodes or goal not in self.nodes:
            return float("inf"), []
        dist: Dict[str, float] = {start: 0.0}
        prev: Dict[str, Optional[str]] = {start: None}
        visited: set = set()
        frontier: List[Tuple[float, str]] = [(0.0, start)]
        import heapq
        while frontier:
            d, u = heapq.heappop(frontier)
            if u in visited:
                continue
            visited.add(u)
            if u == goal:
                break
            for e in self.edges.get(u, []):
                v = e.b
                nd = d + e.cost
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(frontier, (nd, v))
        if goal not in dist:
            return float("inf"), []
        # reconstruct
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return dist[goal], path

    # ----- Persistance -----

    def to_state(self) -> Dict[str, Any]:
        return {
            "nodes": {nid: {"x": n.x, "y": n.y, "tags": list(n.tags)} for nid, n in self.nodes.items()},
            "edges": {nid: [{"a": e.a, "b": e.b, "cost": e.cost} for e in lst] for nid, lst in self.edges.items()},
            "landmarks": list(self.landmarks),
        }

    def from_state(self, state: Dict[str, Any]):
        self.nodes = {}
        for nid, d in state.get("nodes", {}).items():
            self.nodes[nid] = Node2D(nid, float(d["x"]), float(d["y"]), list(d.get("tags", [])))
        self.edges = {}
        for nid, lst in state.get("edges", {}).items():
            self.edges[nid] = [Edge(x["a"], x["b"], float(x["cost"])) for x in lst]
        self.landmarks = list(state.get("landmarks", []))


# ============================================================
# ⏱️ TemporalReasoning
# ============================================================

@dataclass
class TimeWindow:
    start: float
    end: float
    label: str = ""


class TemporalReasoning:
    """
    Représentation simple du temps :
    - Fenêtres temporelles nommées
    - Délais/échéances (deadline pressure)
    - Raisonnement causal minimal (A avant B, délai, expiration)
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_architecture = cognitive_architecture
        self.reward_bus: Optional[StructuredRewardBus] = None
        if cognitive_architecture is not None:
            self.reward_bus = getattr(cognitive_architecture, "reward_bus", None)
        self.windows: List[TimeWindow] = []
        self.deadlines: Dict[str, float] = {}  # label -> timestamp
        self.causes: List[Tuple[str, str]] = []  # (cause, effect)
        self.ttl_choices: List[float] = [3.0, 7.0, 14.0, 30.0]
        self.ttl_unit: float = 3600.0  # interprétation : heures
        self.ttl_bandits: Dict[str, DiscreteThompsonSampler] = {}
        self.ttl_assignments: Dict[str, Tuple[float, float]] = {}
        self.ttl_hazard: Dict[str, BoundedOnlineLinear] = {}

    def add_window(self, start: float, end: float, label: str):
        self.windows.append(TimeWindow(float(start), float(end), label))

    def set_deadline(
        self,
        label: str,
        timestamp: Optional[float] = None,
        item_type: Optional[str] = None,
        now: Optional[float] = None,
    ):
        if timestamp is not None:
            self.deadlines[label] = float(timestamp)
            return
        if item_type is None:
            self.deadlines[label] = _now() if now is None else float(now)
            return
        ttl = self.assign_ttl(item_type, now=now)
        anchor = _now() if now is None else float(now)
        self.deadlines[label] = anchor + ttl * self.ttl_unit

    def add_causal_link(self, cause: str, effect: str):
        self.causes.append((cause, effect))

    # ----- requêtes -----

    def is_within(self, t: float, label: Optional[str] = None) -> bool:
        for w in self.windows:
            if (label is None or w.label == label) and (w.start <= t <= w.end):
                return True
        return False

    def deadline_pressure(self, label: str, now: Optional[float] = None) -> float:
        now = _now() if now is None else float(now)
        ts = self.deadlines.get(label)
        if ts is None:
            return 0.0
        remaining = ts - now
        if remaining <= 0:
            return 1.0
        span = max(ts - min((w.start for w in self.windows), default=now), 1.0)
        return 1.0 - _clamp(remaining / span, 0.0, 1.0)

    def can_happen_after(self, a: str, b: str) -> bool:
        # si (a -> b) causalement, b après a
        return (a, b) in self.causes

    # ----- TTL adaptatif -----

    def assign_ttl(self, item_type: str, now: Optional[float] = None) -> float:
        bandit = self.ttl_bandits.get(item_type)
        if bandit is None:
            bandit = DiscreteThompsonSampler([(str(int(c)), float(c)) for c in self.ttl_choices])
            mid_idx = len(self.ttl_choices) // 2
            mid_key = str(int(self.ttl_choices[mid_idx]))
            if mid_key in bandit.success:
                bandit.success[mid_key] += 1.0
            self.ttl_bandits[item_type] = bandit
        ttl, _ = bandit.sample_value()
        anchor = _now() if now is None else float(now)
        self.ttl_assignments[item_type] = (anchor, ttl)
        return ttl

    def report_ttl_outcome(self, item_type: str, reused: bool, now: Optional[float] = None):
        bandit = self.ttl_bandits.get(item_type)
        anchor, ttl = self.ttl_assignments.get(item_type, (None, None))
        current_time = _now() if now is None else float(now)
        elapsed_ratio = 1.0
        if anchor is not None and ttl and self.ttl_unit > 0:
            elapsed = max(0.0, current_time - anchor)
            elapsed_ratio = elapsed / (ttl * self.ttl_unit)
        if bandit is not None:
            if reused:
                reward = 0.6 + 0.4 * max(0.0, 1.0 - min(1.5, elapsed_ratio))
            else:
                reward = 0.2 * min(1.0, elapsed_ratio)
            bandit.update(None, reward)

        hazard = self.ttl_hazard.get(item_type)
        if hazard is None:
            hazard = BoundedOnlineLinear(
                dim=2,
                base_lr=0.05,
                bounds=[(-4.0, 4.0), (-4.0, 4.0)],
                max_step=0.4,
            )
            self.ttl_hazard[item_type] = hazard
        features = [1.0, min(5.0, max(0.0, elapsed_ratio))]
        hazard.update(features, 1.0 if reused else 0.0)
        if self.reward_bus is not None:
            alignment = 1.0 - min(1.0, abs(elapsed_ratio - 1.0))
            planning_reward = 0.5 * alignment + (0.5 if reused else 0.0)
            self.reward_bus.emit("planning_utility", planning_reward)

    def hazard_probability(self, item_type: str, elapsed_ratio: float) -> float:
        hazard = self.ttl_hazard.get(item_type)
        if hazard is None:
            return 0.5
        features = [1.0, min(5.0, max(0.0, elapsed_ratio))]
        return hazard.predict(features)

    def ttl_policy_snapshot(self) -> Dict[str, Any]:
        """Expose une vue compacte des TTL pour synchroniser la mémoire externe."""

        return {
            "choices": list(self.ttl_choices),
            "unit": self.ttl_unit,
            "assignments": {
                item: {
                    "anchor": anchor,
                    "ttl": ttl,
                }
                for item, (anchor, ttl) in self.ttl_assignments.items()
            },
            "hazard_weights": {
                item: list(model.weights)
                for item, model in self.ttl_hazard.items()
            },
        }

    # ----- Persistance -----

    def to_state(self) -> Dict[str, Any]:
        return {
            "windows": [{"start": w.start, "end": w.end, "label": w.label} for w in self.windows],
            "deadlines": dict(self.deadlines),
            "causes": list(self.causes),
            "ttl": {
                "choices": list(self.ttl_choices),
                "unit": self.ttl_unit,
                "bandits": {k: v.to_state() for k, v in self.ttl_bandits.items()},
                "assignments": {k: (float(a), float(t)) for k, (a, t) in self.ttl_assignments.items()},
                "hazard": {k: v.to_state() for k, v in self.ttl_hazard.items()},
            },
        }

    def from_state(self, state: Dict[str, Any]):
        self.windows = [TimeWindow(float(w.get("start", 0.0)), float(w.get("end", 0.0)), w.get("label", ""))
                        for w in state.get("windows", [])]
        self.deadlines = {k: float(v) for k, v in state.get("deadlines", {}).items()}
        self.causes = [tuple(x) for x in state.get("causes", [])]
        ttl_state = state.get("ttl", {})
        self.ttl_choices = [float(c) for c in ttl_state.get("choices", self.ttl_choices)] or list(self.ttl_choices)
        self.ttl_unit = float(ttl_state.get("unit", self.ttl_unit))
        self.ttl_bandits = {}
        for key, payload in ttl_state.get("bandits", {}).items():
            try:
                self.ttl_bandits[key] = DiscreteThompsonSampler.from_state(payload)
            except Exception:
                continue
        self.ttl_assignments = {
            k: (float(a), float(t))
            for k, (a, t) in ttl_state.get("assignments", {}).items()
            if isinstance(a, (int, float)) and isinstance(t, (int, float))
        }
        self.ttl_hazard = {}
        for key, payload in ttl_state.get("hazard", {}).items():
            try:
                self.ttl_hazard[key] = BoundedOnlineLinear.from_state(payload)
            except Exception:
                continue


# ============================================================
# 🧑‍🤝‍🧑 SocialModel
# ============================================================

@dataclass
class Agent:
    id: str
    name: str
    traits: Dict[str, float] = field(default_factory=dict)    # ex: honesty, assertiveness
    affinity: Dict[str, float] = field(default_factory=dict)  # agent_id -> [-1..1]
    goals: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)


class SocialModel:
    """
    Modèle social minimal :
    - Agents, rôles, affinités
    - Normes sociales (règles simples)
    - Attribution d'intentions (Bayes léger)
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_architecture = cognitive_architecture
        self.reward_bus: Optional[StructuredRewardBus] = None
        if cognitive_architecture is not None:
            self.reward_bus = getattr(cognitive_architecture, "reward_bus", None)
        self.agents: Dict[str, Agent] = {}
        self.norms: List[Tuple[str, str, str]] = []  # (context, action, norm_label)
        self.intent_stats: Dict[str, Dict[str, float]] = {}  # agent -> intent -> weight
        self.intent_models: Dict[str, Dict[str, BoundedOnlineLinear]] = {}
        self.intent_lr_bandit = DiscreteThompsonSampler([
            ("slow", 0.05),
            ("default", 0.1),
            ("fast", 0.2),
        ])
        # Favorise légèrement le réglage historique (0.1)
        self.intent_lr_bandit.success["default"] += 2.0

    # ----- Agents -----

    def add_agent(self, agent_id: str, name: str, traits: Optional[Dict[str, float]] = None):
        self.agents[agent_id] = Agent(agent_id, name, traits or {}, {}, [], [])

    def set_affinity(self, a: str, b: str, value: float):
        if a not in self.agents or b not in self.agents:
            return
        self.agents[a].affinity[b] = _clamp(float(value), -1.0, 1.0)

    def assign_role(self, agent_id: str, role: str):
        if agent_id in self.agents and role not in self.agents[agent_id].roles:
            self.agents[agent_id].roles.append(role)

    # ----- Normes -----

    def add_norm(self, context: str, action: str, label: str):
        self.norms.append((context, action, label))

    def applicable_norms(self, context: str) -> List[Tuple[str, str, str]]:
        return [n for n in self.norms if n[0] == context]

    # ----- Intentions (très simplifié) -----

    def update_intent(self, agent_id: str, intent: str, evidence: float):
        d = self.intent_stats.setdefault(agent_id, {})
        models = self.intent_models.setdefault(agent_id, {})
        model = models.get(intent)
        if model is None:
            model = BoundedOnlineLinear(dim=4, base_lr=0.1, bounds=[(-2.0, 2.0)] * 4, max_step=0.5)
            models[intent] = model

        features = self._intent_features(agent_id, float(evidence))
        lr, key = self.intent_lr_bandit.sample_value()
        _, updated = model.update(features, float(evidence), lr=lr)
        d[intent] = _clamp(updated, 0.0, 1.0)
        reward = 1.0 - abs(updated - float(evidence))
        self.intent_lr_bandit.update(key, reward)
        if self.reward_bus is not None:
            self.reward_bus.emit("social_prediction", reward)
        return d[intent]

    def most_likely_intent(self, agent_id: str) -> Optional[str]:
        d = self.intent_stats.get(agent_id, {})
        return max(d.items(), key=lambda kv: kv[1])[0] if d else None

    def _intent_features(self, agent_id: str, evidence: float) -> List[float]:
        agent = self.agents.get(agent_id)
        affinity_mean = 0.0
        trait_activation = 0.0
        role_density = 0.0
        if agent:
            affinity_mean = _mean(list(agent.affinity.values()), 0.0)
            trait_activation = _mean(list(agent.traits.values()), 0.0)
            role_density = min(1.0, len(agent.roles) / 5.0)
        return [1.0, evidence, affinity_mean, trait_activation + role_density]

    # ----- Persistance -----

    def to_state(self) -> Dict[str, Any]:
        return {
            "agents": {
                aid: {
                    "name": a.name,
                    "traits": dict(a.traits),
                    "affinity": dict(a.affinity),
                    "goals": list(a.goals),
                    "roles": list(a.roles),
                }
                for aid, a in self.agents.items()
            },
            "norms": list(self.norms),
            "intent_stats": {k: dict(v) for k, v in self.intent_stats.items()},
            "intent_models": {
                aid: {intent: model.to_state() for intent, model in intents.items()}
                for aid, intents in self.intent_models.items()
            },
            "intent_lr_bandit": self.intent_lr_bandit.to_state(),
        }

    def from_state(self, state: Dict[str, Any]):
        self.agents = {}
        for aid, d in state.get("agents", {}).items():
            self.agents[aid] = Agent(aid, d.get("name", aid), dict(d.get("traits", {})), dict(d.get("affinity", {})),
                                     list(d.get("goals", [])), list(d.get("roles", [])))
        self.norms = [tuple(x) for x in state.get("norms", [])]
        self.intent_stats = {k: dict(v) for k, v in state.get("intent_stats", {}).items()}
        self.intent_models = {}
        for aid, intents in state.get("intent_models", {}).items():
            models: Dict[str, BoundedOnlineLinear] = {}
            for intent, payload in intents.items():
                try:
                    models[intent] = BoundedOnlineLinear.from_state(payload)
                except Exception:
                    continue
            if models:
                self.intent_models[aid] = models
        try:
            self.intent_lr_bandit = DiscreteThompsonSampler.from_state(state.get("intent_lr_bandit", {}))
        except Exception:
            self.intent_lr_bandit = DiscreteThompsonSampler([
                ("slow", 0.05),
                ("default", 0.1),
                ("fast", 0.2),
            ])
            self.intent_lr_bandit.success["default"] += 2.0


# ============================================================
# ⚙️ PhysicsEngine
# ============================================================

@dataclass
class Body:
    id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    mass: float = 1.0
    radius: float = 0.5
    friction: float = 0.02
    restitution: float = 0.2  # rebond 0..1
    tags: List[str] = field(default_factory=list)


class PhysicsEngine:
    """
    Moteur physique 2D discret et simplifié :
    - intégration d'Euler (dt fixe)
    - collisions disque-disque élastiques partiellement (restitution)
    - friction linéaire
    - contraintes spatiales (mur rectangulaire)
    - hooks d'observation vers cognition (reasoning/perception)
    """
    def __init__(self, cognitive_architecture: Any = None, memory_system: Any = None):
        self.cognitive_arch = cognitive_architecture
        self.memory_system = memory_system
        self.bodies: Dict[str, Body] = {}
        self.bounds: Tuple[float, float, float, float] = (-10.0, -10.0, 10.0, 10.0)  # xmin,ymin,xmax,ymax
        self.dt: float = 0.05
        self.last_step_ts: float = 0.0
        self.events: List[str] = []
        self.reward_bus: Optional[StructuredRewardBus] = None
        self.target_decay: float = 0.05
        self.drift_log: List[Tuple[float, float]] = []
        self._dt_bandit = DiscreteThompsonSampler([
            ("slow", 0.02),
            ("baseline", 0.05),
            ("agile", 0.08),
        ])
        self._dt_bandit.success["baseline"] += 2.0

        # auto-wiring
        ca = self.cognitive_arch
        if ca:
            if self.reward_bus is None:
                self.reward_bus = getattr(ca, "reward_bus", None)
            self.reasoning = getattr(ca, "reasoning", None)
            self.perception = getattr(ca, "perception", None)
            self.goals = getattr(ca, "goals", None)
            self.metacognition = getattr(ca, "metacognition", None)
            # accès aux autres sous-modèles si exposés
            self.social = SocialModel(ca)
            self.temporal = TemporalReasoning(ca)
            self.spatial = SpatialReasoning(ca)
        else:
            self.social = SocialModel(None)
            self.temporal = TemporalReasoning(None)
            self.spatial = SpatialReasoning(None)

    # ----- Monde -----

    def add_body(self, body_id: str, x: float, y: float, **kwargs):
        self.bodies[body_id] = Body(body_id, float(x), float(y), **kwargs)

    def set_bounds(self, xmin: float, ymin: float, xmax: float, ymax: float):
        self.bounds = (float(xmin), float(ymin), float(xmax), float(ymax))

    # ----- Simulation -----

    def step(self, steps: int = 1):
        iterations = max(1, int(steps))
        prev_energy = self._total_kinetic_energy()
        dt_choice, dt_key = self._dt_bandit.sample_value()
        self.dt = _clamp(float(dt_choice), 0.01, 0.12)
        for _ in range(iterations):
            self._integrate()
            self._collisions()
        self.last_step_ts = _now()
        post_energy = self._total_kinetic_energy()
        decay_ratio = 0.0
        if prev_energy > 1e-6:
            decay_ratio = _clamp((prev_energy - post_energy) / prev_energy, -1.0, 1.0)
        reward = math.exp(-abs(decay_ratio - self.target_decay))
        self._dt_bandit.update(dt_key, reward)
        if self.reward_bus is not None:
            stability = 1.0 - min(1.0, abs(decay_ratio - self.target_decay) * 2.0)
            self.reward_bus.emit("physics_stability", max(0.0, stability))
        self._adapt_body_parameters(decay_ratio)
        if len(self.drift_log) >= 200:
            self.drift_log.pop(0)
        self.drift_log.append((self.last_step_ts, abs(decay_ratio)))

    def _integrate(self):
        xmin, ymin, xmax, ymax = self.bounds
        for b in self.bodies.values():
            # friction
            b.vx *= (1.0 - b.friction)
            b.vy *= (1.0 - b.friction)
            # intégration
            b.x += b.vx * self.dt
            b.y += b.vy * self.dt
            # murs
            if b.x - b.radius < xmin or b.x + b.radius > xmax:
                b.vx *= -b.restitution
                b.x = _clamp(b.x, xmin + b.radius, xmax - b.radius)
                self.events.append(f"bounce_x::{b.id}")
            if b.y - b.radius < ymin or b.y + b.radius > ymax:
                b.vy *= -b.restitution
                b.y = _clamp(b.y, ymin + b.radius, ymax - b.radius)
                self.events.append(f"bounce_y::{b.id}")

    def _collisions(self):
        ids = list(self.bodies.keys())
        n = len(ids)
        for i in range(n):
            bi = self.bodies[ids[i]]
            for j in range(i + 1, n):
                bj = self.bodies[ids[j]]
                dx = bj.x - bi.x
                dy = bj.y - bi.y
                dist = math.hypot(dx, dy)
                min_dist = bi.radius + bj.radius
                if dist < min_dist and dist > 1e-6:
                    # normaliser
                    nx, ny = dx / dist, dy / dist
                    # correction de position simple
                    overlap = min_dist - dist
                    bi.x -= nx * overlap * 0.5
                    bi.y -= ny * overlap * 0.5
                    bj.x += nx * overlap * 0.5
                    bj.y += ny * overlap * 0.5
                    # échange de vitesse projetée sur la normale (restitution)
                    vi = bi.vx * nx + bi.vy * ny
                    vj = bj.vx * nx + bj.vy * ny
                    vi_new = vj * bi.restitution
                    vj_new = vi * bj.restitution
                    bi.vx += (vi_new - vi) * nx
                    bi.vy += (vi_new - vi) * ny
                    bj.vx += (vj_new - vj) * nx
                    bj.vy += (vj_new - vj) * ny
                    self.events.append(f"collide::{bi.id}::{bj.id}")

    def _total_kinetic_energy(self) -> float:
        return 0.5 * sum(b.mass * (b.vx ** 2 + b.vy ** 2) for b in self.bodies.values())

    def _adapt_body_parameters(self, decay_ratio: float):
        if not self.bodies:
            return
        error = self.target_decay - decay_ratio
        for b in self.bodies.values():
            delta = _clamp(error * 0.02, -0.01, 0.01)
            b.friction = _clamp(b.friction + delta, 0.0, 0.3)
            b.restitution = _clamp(b.restitution - delta * 0.5, 0.0, 1.0)

    # ----- Observations simplifiées -----

    def snapshot(self) -> Dict[str, Any]:
        return {
            "bodies": {k: vars(v) for k, v in self.bodies.items()},
            "bounds": self.bounds,
            "events": list(self.events[-50:]),
            "time": self.last_step_ts,
        }

    # ----- Persistance -----

    def to_state(self) -> Dict[str, Any]:
        return {
            "bodies": {bid: {
                "x": b.x, "y": b.y, "vx": b.vx, "vy": b.vy,
                "mass": b.mass, "radius": b.radius,
                "friction": b.friction, "restitution": b.restitution,
                "tags": list(b.tags),
            } for bid, b in self.bodies.items()},
            "bounds": tuple(self.bounds),
            "dt": self.dt,
            "events": list(self.events[-200:]),
            "temporal": self.temporal.to_state(),
            "social": self.social.to_state(),
            "spatial": self.spatial.to_state(),
            "dt_bandit": self._dt_bandit.to_state(),
            "target_decay": self.target_decay,
            "drift_log": list(self.drift_log[-100:]),
        }

    def from_state(self, state: Dict[str, Any]):
        self.bodies = {}
        for bid, d in state.get("bodies", {}).items():
            self.bodies[bid] = Body(
                bid, float(d["x"]), float(d["y"]), float(d.get("vx", 0.0)), float(d.get("vy", 0.0)),
                float(d.get("mass", 1.0)), float(d.get("radius", 0.5)),
                float(d.get("friction", 0.02)), float(d.get("restitution", 0.2)),
                list(d.get("tags", []))
            )
        b = state.get("bounds", (-10.0, -10.0, 10.0, 10.0))
        self.bounds = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        self.dt = float(state.get("dt", 0.05))
        self.events = list(state.get("events", []))
        self.temporal.from_state(state.get("temporal", {}))
        self.social.from_state(state.get("social", {}))
        self.spatial.from_state(state.get("spatial", {}))
        try:
            self._dt_bandit = DiscreteThompsonSampler.from_state(state.get("dt_bandit", {}))
        except Exception:
            self._dt_bandit = DiscreteThompsonSampler([
                ("slow", 0.02),
                ("baseline", 0.05),
                ("agile", 0.08),
            ])
            self._dt_bandit.success["baseline"] += 2.0
        self.target_decay = float(state.get("target_decay", self.target_decay))
        self.drift_log = [(float(ts), float(val)) for ts, val in state.get("drift_log", [])]


# ============================================================
# 🔗 Exports publics
# ============================================================

__all__ = [
    "StructuredRewardBus",
    "PhysicsEngine",
    "SocialModel",
    "TemporalReasoning",
    "SpatialReasoning",
]
