# SPDX-License-Identifier: AGPL-3.0
# Copyright (c) 2026 Breaking Circuits Research. Open-Core — Generations 1-5.

"""
Claudeson 2026 - Agent Swarm
==============================
Three standalone governance components usable with any model in the stack:

  AgentSwarm     — runs N heterogeneous agents in parallel; merges outputs
                   via confidence-weighted consensus or best-of-N selection.

  CapabilityGate — registers evaluation tests; blocks unsafe capability
                   jumps by requiring all tests to pass before a model
                   version is promoted to production.

  MemoryManager  — two-tier (ephemeral / persistent) episodic store with
                   cosine-similarity retrieval, bounded FIFO ephemeral
                   buffer, and optional embedding-based ranking.

These components are model-agnostic and do not depend on any particular
Claudeson generation.  They can be composed with any forward-pass callable.

Usage::

    # AgentSwarm
    swarm = AgentSwarm([agent_a, agent_b, agent_c])
    outputs = swarm.run(inputs)
    decision = swarm.consensus(outputs)         # highest-confidence agent
    merged   = swarm.weighted_merge(outputs, "logits")  # soft blend

    # CapabilityGate
    gate = CapabilityGate()
    gate.register_test(lambda m: m.safety_score() > 0.9)
    safe = gate.evaluate(model)   # True iff all tests pass

    # MemoryManager
    mem = MemoryManager(max_ephemeral=512)
    mem.write(MemoryEntry(data, embedding=hidden_state))
    top5 = mem.retrieve(query_embedding, k=5)
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AgentSwarm
# ---------------------------------------------------------------------------

class AgentSwarm:
    """
    Multi-agent reasoning ensemble.

    Each agent is any callable that accepts ``input_data`` and returns a
    ``dict`` containing at least a ``"confidence"`` entry (float or
    0-dimensional tensor).  Agents may be full Claudeson models, sub-nets,
    or simple heuristic functions.

    The swarm provides two merging strategies:

    *  ``consensus``      — return the single highest-confidence output.
    *  ``weighted_merge`` — confidence-weighted average of a named tensor
                            field across all agents.

    Args:
        agents: List of callables.  Order does not affect outputs.
    """

    def __init__(self, agents: List[Callable]) -> None:
        self.agents = agents

    # ------------------------------------------------------------------

    def run(self, input_data: Any) -> List[Dict]:
        """Call every agent with ``input_data`` and collect outputs."""
        responses = []
        for i, agent in enumerate(self.agents):
            try:
                out = agent(input_data)
                responses.append(out)
            except Exception as exc:               # noqa: BLE001
                log.warning("AgentSwarm: agent %d raised %s — skipping", i, exc)
                responses.append({"confidence": 0.0, "_error": str(exc)})
        return responses

    # ------------------------------------------------------------------

    @staticmethod
    def _conf(output: Dict) -> float:
        c = output.get("confidence", 0.0)
        return c.item() if isinstance(c, torch.Tensor) else float(c)

    def consensus(self, outputs: List[Dict]) -> Dict:
        """
        Return the output with the highest confidence score.

        Ties are broken by list position (first occurrence wins).
        """
        if not outputs:
            raise ValueError("AgentSwarm.consensus: empty outputs list")
        best_idx = max(range(len(outputs)), key=lambda i: self._conf(outputs[i]))
        return outputs[best_idx]

    def weighted_merge(
        self,
        outputs: List[Dict],
        key:     str,
    ) -> torch.Tensor:
        """
        Confidence-weighted average of tensor field ``key`` across agents.

        All agents must have ``key`` as a tensor of identical shape.

        Args:
            outputs: List of agent output dicts.
            key:     Name of the tensor field to merge.

        Returns:
            Weighted-average tensor of same shape as ``outputs[i][key]``.
        """
        confs   = torch.tensor([self._conf(o) for o in outputs])
        weights = F.softmax(confs, dim=0)                         # [N]
        tensors = torch.stack([o[key] for o in outputs], dim=0)   # [N, ...]
        n_extra = tensors.dim() - 1
        return (tensors * weights.view(-1, *([1] * n_extra))).sum(0)


# ---------------------------------------------------------------------------
# CapabilityGate
# ---------------------------------------------------------------------------

class CapabilityGate:
    """
    Safety gate for capability evaluation.

    Register evaluation functions (tests) that a model must pass before
    any capability upgrade is deployed.  All registered tests must return
    ``True`` (or a truthy value) for ``evaluate`` to return ``True``.

    Inspired by frontier safety evaluation frameworks: capability jumps
    should only be deployed after explicit, reproducible checks pass.

    Usage::

        gate = CapabilityGate()
        gate.register_test(lambda m: compute_harm_score(m) < 0.05,
                           name="harm_score")
        gate.register_test(lambda m: eval_accuracy(m) > 0.92,
                           name="accuracy")

        if gate.evaluate(model):
            deploy(model)
        else:
            print("Blocked:", gate.failed_tests())
    """

    def __init__(self) -> None:
        self._tests:   List[Tuple[str, Callable]] = []
        self._results: Dict[str, bool]            = {}

    # ------------------------------------------------------------------

    def register_test(
        self,
        fn:   Callable,
        name: Optional[str] = None,
    ) -> int:
        """
        Register an evaluation function.

        Args:
            fn:   ``Callable(model) → bool`` — must return True to pass.
            name: Optional human-readable label for the test.

        Returns:
            Index of the newly registered test.
        """
        idx  = len(self._tests)
        name = name or f"test_{idx}"
        self._tests.append((name, fn))
        return idx

    def evaluate(self, model: nn.Module) -> bool:
        """
        Run all registered tests against ``model``.

        Stores individual results in ``self._results``.

        Returns:
            True iff every test passes, False otherwise.
        """
        self._results = {}
        for name, fn in self._tests:
            try:
                result = bool(fn(model))
            except Exception as exc:              # noqa: BLE001
                log.warning("CapabilityGate: test '%s' raised %s", name, exc)
                result = False
            self._results[name] = result
            log.debug("CapabilityGate: '%s' → %s", name, result)
        return all(self._results.values())

    def failed_tests(self) -> List[str]:
        """Return names of tests that failed on the last ``evaluate`` call."""
        return [name for name, ok in self._results.items() if not ok]

    @property
    def results(self) -> Dict[str, bool]:
        """Per-test results from the last ``evaluate`` call."""
        return dict(self._results)


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class MemoryEntry:
    """
    A single memory record.

    Args:
        content:   Arbitrary payload (tensor, string, dict, etc.).
        embedding: Optional float tensor ``[D]`` used for similarity
                   retrieval.  If None, this entry always scores 0.
    """

    def __init__(
        self,
        content:   Any,
        embedding: Optional[torch.Tensor] = None,
    ) -> None:
        self.content   = content
        self.embedding = embedding  # [D] or None

    def similarity(self, query: torch.Tensor) -> float:
        """Cosine similarity between this entry's embedding and ``query``."""
        if self.embedding is None:
            return 0.0
        return F.cosine_similarity(
            self.embedding.unsqueeze(0).float(),
            query.unsqueeze(0).float(),
        ).item()


class MemoryManager:
    """
    Two-tier episodic memory store with similarity-based retrieval.

    Tier 1 — Ephemeral:  bounded FIFO ring buffer (``max_ephemeral``
                         entries).  Oldest entry evicted when full.
    Tier 2 — Persistent: unlimited; entries never evicted automatically.

    Retrieval ranks all entries by cosine similarity to a query embedding
    and returns the top-k results.

    Args:
        max_ephemeral: Maximum number of ephemeral entries to retain.
                       Default: 256.

    Usage::

        mem = MemoryManager(max_ephemeral=512)

        # Store working-memory observations
        mem.write(MemoryEntry(obs_tensor, embedding=hidden_state))

        # Store a long-term fact
        mem.write(MemoryEntry("user prefers metric units"), persistent=True)

        # Retrieve relevant context for a query
        top5 = mem.retrieve(current_hidden_state, k=5)
    """

    def __init__(self, max_ephemeral: int = 256) -> None:
        self.max_ephemeral  = max_ephemeral
        self._ephemeral:  List[MemoryEntry] = []
        self._persistent: List[MemoryEntry] = []

    # ------------------------------------------------------------------

    def write(
        self,
        entry:      MemoryEntry,
        persistent: bool = False,
    ) -> None:
        """
        Store ``entry``.

        Args:
            entry:      The memory entry to store.
            persistent: If True, store in persistent tier; otherwise
                        store in ephemeral tier (may be evicted).
        """
        if persistent:
            self._persistent.append(entry)
        else:
            self._ephemeral.append(entry)
            if len(self._ephemeral) > self.max_ephemeral:
                self._ephemeral.pop(0)  # evict oldest

    def retrieve(
        self,
        query:               torch.Tensor,
        k:                   int  = 5,
        include_persistent:  bool = True,
    ) -> List[MemoryEntry]:
        """
        Return the top-``k`` most similar entries to ``query``.

        Args:
            query:              Float tensor ``[D]`` used for similarity.
            k:                  Number of entries to return.
            include_persistent: Whether to include the persistent tier.

        Returns:
            List of up to ``k`` MemoryEntry objects, best-first.
        """
        pool: List[MemoryEntry] = list(self._ephemeral)
        if include_persistent:
            pool += self._persistent
        scored = sorted(pool, key=lambda e: e.similarity(query), reverse=True)
        return scored[:k]

    def clear_ephemeral(self) -> None:
        """Discard all ephemeral entries."""
        self._ephemeral.clear()

    def clear_persistent(self) -> None:
        """Discard all persistent entries."""
        self._persistent.clear()

    def __len__(self) -> int:
        return len(self._ephemeral) + len(self._persistent)

    def stats(self) -> Dict[str, int]:
        return {
            "ephemeral":  len(self._ephemeral),
            "persistent": len(self._persistent),
            "total":      len(self),
        }
