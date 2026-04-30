# pip install --upgrade pip
# pip install rdflib pykeen pandas owlready2 torch owlrl matplotlib numpy
import dataclasses

import math
import os
import pickle
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple, Set, Optional, Callable
import itertools
import copy
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import rdflib
import torch
import torch.nn.functional as F
from rdflib.collection import Collection
from rdflib.namespace import RDF, RDFS, OWL

# OWL loading helpers + load_owl

torch.set_default_dtype(torch.float32)


def _uri_str(node: rdflib.term.Node) -> Optional[str]:
    return str(node) if isinstance(node, rdflib.term.URIRef) else None


def _ordered_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _local_name(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rsplit("/", 1)[-1]


def load_owl(
        owl_path: str,
        allowed_namespaces: Optional[Set[str]] = None,
) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Extract named classes, subclass relations, and disjoint pairs
    from an OWL ontology file.
    """
    g = rdflib.Graph()
    g.parse(owl_path, format="xml")

    def _keep(uri: str) -> bool:
        if allowed_namespaces is None:
            return True
        return any(uri.startswith(ns) for ns in allowed_namespaces)

    classes: Set[str] = set()

    for c in g.subjects(RDF.type, OWL.Class):
        uri = _uri_str(c)
        if uri and _keep(uri):
            classes.add(uri)

    for child, _, parent in g.triples((None, RDFS.subClassOf, None)):
        child_uri = _uri_str(child)
        parent_uri = _uri_str(parent)
        if child_uri and _keep(child_uri):  classes.add(child_uri)
        if parent_uri and _keep(parent_uri): classes.add(parent_uri)

    subclass_of: Set[Tuple[str, str]] = set()

    for child, _, parent in g.triples((None, RDFS.subClassOf, None)):
        child_uri, parent_uri = _uri_str(child), _uri_str(parent)
        if child_uri and parent_uri and _keep(child_uri) and _keep(parent_uri):
            subclass_of.add((child_uri, parent_uri))

    for a, _, b in g.triples((None, OWL.equivalentClass, None)):
        a_uri, b_uri = _uri_str(a), _uri_str(b)
        if a_uri and b_uri and _keep(a_uri) and _keep(b_uri):
            subclass_of.add((a_uri, b_uri))
            subclass_of.add((b_uri, a_uri))

    disjoint_pairs: Set[Tuple[str, str]] = set()

    for a, _, b in g.triples((None, OWL.disjointWith, None)):
        a_uri, b_uri = _uri_str(a), _uri_str(b)
        if a_uri and b_uri and _keep(a_uri) and _keep(b_uri):
            disjoint_pairs.add(_ordered_pair(a_uri, b_uri))

    for adc in g.subjects(RDF.type, OWL.AllDisjointClasses):
        for _, _, members in g.triples((adc, OWL.members, None)):
            try:
                collection = Collection(g, members)
            except Exception:
                continue
            member_uris = [
                uri for m in collection
                if (uri := _uri_str(m)) and _keep(uri)
            ]
            for i in range(len(member_uris)):
                for j in range(i + 1, len(member_uris)):
                    disjoint_pairs.add(_ordered_pair(member_uris[i], member_uris[j]))

    classes_list = sorted(classes)
    subclass_list = sorted(subclass_of)
    disjoint_list = sorted(disjoint_pairs)

    print(f"Loaded {owl_path}: {len(classes_list)} classes, "
          f"{len(subclass_list)} subclass axioms, {len(disjoint_list)} disjoint pairs")

    return classes_list, subclass_list, disjoint_list


# adding noise to ontology

def load_owl_with_errors(
        owl_path: str,
        allowed_namespaces: Optional[Set[str]] = None,
        subclass_drop_rate: float = 0.1,
        subclass_flip_rate: float = 0.0,
        disjoint_inject_rate: float = 0.05,
        class_drop_rate: float = 0.0,
        seed: int = 40,
) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Load ontology and inject controlled noise."""

    rng = random.Random(seed)
    classes, subclass_of, disjoint_pairs = load_owl(owl_path, allowed_namespaces)

    classes = set(classes)
    subclass_of = set(subclass_of)
    disjoint_pairs = set(disjoint_pairs)

    if class_drop_rate > 0:
        drop_classes = set(rng.sample(list(classes), int(len(classes) * class_drop_rate)))
        classes -= drop_classes
        subclass_of = {(c, p) for c, p in subclass_of if c not in drop_classes and p not in drop_classes}
        disjoint_pairs = {(a, b) for a, b in disjoint_pairs if a not in drop_classes and b not in drop_classes}

    if subclass_drop_rate > 0:
        subclass_of = {e for e in subclass_of if rng.random() > subclass_drop_rate}

    if subclass_flip_rate > 0:
        subclass_of = {(p, c) if rng.random() < subclass_flip_rate else (c, p)
                       for c, p in subclass_of}

    if disjoint_inject_rate > 0:
        class_list = list(classes)
        num_inject = int(disjoint_inject_rate * max(1, len(disjoint_pairs) + 1))
        injected = set()
        max_attempts = num_inject * 10
        attempts = 0
        while len(injected) < num_inject and attempts < max_attempts:
            pair = _ordered_pair(*rng.sample(class_list, 2))
            if pair not in disjoint_pairs:
                injected.add(pair)
            attempts += 1
        disjoint_pairs |= injected

    classes_list = sorted(classes)
    subclass_list = sorted(subclass_of)
    disjoint_list = sorted(disjoint_pairs)

    print(f"[NOISY] {owl_path}: {len(classes_list)} classes | "
          f"{len(subclass_list)} subclass | {len(disjoint_list)} disjoint")

    return classes_list, subclass_list, disjoint_list


class OntologyEdges:
    """
    Stores all subclass and disjoint edges for a given ontology,
    including asserted and entailed closures, depths, and entailment weights.
    Provides loss and violation helpers used by the training loops.
    """

    def __init__(self,
                 classes: List[str],
                 subclass_of: List[Tuple[str, str]],
                 disjoint_pairs: List[Tuple[str, str]],
                 device: Optional[str] = None):

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.cls2id = {uri: i for i, uri in enumerate(classes)}
        self.id2cls = {i: uri for i, uri in enumerate(classes)}
        self.num_classes = len(classes)

        # Subclass edges
        self.asserted_sub_edges = [(self.cls2id[c], self.cls2id[p]) for c, p in subclass_of]

        self.closure_sub_edges_with_depth = self._compute_closure_with_depth(self.asserted_sub_edges)
        self.closure_sub_edges = torch.tensor(
            [(c, a) for c, a, _ in self.closure_sub_edges_with_depth],
            dtype=torch.long, device=self.device
        )
        self.closure_depths = torch.tensor(
            [d for _, _, d in self.closure_sub_edges_with_depth],
            dtype=torch.float, device=self.device
        )

        # Disjoint edges
        self.asserted_disjoint_edges = [(self.cls2id[a], self.cls2id[b]) for a, b in disjoint_pairs]

        entailed_disjoint = self._entail_disjointness(
            self.asserted_disjoint_edges, self.closure_sub_edges_with_depth
        )
        self.entailed_disjoint_edges = torch.tensor(
            list(entailed_disjoint), dtype=torch.long, device=self.device
        )

        # Per-edge entailment weight
        self.entail_count = self._compute_entail_count()

        # Per-class depth from root (used in oversize_loss)
        depths = torch.full((self.num_classes,), float("inf"), device=self.device)
        all_children = {c for c, _ in self.asserted_sub_edges}
        all_parents = {p for _, p in self.asserted_sub_edges}
        for r in all_parents - all_children:
            depths[r] = 0.0

        changed = True
        while changed:
            changed = False
            for c, p, _ in self.closure_sub_edges_with_depth:
                if depths[p] + 1 < depths[c]:
                    depths[c] = depths[p] + 1
                    changed = True

        depths[torch.isinf(depths)] = depths[~torch.isinf(depths)].max()
        self.class_depths = depths

        # Sibling edges
        sibling_pairs = self._compute_sibling_pairs()
        self.sibling_edges = (
            torch.tensor(sibling_pairs, dtype=torch.long, device=self.device)
            if sibling_pairs else None
        )

        # Violation tracking
        self.last_subclass_violations: Optional[int] = None
        self.last_disjoint_violations: Optional[int] = None
        self.last_sibling_distance: Optional[float] = None

    # Internal helpers
    def _compute_closure_with_depth(self, edges):
        parents = {i: {} for i in range(self.num_classes)}
        for c, p in edges:
            parents[c][p] = 1

        changed = True
        while changed:
            changed = False
            for c in range(self.num_classes):
                for p, d in list(parents[c].items()):
                    for gp, gd in parents[p].items():
                        new_d = d + gd
                        if gp not in parents[c] or new_d < parents[c][gp]:
                            parents[c][gp] = new_d
                            changed = True

        return [(c, a, d) for c, anc in parents.items() for a, d in anc.items()]

    def _entail_disjointness(self, disjoint_edges, subclass_closure) -> set:
        descendants = {}
        for c, a, _ in subclass_closure:
            descendants.setdefault(a, set()).add(c)

        entailed = set(disjoint_edges)
        for a, b in disjoint_edges:
            for da in descendants.get(a, []):
                entailed.add(tuple(sorted((da, b))))
            for db in descendants.get(b, []):
                entailed.add(tuple(sorted((a, db))))
        return entailed

    def _compute_entail_count(self):
        descendants = defaultdict(set)
        ancestors = defaultdict(set)
        for c, p, _ in self.closure_sub_edges_with_depth:
            descendants[p].add(c)
            ancestors[c].add(p)
        for i in range(self.num_classes):
            descendants[i].add(i)
            ancestors[i].add(i)

        counts = [
            len(descendants[c]) * len(ancestors[p])
            for c, p, _ in self.closure_sub_edges_with_depth
        ]
        return torch.log1p(torch.tensor(counts, dtype=torch.float, device=self.device))

    def _compute_sibling_pairs(self) -> List[Tuple[int, int]]:
        parent_to_children: dict = defaultdict(set)
        for child, parent in self.asserted_sub_edges:
            parent_to_children[parent].add(child)

        pairs: set = set()
        for children in parent_to_children.values():
            children = list(children)
            for i in range(len(children)):
                for j in range(i + 1, len(children)):
                    pairs.add(tuple(sorted((children[i], children[j]))))
        return list(pairs)

    def check_entailed_and_closure_edges(
            self,
            mn: torch.Tensor,
            mx: torch.Tensor,
            disjoint_margin: float = 0.02,
    ) -> dict:
        """
        Check how many entailed disjoint and transitive closure subclass edges
        are correctly satisfied (found/predicted) by the current box embedding.

        An edge is considered 'found' when the geometric constraint holds:
          - Subclass closure edge (c ⊆ p): mn[p] <= mn[c] AND mx[c] <= mx[p]
                                            in ALL dimensions.
          - Entailed disjoint edge (a ⊓ b = ⊥): boxes are separated by at
                                                  least `disjoint_margin` in
                                                  at least one dimension.

        """

        result = {}

        # Closure subclass edges
        if self.closure_sub_edges.numel() > 0:
            child = self.closure_sub_edges[:, 0]  # (E,)
            parent = self.closure_sub_edges[:, 1]

            # Containment holds iff mn[parent] <= mn[child] AND mx[child] <= mx[parent] for all dims
            lower_ok = (mn[parent] <= mn[child])
            upper_ok = (mx[child] <= mx[parent])

            edge_ok = (lower_ok & upper_ok).all(dim=1)
            total = edge_ok.numel()
            found = edge_ok.sum().item()
        else:
            total = found = 0

        result["closure_sub_total"] = total
        result["closure_sub_found"] = int(found)
        result["closure_sub_rate"] = found / total if total > 0 else 1.0

        # Entailed disjoint edges
        if self.entailed_disjoint_edges.numel() > 0:
            a = self.entailed_disjoint_edges[:, 0]
            b = self.entailed_disjoint_edges[:, 1]

            # Separation in each dim (positive = gap, negative = overlap)
            sep_ab = mn[b] - mx[a]
            sep_ba = mn[a] - mx[b]
            sep = torch.maximum(sep_ab, sep_ba)  # best-case separation per dim

            # A pair is 'satisfied' if at least one dimension achieves >= margin
            pair_ok = (sep.max(dim=1).values >= disjoint_margin)
            dis_total = pair_ok.numel()
            dis_found = pair_ok.sum().item()
        else:
            dis_total = dis_found = 0

        result["entailed_dis_total"] = dis_total
        result["entailed_dis_found"] = int(dis_found)
        result["entailed_dis_rate"] = dis_found / dis_total if dis_total > 0 else 1.0

        return result

    # Violation counters
    def count_subclass_violations(self, mn: torch.Tensor, mx: torch.Tensor) -> int | None:
        if self.closure_sub_edges.numel() == 0:
            self.last_subclass_violations = 0
            return 0
        child, parent = self.closure_sub_edges[:, 0], self.closure_sub_edges[:, 1]
        lower = (mn[child] < mn[parent]).sum().item()
        upper = (mx[child] > mx[parent]).sum().item()
        self.last_subclass_violations = int(lower) + int(upper)
        return self.last_subclass_violations

    def count_disjoint_violations(self, mn: torch.Tensor, mx: torch.Tensor, margin: float = 0.02) -> int | None:
        if len(self.entailed_disjoint_edges) == 0:
            self.last_disjoint_violations = 0
            return 0
        a, b = self.entailed_disjoint_edges[:, 0], self.entailed_disjoint_edges[:, 1]
        sep_ab = mn[b] - mx[a]
        sep_ba = mn[a] - mx[b]
        sep = torch.maximum(sep_ab, sep_ba)
        violated = (sep.max(dim=1).values < margin).sum().item()
        self.last_disjoint_violations = int(violated)
        return self.last_disjoint_violations

    # Loss functions
    def subclass_loss(self, mn: torch.Tensor, mx: torch.Tensor) -> torch.Tensor:
        """Containment loss weighted by entailment count per edge."""
        child, parent = self.closure_sub_edges[:, 0], self.closure_sub_edges[:, 1]
        lower = F.softplus(mn[parent] - mn[child])
        upper = F.softplus(mx[child] - mx[parent])
        per_edge_loss = (lower + upper).sum(dim=1)
        return (self.entail_count * per_edge_loss).mean()

    def disjoint_loss(self, mn: torch.Tensor, mx: torch.Tensor,
                      margin: float = 0.02, use_entailed: bool = True) -> torch.Tensor:
        dis_edges = (self.entailed_disjoint_edges if use_entailed
                     else torch.tensor(self.asserted_disjoint_edges,
                                       dtype=torch.long, device=self.device))
        if len(dis_edges) == 0:
            return torch.tensor(0.0, device=self.device)

        a, b = dis_edges[:, 0], dis_edges[:, 1]
        sep_ab = mn[b] - mx[a]
        sep_ba = mn[a] - mx[b]
        sep = torch.maximum(sep_ab, sep_ba)
        return F.softplus(margin - sep.max(dim=1).values).mean()

    def oversize_loss(self, log_vols: torch.Tensor, cfg) -> torch.Tensor:
        target = cfg.base_log_volume - cfg.depth_scale * torch.sqrt(self.class_depths)
        target = torch.clamp(target, min=cfg.min_log_volume)
        return (
                F.softplus(log_vols - target) +
                F.softplus(cfg.min_log_volume - log_vols)
        ).mean()

    def distance_loss(self, mn: torch.Tensor, mx: torch.Tensor) -> torch.Tensor:
        """
        Penalises siblings that have empty space between them.
        A gap of zero means boxes touch or overlap — no penalty.
        """
        if self.sibling_edges is None or len(self.sibling_edges) == 0:
            return torch.tensor(0.0, device=self.device)

        a, b = self.sibling_edges[:, 0], self.sibling_edges[:, 1]

        # Per-dim gap: positive = actual separation, negative = overlap
        gap = torch.maximum(mn[b] - mx[a], mn[a] - mx[b])  # (n_pairs, dim)

        # Only penalise positive gaps (separation); overlapping pairs cost nothing
        return F.softplus(gap).mean()

    def avg_sibling_distance(self, mn: torch.Tensor, mx: torch.Tensor) -> float:
        """
        Average gap between sibling boxes across all pairs and dimensions.
        Positive values = separation (boxes apart), negative = overlap.
        """
        if self.sibling_edges is None or len(self.sibling_edges) == 0:
            self.last_sibling_distance = 0.0
            return 0.0
        a, b = self.sibling_edges[:, 0], self.sibling_edges[:, 1]
        gap = torch.maximum(mn[b] - mx[a], mn[a] - mx[b])  # (n_pairs, dim)
        self.last_sibling_distance = gap.mean().item()
        return self.last_sibling_distance


@dataclass
class BoxConfig:
    # geometry
    dim: int = 6

    # optimisation
    steps: int = 3000
    lr: float = 1.0 / math.sqrt(dim)
    seed: int = 42

    # regularisation
    min_box_size: float = 0.05
    size_weight: float = 0.1

    # disjoint margin
    disjoint_margin: float = 0.02

    # constraint weights
    subclass_weight: float = 10.0
    disjoint_weight: float = 1.0

    # oversized-box control
    big_box_weight: float = 0.1
    base_log_volume: float = 2.0
    depth_scale: float = 0.5
    min_log_volume: float = -4.0

    # sibling proximity
    distance_weight: float = 0.1


class BoxEmbedding(torch.nn.Module):
    """
    Each class is represented by an axis-aligned box in R^d.
      center ∈ R^d
      half_size ∈ R_+^d (via softplus)
      min = center - half_size
      max = center + half_size
    """

    def __init__(self, n_classes: int, dim: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.center = torch.nn.Parameter(torch.randn(n_classes, dim, generator=g) * 0.1)
        self.raw_half_size = torch.nn.Parameter(torch.randn(n_classes, dim, generator=g) * 0.1)
        self._eps = 1e-6

    def half_size(self) -> torch.Tensor:
        return F.softplus(self.raw_half_size, beta=1.0) + self._eps

    def get_min_max(self) -> Tuple[torch.Tensor, torch.Tensor]:
        hs = self.half_size()
        return self.center - hs, self.center + hs

    def side_lengths(self) -> torch.Tensor:
        return 2.0 * self.half_size()

    def volumes(self) -> torch.Tensor:
        """
        Log-volume per box: sum of log(side_length) across dims.
        """
        return torch.log(self.side_lengths().clamp(min=1e-8)).sum(dim=-1)


@dataclass
class CurriculumSchedule:
    subclass_start: float = 0.0  # structural foundation first
    disjoint_start: float = 0.4  # separation only after containment exists
    sibling_start: float = 0.5  # siblings should be close
    big_box_start: float = 0.7  # size control
    ramp: bool = False


def _build_df(model, edges, cfg):
    with torch.no_grad():
        mn_np, mx_np = (t.cpu().numpy() for t in model.get_min_max())
    classes = [edges.id2cls[i] for i in range(edges.num_classes)]
    return pd.DataFrame({
        "class_uri": classes,
        "class_name": [_local_name(u) for u in classes],
        **{f"min_{d}": mn_np[:, d] for d in range(cfg.dim)},
        **{f"max_{d}": mx_np[:, d] for d in range(cfg.dim)},
    }).sort_values("class_name").reset_index(drop=True)


def _load_ontology(owl_path, noise, _preloaded):
    if noise:
        return load_owl_with_errors(owl_path)
    return _preloaded or load_owl(owl_path)


def learn_boxes_from_owl(owl_path, cfg, device=None, _preloaded=None, noise=False):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

    classes, subclass_of, disjoint_pairs = _load_ontology(owl_path, noise, _preloaded)
    edges = OntologyEdges(classes, subclass_of, disjoint_pairs, device=device)

    if noise:
        clean_classes, clean_sub, clean_dis = _preloaded or load_owl(owl_path)
        eval_edges = OntologyEdges(clean_classes, clean_sub, clean_dis, device=device)
    else:
        eval_edges = edges

    model = torch.compile(BoxEmbedding(len(classes), cfg.dim, cfg.seed).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    for step in range(1, cfg.steps + 1):
        opt.zero_grad()
        mn, mx = model.get_min_max()
        mn, mx = mn.clamp(-1e6, 1e6), mx.clamp(-1e6, 1e6)

        loss = cfg.size_weight * F.softplus(cfg.min_box_size - (mx - mn).clamp(min=1e-6)).mean()

        if edges.closure_sub_edges.numel() > 0:
            loss += cfg.subclass_weight * edges.subclass_loss(mn, mx)

        if edges.sibling_edges is not None and len(edges.sibling_edges) > 0:
            loss += cfg.distance_weight * edges.distance_loss(mn, mx)

        if len(edges.asserted_disjoint_edges) > 0:
            loss += cfg.disjoint_weight * edges.disjoint_loss(mn, mx, margin=cfg.disjoint_margin, use_entailed=True)

        loss += cfg.big_box_weight * edges.oversize_loss(model.volumes(), cfg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if step % max(1, cfg.steps // 10) == 0:
            with torch.no_grad():
                mn_eval, mx_eval = model.get_min_max()
                mn_eval = mn_eval.clamp(-1e6, 1e6)
                mx_eval = mx_eval.clamp(-1e6, 1e6)
                sub_viol = edges.count_subclass_violations(mn_eval, mx_eval)
                dis_viol = edges.count_disjoint_violations(mn_eval, mx_eval, cfg.disjoint_margin)
                sib_dist = edges.avg_sibling_distance(mn_eval, mx_eval)

                edges.last_subclass_violations = sub_viol
                edges.last_disjoint_violations = dis_viol

            print(f"step {step:>5}/{cfg.steps} | loss={loss.item():.4f} "
                  f"| sub_viol={sub_viol} | dis_viol={dis_viol} | avg_sib_dist={sib_dist:.4f}")

    with torch.no_grad():
        mn, mx = model.get_min_max()
        mn, mx = mn.clamp(-1e6, 1e6), mx.clamp(-1e6, 1e6)

        edges.last_subclass_violations = eval_edges.count_subclass_violations(mn, mx)
        edges.last_disjoint_violations = eval_edges.count_disjoint_violations(
            mn, mx, cfg.disjoint_margin
        )
        edges.last_sibling_distance = edges.avg_sibling_distance(mn, mx)

    return model, _build_df(model, edges, cfg), edges


def learn_boxes_with_curriculum(owl_path, cfg, device=None, _preloaded=None,
                                schedule=None, noise=False):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

    classes, subclass_of, disjoint_pairs = _load_ontology(owl_path, noise, _preloaded)
    edges = OntologyEdges(classes, subclass_of, disjoint_pairs, device=device)

    if noise:
        clean_classes, clean_sub, clean_dis = _preloaded or load_owl(owl_path)
        eval_edges = OntologyEdges(clean_classes, clean_sub, clean_dis, device=device)
    else:
        eval_edges = edges

    model = torch.compile(BoxEmbedding(len(classes), cfg.dim, cfg.seed).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    def scheduled_weight(base_weight, start_frac, step):
        if schedule is None:
            return base_weight
        start_step = int(start_frac * cfg.steps)
        if step < start_step:
            return 0.0
        if not schedule.ramp or start_frac == 0.0:
            return base_weight
        progress = (step - start_step) / max(1, cfg.steps - start_step)
        return base_weight * min(max(progress, 0.0), 1.0)

    final_loss = None

    for step in range(1, cfg.steps + 1):
        opt.zero_grad()
        mn, mx = model.get_min_max()
        mn, mx = mn.clamp(-1e6, 1e6), mx.clamp(-1e6, 1e6)

        loss = cfg.size_weight * F.softplus(cfg.min_box_size - (mx - mn).clamp(min=1e-6)).mean()

        subclass_w = scheduled_weight(cfg.subclass_weight, schedule.subclass_start if schedule else 0.0, step)
        sibling_w = scheduled_weight(cfg.distance_weight, schedule.sibling_start if schedule else 0.0, step)
        disjoint_w = scheduled_weight(cfg.disjoint_weight, schedule.disjoint_start if schedule else 0.0, step)
        bigbox_w = scheduled_weight(cfg.big_box_weight, schedule.big_box_start if schedule else 0.0, step)

        if subclass_w > 0 and edges.closure_sub_edges.numel() > 0:
            loss += subclass_w * edges.subclass_loss(mn, mx)

        if sibling_w > 0:
            loss += sibling_w * edges.distance_loss(mn, mx)

        if disjoint_w > 0:
            loss += disjoint_w * edges.disjoint_loss(mn, mx, margin=cfg.disjoint_margin, use_entailed=True)

        if bigbox_w > 0:
            loss += bigbox_w * edges.oversize_loss(model.volumes(), cfg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        final_loss = loss.item()

        if step % max(1, cfg.steps // 10) == 0:
            mn_eval, mx_eval = model.get_min_max()
            mn_eval = mn_eval.clamp(-1e6, 1e6)
            mx_eval = mx_eval.clamp(-1e6, 1e6)

            sub_viol = edges.count_subclass_violations(mn_eval, mx_eval)
            dis_viol = edges.count_disjoint_violations(mn_eval, mx_eval, cfg.disjoint_margin)
            sib_dist = edges.avg_sibling_distance(mn_eval, mx_eval)

            edges.last_subclass_violations = sub_viol
            edges.last_disjoint_violations = dis_viol

            print(f"step {step:>5}/{cfg.steps} | loss={loss.item():.4f} "
                  f"| sub_viol={sub_viol} | dis_viol={dis_viol} | avg_sib_dist={sib_dist:.4f}")

    with torch.no_grad():
        mn, mx = model.get_min_max()
        mn, mx = mn.clamp(-1e6, 1e6), mx.clamp(-1e6, 1e6)

        edges.last_subclass_violations = eval_edges.count_subclass_violations(mn, mx)
        edges.last_disjoint_violations = eval_edges.count_disjoint_violations(
            mn, mx, cfg.disjoint_margin
        )
        edges.last_sibling_distance = edges.avg_sibling_distance(mn, mx)

    return model, _build_df(model, edges, cfg), edges, final_loss


def _train_one_dim(args: tuple) -> tuple[int, dict]:
    d, owl_path, device, learn_fn, classes, subclass_of, disjoint_pairs, schedule, noise, cfg = args

    if cfg is None:
        cfg = BoxConfig(dim=d, steps=10000, size_weight=0.1)

    else:
        cfg = dataclasses.replace(cfg, dim=d)

    kwargs = dict(
        owl_path=owl_path,
        cfg=cfg,
        device=device,
        _preloaded=(classes, subclass_of, disjoint_pairs),
        noise=noise,
    )

    if schedule is not None:
        kwargs["schedule"] = schedule

    result = learn_fn(**kwargs)

    if len(result) == 4:
        model, df, edges, loss = result
    else:
        model, df, edges = result
        loss = None

    return d, {
        "model": model,
        "df": df,
        "edges": edges,
        "sub_viol": edges.last_subclass_violations,
        "dis_viol": edges.last_disjoint_violations,
        "avg_sibling_dist": edges.last_sibling_distance,
    }


def sweep_dimensions(
        owl_path: str,
        learn_fn: Callable,
        dims=range(2, 11),
        device: Optional[str] = None,
        max_workers: Optional[int] = None,
        noise: bool = False,
        schedule=None,
        cfg=None,
        path: Optional[str] = None,
) -> Dict[int, Dict]:
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    dims = list(dims)

    print(f"Loading OWL: {owl_path}")
    classes, subclass_of, disjoint_pairs = (
        load_owl_with_errors(owl_path) if noise else load_owl(owl_path)
    )

    print(f"Sweeping dims={dims} | device={device} | "
          f"noise={noise} | schedule={'yes' if schedule else 'no'}")

    # prepare output directory once
    if path is not None:
        os.makedirs(path, exist_ok=True)

    results: Dict[int, Dict] = {}

    def _save_single(dim: int, info: dict):
        """Save one dimension."""
        if path is None:
            return

        dim_path = os.path.join(path, f"dim_{dim}")
        os.makedirs(dim_path, exist_ok=True)

        # unwrap compiled model
        raw_model = getattr(info["model"], "_orig_mod", info["model"])
        state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in raw_model.state_dict().items()
        }

        torch.save(state_dict, os.path.join(dim_path, "model.pt"))

        with open(os.path.join(dim_path, "data.pkl"), "wb") as f:
            pickle.dump({
                "df": info["df"],
                "sub_viol": info["sub_viol"],
                "dis_viol": info["dis_viol"],
                "avg_sibling_dist": info.get("avg_sibling_dist"),
            }, f)

    if device == "mps":
        for d in dims:
            print("=" * 60)
            print(f"Training dim={d} with {learn_fn.__name__}")

            _, result = _train_one_dim((
                d, owl_path, device, learn_fn,
                classes, subclass_of, disjoint_pairs,
                schedule, noise, cfg
            ))

            results[d] = result

            _save_single(d, result)

            print(f"dim={d} done | sub_viol={result['sub_viol']} | dis_viol={result['dis_viol']}")

    else:
        work_items = [
            (d, owl_path, device, learn_fn,
             classes, subclass_of, disjoint_pairs,
             schedule, noise, cfg)
            for d in dims
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_train_one_dim, item): item[0] for item in work_items}

            for fut in as_completed(futures):
                d = futures[fut]
                try:
                    d, result = fut.result()
                    results[d] = result

                    _save_single(d, result)

                    print(f"dim={d} done | sub_viol={result['sub_viol']} | dis_viol={result['dis_viol']}")
                except Exception as e:
                    print(f"[ERROR] dim={d} failed: {e}")

    print(f"\nFinished sweep. Stored {len(results)} dimensions.")
    return dict(sorted(results.items()))


def load_sweep_results(path, classes, subclass_of, disjoint_pairs, device=None):
    if device is None:
        device = torch.device("mps" if torch.mps.is_available() else "cpu")

    results = {}

    dim_dirs = sorted([d for d in os.listdir(path) if d.startswith("dim_")])

    for d in dim_dirs:
        dim_path = os.path.join(path, d)

        with open(os.path.join(dim_path, "data.pkl"), "rb") as f:
            data = pickle.load(f)

        state_dict = torch.load(
            os.path.join(dim_path, "model.pt"),
            map_location=device
        )

        # remove compile prefix if present
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # infer actual dimension
        dim = state_dict["center"].shape[1]

        raw_model = BoxEmbedding(len(classes), dim, seed=0).to(device)
        raw_model.load_state_dict(state_dict)

        data["model"] = torch.compile(raw_model)
        data["edges"] = OntologyEdges(classes, subclass_of, disjoint_pairs, device=device)
        results[dim] = data

    print(f"Loaded {len(results)} dimensions from {path}")
    return results


def plot_sweep_comparison(
        results_plain: dict,
        results_curriculum: dict,
        onto: str,
        figsize: tuple = (16, 5),
):
    dims = sorted(set(results_plain.keys()) & set(results_curriculum.keys()))
    x = np.arange(len(dims))
    bar_w = 0.35

    # Compute avg_box_size inline if not already attached by evaluate_models
    def _compute_metrics(results, d):
        model = results[d]["model"]
        edges = results[d]["edges"]
        with torch.no_grad():
            mn, mx = model.get_min_max()
        return {
            "sub_viol": edges.count_subclass_violations(mn, mx),
            "dis_viol": edges.count_disjoint_violations(mn, mx),
            "avg_box_size": (mx - mn).mean().item(),
            "avg_sibling_dist": edges.avg_sibling_distance(mn, mx),
        }

    METRICS = [
        ("sub_viol", "Final Subclass Violations"),
        ("dis_viol", "Final Disjoint Violations"),
        ("avg_box_size", "Avg Box Side Length"),
        ("avg_sibling_dist", "Avg Sibling Distance"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(figsize[0] + 5, figsize[1]))
    for ax, (key, title) in zip(axes, METRICS):
        plain_vals = [_compute_metrics(results_plain, d)[key] for d in dims]
        curr_vals = [_compute_metrics(results_curriculum, d)[key] for d in dims]

        ax.bar(x - bar_w / 2, plain_vals, bar_w, label="Plain", color="#5B8DB8", alpha=0.85)
        ax.bar(x + bar_w / 2, curr_vals, bar_w, label="Curriculum", color="#E07B54", alpha=0.85)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Embedding Dimension")
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in dims])
        ax.set_ylabel(title)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Plain vs Curriculum {onto} — final metrics across embedding dimensions",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"sweep_comparison_{onto}.png", dpi=150, bbox_inches="tight")
    plt.show()


def evaluate_models(results_dict) -> pd.DataFrame:
    records = []

    for dim in sorted(results_dict.keys()):
        model = results_dict[dim]["model"]
        edges = results_dict[dim]["edges"]

        with torch.no_grad():
            mn, mx = model.get_min_max()

        records.append({
            "dim": dim,
            "sub_viol": edges.count_subclass_violations(mn, mx),
            "dis_viol": edges.count_disjoint_violations(mn, mx),
            "avg_box_size": (mx - mn).mean().item(),
            "avg_sibling_dist": edges.avg_sibling_distance(mn, mx),
        })

    return pd.DataFrame(records)


def plot_evaluation(eval_df, title: str = "Ontology Box Evaluation"):
    dims = eval_df["dim"]
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("Embedding dimension")
    ax1.set_ylabel("Violations", color="tab:red")
    ax1.plot(dims, eval_df["sub_viol"], marker="o", color="tab:red", label="Subclass violations")
    ax1.plot(dims, eval_df["dis_viol"], marker="s", color="tab:orange", label="Disjoint violations")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Average values", color="tab:blue")
    ax2.plot(dims, eval_df["avg_box_size"], marker="^", color="tab:blue", label="Avg box size")
    if "avg_sibling_dist" in eval_df.columns:
        ax2.plot(dims, eval_df["avg_sibling_dist"], marker="D", color="tab:purple",
                 linestyle="--", label="Avg sibling distance")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.legend(loc="upper right")

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def evaluate_concluded_relationships(results_dict, disjoint_margin: float):
    """
    Compute violations for concluded (entailed, non-asserted) relationships.
    """
    records = []

    for dim, info in results_dict.items():
        model = info['model']
        edges = info['edges']

        with torch.no_grad():
            mn, mx = model.get_min_max()

        embedding_dim = mn.shape[1]

        # concluded subclass edges
        asserted_sub_set = {(int(c), int(p)) for c, p in edges.asserted_sub_edges}

        closure_list = edges.closure_sub_edges.tolist()  # [[c, p], ...]
        concluded_sub_idx = [
            i for i, cp in enumerate(closure_list)
            if (int(cp[0]), int(cp[1])) not in asserted_sub_set
        ]

        if concluded_sub_idx:
            idx_t = torch.tensor(concluded_sub_idx, dtype=torch.long,
                                 device=edges.device)
            c_edges = edges.closure_sub_edges[idx_t]
            c_ids, p_ids = c_edges[:, 0], c_edges[:, 1]

            # A subclass edge violates when child is NOT contained in parent.
            lower_viol = (mn[c_ids] < mn[p_ids]).sum().item()
            upper_viol = (mx[c_ids] > mx[p_ids]).sum().item()
            conc_sub_viol = int(lower_viol) + int(upper_viol)
            conc_sub_total = len(concluded_sub_idx)
        else:
            conc_sub_viol = 0
            conc_sub_total = 0

        # concluded disjoint pairs
        asserted_dis_set = {frozenset(p) for p in edges.asserted_disjoint_edges}

        if len(edges.entailed_disjoint_edges) > 0:
            entailed_list = edges.entailed_disjoint_edges.tolist()
            concluded_dis_idx = [
                i for i, pair in enumerate(entailed_list)
                if frozenset(pair) not in asserted_dis_set
            ]
        else:
            concluded_dis_idx = []

        if concluded_dis_idx:
            idx_t = torch.tensor(concluded_dis_idx, dtype=torch.long,
                                 device=edges.device)
            c_dis = edges.entailed_disjoint_edges[idx_t]
            a_ids, b_ids = c_dis[:, 0], c_dis[:, 1]

            sep_ab = mn[b_ids] - mx[a_ids]
            sep_ba = mn[a_ids] - mx[b_ids]
            sep = torch.maximum(sep_ab, sep_ba)

            # A disjoint pair is violated when boxes still overlap (max-sep < margin).
            violated = (sep.max(dim=1).values < disjoint_margin).sum().item()
            conc_dis_viol = int(violated)
            conc_dis_total = len(concluded_dis_idx)
        else:
            conc_dis_viol = 0
            conc_dis_total = 0

        records.append({
            'dim': dim,
            'conc_sub_total': conc_sub_total,
            'conc_sub_viol': conc_sub_viol,
            'conc_sub_rate': conc_sub_viol / max(conc_sub_total * embedding_dim, 1),
            'conc_dis_total': conc_dis_total,
            'conc_dis_viol': conc_dis_viol,
            'conc_dis_rate': conc_dis_viol / max(conc_dis_total, 1),
        })

    return pd.DataFrame(records).sort_values('dim').reset_index(drop=True)


def plot_concluded_evaluation(eval_df, title="Concluded Relationship Violations"):
    dims = eval_df["dim"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (y_plain, y_rate, ylabel, t) in zip(axes, [
        (eval_df["conc_sub_viol"], eval_df["conc_dis_viol"], "Violation count", "Raw violation counts"),
        (eval_df["conc_sub_rate"], eval_df["conc_dis_rate"], "Violation rate",
         "Violation rates (fraction of possible)"),
    ]):
        ax.plot(dims, y_plain, marker="o", color="tab:red", label="Concluded subclass")
        ax.plot(dims, y_rate, marker="s", color="tab:orange", label="Concluded disjoint")
        ax.set_xlabel("Embedding dimension")
        ax.set_ylabel(ylabel)
        ax.set_title(t, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def sweep_schedule_combinations(
        owl_path,
        params,
        dim,
        steps,
        base_schedule,
        _preloaded=None,
        device=None,
):
    """
    Sweep over combinations of curriculum schedule parameters.
    """

    # Build all combinations
    keys = list(params.keys())
    values = list(params.values())
    combos = list(itertools.product(*values))

    results = []

    for combo in combos:
        # Build schedule 
        sched = copy.deepcopy(base_schedule)
        for k, v in zip(keys, combo):
            setattr(sched, k, v)

        # Build config 
        cfg = BoxConfig(
            dim=dim,
            steps=steps,
        )

        # Train 
        model, df, edges, final_loss = learn_boxes_with_curriculum(
            owl_path=owl_path,
            cfg=cfg,
            device=device,
            _preloaded=_preloaded,
            schedule=sched,
        )

        # Final metrics 
        mn, mx = model.get_min_max()

        sub_viol = edges.count_subclass_violations(mn, mx)
        dis_viol = edges.count_disjoint_violations(mn, mx, cfg.disjoint_margin)

        result_row = {
            **{k: v for k, v in zip(keys, combo)},
            "sub_viol": sub_viol,
            "dis_viol": dis_viol,
            "loss_final": final_loss,
        }

        results.append(result_row)

        print(f"Done: {result_row}")

    return pd.DataFrame(results)


# Plotting — pairwise heatmaps
def plot_combo_heatmaps(
        df: pd.DataFrame,
        title: str = "Schedule Parameter Interaction",
        use_norm: bool = False,
        metrics: list[str] | None = None,
        cell_w: float = 5.0,
        cell_h: float = 4.5,
):
    param_cols = [
        c for c in df.columns
        if c not in ("sub_viol", "dis_viol", "sub_viol_norm", "dis_viol_norm", "loss_final")
    ]

    if len(param_cols) < 2:
        raise ValueError("Need at least 2 swept parameters.")

    pairs = [
        (param_cols[i], param_cols[j])
        for i in range(len(param_cols))
        for j in range(i + 1, len(param_cols))
    ]

    # Default: sub_viol + dis_viol + loss_final
    if metrics is None:
        suffix = "_norm" if use_norm else ""
        metrics = [f"sub_viol{suffix}", f"dis_viol{suffix}", "loss_final"]

    # Only keep metrics that actually exist in the dataframe
    metrics = [m for m in metrics if m in df.columns]

    n_rows = len(pairs)
    n_cols = len(metrics)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_w * n_cols, cell_h * n_rows),
        squeeze=False,
    )

    for row, (px, py) in enumerate(pairs):
        for col, metric in enumerate(metrics):
            ax = axes[row][col]

            pivot = pd.pivot_table(
                df, values=metric, index=py, columns=px, aggfunc="mean"
            )
            pivot = pivot.sort_index(ascending=False).sort_index(axis=1)

            data = np.nan_to_num(pivot.values, nan=0.0)

            is_norm_metric = use_norm and not metric.startswith("loss")
            im = ax.imshow(
                data,
                aspect="auto",
                cmap="RdYlGn_r",
                vmin=0,
                vmax=1 if is_norm_metric else None,
            )

            x_vals = pivot.columns.tolist()
            y_vals = pivot.index.tolist()

            ax.set_xticks(range(len(x_vals)))
            ax.set_yticks(range(len(y_vals)))
            ax.set_xticklabels([f"{v:.0%}" for v in x_vals], fontsize=9)
            ax.set_yticklabels([f"{v:.0%}" for v in y_vals], fontsize=9)

            ax.set_xlabel(px, fontsize=10)
            ax.set_ylabel(py, fontsize=10)
            ax.set_title(metric, fontsize=10, pad=6)

            # Annotate cells with raw value
            raw_metric = metric.replace("_norm", "")
            pivot_raw = pd.pivot_table(
                df, values=raw_metric, index=py, columns=px, aggfunc="mean"
            ).reindex_like(pivot)

            for yi in range(len(y_vals)):
                for xi in range(len(x_vals)):
                    val = pivot_raw.iloc[yi, xi]
                    if not np.isnan(val):
                        fmt = f"{val:.3f}" if metric.startswith("loss") else f"{int(val)}"
                        ax.text(xi, yi, fmt,
                                ha="center", va="center", fontsize=8,
                                color="white" if data[yi, xi] > 0.6 else "black")

            plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_combo_lines(
        df: pd.DataFrame,
        x_param: str,
        color_param: str,
        metric: str = "sub_viol",
        title: Optional[str] = None,
):
    # aggregate over all params
    group_cols = [x_param, color_param]

    df_agg = (
        df.groupby(group_cols)[metric]
        .mean()
        .reset_index()
    )

    values = sorted(df_agg[color_param].unique())
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, val in enumerate(values):
        sub = df_agg[df_agg[color_param] == val].sort_values(x_param)

        ax.plot(
            sub[x_param],
            sub[metric],
            marker="o",
            linewidth=2,
            color=cmap(i / max(len(values) - 1, 1)),
            label=f"{color_param}={val:.0%}",
        )

    ax.set_xlabel(x_param)
    ax.set_ylabel(metric)

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )

    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    ax.set_title(title or f"{metric}: {x_param} vs {color_param}")

    plt.tight_layout()
    plt.show()


def plot_all_starts_heatmap(
        df: pd.DataFrame,
        title: str = "Parameter Main Effects",
        use_norm: bool = False,
):
    """
    Single heatmap showing the marginal effect of each _start parameter.

    Rows   = parameters (disjoint_start, subclass_start, …)
    Cols   = the swept values (0.0, 0.5, …)
    Color  = mean violation when that param is set to that value,
             averaged over all combinations of the other params.
    """
    metric_a = "sub_viol_norm" if use_norm else "sub_viol"
    metric_b = "dis_viol_norm" if use_norm else "dis_viol"

    param_cols = [
        c for c in df.columns
        if c not in ("sub_viol", "dis_viol", "sub_viol_norm", "dis_viol_norm")
    ]

    # Collect every unique value that appears across all params
    all_values = sorted({v for col in param_cols for v in df[col].unique()})

    fig, axes = plt.subplots(1, 2, figsize=(max(6, len(all_values) * 2.5),
                                            len(param_cols) * 1.4 + 1.5))

    for ax, metric in zip(axes, [metric_a, metric_b]):
        raw_metric = metric.replace("_norm", "")

        # Build (n_params × n_values) matrices for colour and annotation
        colour_mat = np.full((len(param_cols), len(all_values)), np.nan)
        label_mat = np.full((len(param_cols), len(all_values)), np.nan)

        for ri, param in enumerate(param_cols):
            for ci, val in enumerate(all_values):
                mask = df[param] == val
                if mask.any():
                    colour_mat[ri, ci] = df.loc[mask, metric].mean()
                    label_mat[ri, ci] = df.loc[mask, raw_metric].mean()

        vmax = 1.0 if use_norm else np.nanmax(colour_mat)
        im = ax.imshow(colour_mat, aspect="auto", cmap="RdYlGn_r",
                       vmin=0, vmax=vmax)

        # Axis labels
        ax.set_xticks(range(len(all_values)))
        ax.set_xticklabels([f"{v:.0%}" for v in all_values], fontsize=9)
        ax.set_yticks(range(len(param_cols)))
        ax.set_yticklabels(param_cols, fontsize=9)
        ax.set_xlabel("start value")
        ax.set_title(metric, fontsize=10, fontweight="bold")

        # Annotate cells with the raw count / value
        for ri in range(len(param_cols)):
            for ci in range(len(all_values)):
                v = label_mat[ri, ci]
                if not np.isnan(v):
                    ax.text(ci, ri, f"{v:.1f}",
                            ha="center", va="center", fontsize=8)

        plt.colorbar(im, ax=ax, shrink=0.75)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
