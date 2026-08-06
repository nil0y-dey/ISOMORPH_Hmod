"""
Supply Chain Simulation, extended with physical-realism and
scenario-control knobs.

Physical-realism (Bernoulli-random) knobs:

  1. Supplier disruption: Bernoulli-onset outages at each source, fixed
     length. While a source is down, its reservoir does not refill, its
     (s,S) does not fire, and _replenish_warehouses cannot pull from it.

  2. Finite source supply: sources hold a per-item raw-material reservoir
     that refills at production_rate/day and caps at reservoir_cap. When
     the source's (s,S) fires, the outbound order is capped at the
     reservoir level, so shortages propagate downstream. Opt-in: only
     installed when --prod_rate_per_item or --source_production is set.

  3. Finite warehouse cap: each non-source, non-destination node has an
     optional total volumetric capacity. `_replenish_warehouses` caps the
     order quantity at (cap - current_used_vol - committed_incoming_vol)
     / item.volume, so shipments never overshoot capacity.

  4. Stochastic edge transit: per-shipment multiplicative Gaussian noise
     on the deterministic edge-sum, sampled_tt = base_tt * (1 + N(0,std)),
     clipped to >=1 day.

Deterministic-scenario knobs:

  5. Targeted edge cut window: zero the capacity of a chosen set of
     directed edges during [disable_from_day, disable_from_day+disable_days),
     with an optional linear post-cut restore ramp.

  6. Per-tier (s,S) scale overrides: src, hub, t2, t3, t4, t5 each accept
     an independent scale, falling back to ss_scale when unset. Useful
     for building tier-by-tier staircase depletions.

  7. Configurable SKU count: --n_items selects the first N of I01..I50
     for small-catalog experiments.

Reproducibility: with every knob at its default (no reservoir, no
warehouse cap, p_onset=0, edge_tt_std_frac=0, no disable_edges, no
per-tier overrides, n_items=50), the runtime RNG consumption and node
semantics reduce exactly to the base Supplychaingeo_item50_inv.py, so
the same CLI produces byte-identical outputs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import os
import csv
import random

import numpy as np
import pandas as pd

from Supplychaingeo_item50_inv import (
    Network,
    Node,
    SupplyChainSimulation,
    allocate_units_along_path_greedy,
    build_example_simulation_from_adjacency,
)


# ============================================================================
# ReservoirNode — source node with finite raw-material buffer
# ============================================================================

@dataclass
class ReservoirNode(Node):
    reservoir_cap: Dict[str, float] = field(default_factory=dict)
    reservoir_level: Dict[str, float] = field(default_factory=dict)
    production_rate: Dict[str, float] = field(default_factory=dict)
    frozen: bool = False  # set by sim each step; True during disruption

    def refill_reservoir(self) -> None:
        if self.frozen:
            return
        for iid, cap in self.reservoir_cap.items():
            r = self.production_rate.get(iid, 0.0)
            cur = self.reservoir_level.get(iid, 0.0)
            self.reservoir_level[iid] = min(cap, cur + r)

    def maybe_place_orders(self, day: int, rng: random.Random) -> None:
        if self.is_destination or not self.is_source:
            return
        if self.frozen:
            return
        for item_id, s in self.s_levels.items():
            on_hand = self.inventory.get(item_id, 0)
            if on_hand >= s:
                continue
            if self.outstanding_orders.get(item_id) is not None:
                continue
            S = self.S_levels.get(item_id, on_hand)
            requested = max(S - on_hand, 0)
            if requested <= 0:
                continue
            avail = int(self.reservoir_level.get(item_id, 0.0))
            qty = min(requested, avail)
            if qty <= 0:
                continue
            self.reservoir_level[item_id] -= qty
            mean_lt = max(self.lead_time_mean.get(item_id, 1.0), 0.1)
            std = self.lead_time_std_frac * mean_lt
            sampled = rng.normalvariate(mean_lt, std)
            lt_days = max(1, int(math.ceil(sampled)))
            self.outstanding_orders[item_id] = (day + lt_days, qty)


def install_reservoirs(
    net: Network,
    item_ids: List[str],
    prod_rate_per_item: float = 100.0,
    reservoir_cap_per_item: float = 5000.0,
    init_frac: float = 1.0,
) -> None:
    for nid, node in list(net.nodes.items()):
        if not node.is_source:
            continue
        prod = {iid: float(prod_rate_per_item) for iid in item_ids}
        cap = {iid: float(reservoir_cap_per_item) for iid in item_ids}
        init = {iid: float(reservoir_cap_per_item * init_frac)
                for iid in item_ids}
        net.nodes[nid] = ReservoirNode(
            node_id=node.node_id, lat=node.lat, lon=node.lon,
            is_destination=node.is_destination, is_source=True,
            inventory=dict(node.inventory),
            s_levels=dict(node.s_levels),
            S_levels=dict(node.S_levels),
            lead_time_mean=dict(node.lead_time_mean),
            lead_time_std_frac=node.lead_time_std_frac,
            outstanding_orders=dict(node.outstanding_orders),
            backlog=dict(node.backlog),
            reservoir_cap=cap,
            reservoir_level=init,
            production_rate=prod,
        )


# ============================================================================
# Outage schedule generator
# ============================================================================

def _generate_outage_schedule(
    rng: random.Random,
    horizon: int,
    source_ids: List[str],
    p_onset: float,
    length_days: int,
) -> Dict[str, np.ndarray]:
    """For each source, return an availability bool array of length `horizon`.
    Each day, if available, an outage of exactly `length_days` starts with
    prob `p_onset`. The source stays down for the full window (no overlap)
    before becoming eligible for a new outage."""
    out: Dict[str, np.ndarray] = {}
    for src in source_ids:
        a = np.ones(horizon, dtype=bool)
        if p_onset <= 0 or length_days <= 0:
            out[src] = a
            continue
        d = 0
        while d < horizon:
            if rng.random() < p_onset:
                end = min(d + length_days, horizon)
                a[d:end] = False
                d = end
            else:
                d += 1
        out[src] = a
    return out


# ============================================================================
# Scenario-knob helpers (per-tier ss overrides + n_items trim)
# ============================================================================

# Tier assignments for the released 13-node US graph. Only referenced by
# per-tier ss overrides; if none are set, this map is not used.
NODE_TIER: Dict[str, str] = {
    "SanFrancisco": "src", "StLouis": "src", "Orlando": "src",
    "Nashville":    "hub",
    "Atlanta":      "t2",
    "Chicago":      "t3", "Charlotte": "t3", "Memphis": "t3",
    "Columbus":     "t4", "Richmond":  "t4",
    "Philadelphia": "t5", "Baltimore": "t5",
}


def _apply_per_tier_ss(net: Network, scenario: Optional[dict]) -> None:
    """Rescale inventory/s_levels/S_levels per-tier, on top of the
    ss_scale that the base builder already applied. No-op when none of
    the per-tier keys are set. Does not touch the RNG."""
    sc = scenario or {}
    ss_scale = float(sc.get("ss_scale", 1.0))
    tier_over: Dict[str, Optional[float]] = {
        "src": sc.get("ss_src"),
        "hub": sc.get("ss_hub"),
        "t2":  sc.get("ss_tier2"),
        "t3":  sc.get("ss_tier3"),
        "t4":  sc.get("ss_tier4"),
        "t5":  sc.get("ss_tier5"),
    }
    if all(v is None for v in tier_over.values()):
        return
    for nid, node in net.nodes.items():
        if node.is_destination:
            continue
        tier = NODE_TIER.get(nid)
        if tier is None:
            continue
        override = tier_over[tier]
        if override is None:
            continue
        factor = float(override) / max(ss_scale, 1e-9)
        for iid in list(node.inventory.keys()):
            node.inventory[iid] = int(round(node.inventory[iid] * factor))
        for iid in list(node.s_levels.keys()):
            node.s_levels[iid] = max(0, int(round(
                node.s_levels[iid] * factor)))
        for iid in list(node.S_levels.keys()):
            s_new = node.s_levels.get(iid, 0)
            node.S_levels[iid] = max(s_new + 1, int(round(
                node.S_levels[iid] * factor)))


def _trim_items(
    net: Network,
    items: Dict[str, "object"],
    demand_signals: np.ndarray,
    n_items: Optional[int],
) -> Tuple[Dict[str, "object"], np.ndarray]:
    """Reduce the item catalog to the first n_items SKUs (I01..I{n_items}).
    The base builder always generates all 50; this trims after the fact
    for small-catalog experiments. Extra items already generated are
    thrown away but their RNG draws are preserved, keeping the kept
    items' volumes / (s,S) parameters identical to a full 50-item run."""
    if n_items is None or n_items >= len(items):
        return items, demand_signals
    item_ids_full = sorted(items.keys())
    kept = set(item_ids_full[:n_items])
    for iid in list(items.keys()):
        if iid not in kept:
            del items[iid]
    for node in net.nodes.values():
        for d in (node.inventory, node.s_levels, node.S_levels,
                  node.lead_time_mean, node.backlog,
                  node.outstanding_orders):
            for iid in list(d.keys()):
                if iid not in kept:
                    del d[iid]
    return items, demand_signals[:, :n_items]


# ============================================================================
# Extended simulator
# ============================================================================

class ExtendedSupplyChainSimulation(SupplyChainSimulation):
    def __init__(
        self,
        *args,
        warehouse_cap: Optional[Dict[str, float]] = None,
        disruption_p_onset: float = 0.0,
        disruption_length_days: int = 0,
        edge_tt_std_frac: float = 0.0,
        disable_edges: Optional[List[Tuple[str, str]]] = None,
        disable_from_day: int = -1,
        disable_days: int = 0,
        restore_ramp_days: int = 0,
        log_reservoir: bool = True,
        log_availability: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.warehouse_cap = warehouse_cap or {}
        self.disruption_p_onset = float(disruption_p_onset)
        self.disruption_length_days = int(disruption_length_days)
        self.edge_tt_std_frac = float(edge_tt_std_frac)
        self.log_reservoir = log_reservoir
        self.log_availability = log_availability

        # Edge-cut window: deterministic zero-capacity on selected edges
        # over a fixed day range, with optional linear restore ramp.
        self.disable_edges = set(tuple(e) for e in (disable_edges or []))
        self.disable_from_day = int(disable_from_day)
        self.disable_days = int(disable_days)
        self.restore_ramp_days = int(restore_ramp_days)
        self._active_day: int = 0

        self._source_ids = [nid for nid, n in self.network.nodes.items()
                            if n.is_source]
        self._reservoir_source_ids = [
            nid for nid in self._source_ids
            if isinstance(self.network.nodes[nid], ReservoirNode)]

        # Precompute per-source availability (uses self.rng so runs are
        # reproducible per seed). With p_onset=0 or length=0 no rng draws
        # are consumed, keeping the runtime RNG stream aligned with the
        # base simulator's for reproduction.
        self.source_available = _generate_outage_schedule(
            self.rng, self.horizon_days, self._source_ids,
            self.disruption_p_onset, self.disruption_length_days)

        # Summary printout (only when there is something to report)
        if self.disruption_p_onset > 0 and self.disruption_length_days > 0:
            for nid, a in self.source_available.items():
                down = int((~a).sum())
                flips = int(np.sum(np.diff(a.astype(int)) < 0))
                n_events = flips + (1 if (not bool(a[0])) else 0)
                print(f"    disruption[{nid}]: {down} down-days "
                      f"across {n_events} outage windows")

        self.reservoir_history: List[dict] = []
        self.availability_history: List[dict] = []

        # Wrap net.reset_daily_edges so the targeted edge cut window +
        # linear restore ramp are applied at the base sim's step-3
        # without needing to override the entire step(). Only installed
        # when a cut is actually configured, so the default path is a
        # zero-overhead pass-through to the base method.
        if self.disable_edges and self.disable_days > 0:
            _orig_reset = self.network.reset_daily_edges

            def _reset_with_cut(cap_factor: float = 1.0) -> None:
                day = self._active_day
                cut_end = self.disable_from_day + self.disable_days
                in_cut = self.disable_from_day <= day < cut_end
                if in_cut:
                    _orig_reset(cap_factor, disabled=self.disable_edges)
                    return
                _orig_reset(cap_factor)
                if (self.restore_ramp_days > 0
                        and cut_end <= day
                        < cut_end + self.restore_ramp_days):
                    frac = (day - cut_end + 1) / float(
                        self.restore_ramp_days)
                    frac = max(0.0, min(1.0, frac))
                    for e_key in self.disable_edges:
                        e = self.network.edges.get(e_key)
                        if e is not None:
                            e.reset_daily(frac)

            self.network.reset_daily_edges = _reset_with_cut

    # ---- helpers --------------------------------------------------------

    def _is_source_available(self, node_id: str, day: int) -> bool:
        arr = self.source_available.get(node_id)
        if arr is None:
            return True
        return bool(arr[day])

    def _sample_tt(self, base_tt: float) -> float:
        if self.edge_tt_std_frac <= 0:
            return float(base_tt)
        noise = self.rng.normalvariate(0.0, self.edge_tt_std_frac)
        return max(1.0, float(base_tt) * (1.0 + noise))

    def _total_tt(self, pe):
        # Called by base step() when shipping to destination.
        base = sum(self.network.edges[e].travel_time_days for e in pe)
        return self._sample_tt(base)

    def _warehouse_headroom_units(self, node, item) -> int:
        cap = self.warehouse_cap.get(node.node_id)
        if cap is None:
            return 10**12  # effectively unlimited
        used_vol = 0.0
        for iid in self.items:
            used_vol += node.inventory.get(iid, 0) * self.items[iid].volume
        committed_vol = 0.0
        for iid, order in node.outstanding_orders.items():
            if order is not None:
                committed_vol += order[1] * self.items[iid].volume
        headroom_vol = cap - used_vol - committed_vol
        if headroom_vol <= 0 or item.volume <= 0:
            return 0
        return int(headroom_vol / item.volume)

    # ---- overrides ------------------------------------------------------

    def _replenish_warehouses(self, day: int) -> None:
        """Adds warehouse-cap enforcement, source-disruption skipping, and
        stochastic transit time to the base's _replenish_warehouses."""
        net = self.network
        if self.round_robin_items and self.item_order:
            k = day % len(self.item_order)
            ids_today = self.item_order[k:] + self.item_order[:k]
        else:
            ids_today = self.item_order[:]

        for nid in self.replenish_order:
            node = net.nodes[nid]
            for iid in ids_today:
                on_hand = node.inventory.get(iid, 0)
                s = node.s_levels.get(iid, 0)
                if on_hand >= s:
                    continue
                if node.outstanding_orders.get(iid) is not None:
                    continue
                S = node.S_levels.get(iid, on_hand)
                qty_needed = max(S - on_hand, 0)
                if qty_needed <= 0:
                    continue

                # (3) Cap by remaining warehouse headroom
                qty_needed = min(
                    qty_needed,
                    self._warehouse_headroom_units(node, self.items[iid]))
                if qty_needed <= 0:
                    continue

                for sup_id, tt, path, pe in \
                        self.node_suppliers.get(nid, []):
                    sup_node = net.nodes[sup_id]
                    # (1) Skip disrupted sources
                    if sup_node.is_source and \
                            not self._is_source_available(sup_id, day):
                        continue
                    avail_at_sup = sup_node.inventory.get(iid, 0)
                    if avail_at_sup <= 0:
                        continue
                    attempt = min(avail_at_sup, qty_needed)
                    placed = allocate_units_along_path_greedy(
                        self.items[iid], attempt, pe, net.edges)
                    if placed <= 0:
                        continue

                    sup_node.inventory[iid] -= placed
                    # (4) Stochastic transit time
                    tt_sampled = self._sample_tt(tt)
                    arr = day + max(1, int(math.ceil(tt_sampled)))
                    node.outstanding_orders[iid] = (arr, placed)

                    r = [day, arr, sup_id, nid, iid, placed,
                         str(path),
                         str([net.edges[e].travel_time_days
                              for e in pe])]
                    if self.streaming_out_dir:
                        self._csv_buffers.setdefault(
                            'ship', []).append(r)
                    else:
                        self.shipments_log.append({
                            "day": day, "arrival_day": arr,
                            "from": sup_id, "to": nid,
                            "item": iid, "units": placed,
                            "path_nodes": path,
                            "edge_times": [net.edges[e].travel_time_days
                                           for e in pe]})
                    break

    def step(self, day: int) -> None:
        # Make `day` visible to the monkey-patched reset_daily_edges.
        self._active_day = day

        # 0) Set today's disruption flags on ReservoirNodes
        for nid in self._reservoir_source_ids:
            self.network.nodes[nid].frozen = \
                not self._is_source_available(nid, day)

        # 0a) Refill reservoirs (refill_reservoir short-circuits when frozen)
        for nid in self._reservoir_source_ids:
            self.network.nodes[nid].refill_reservoir()

        super().step(day)

        # 8) Snapshots
        if self.log_reservoir:
            if self.streaming_out_dir:
                for nid in self._reservoir_source_ids:
                    node = self.network.nodes[nid]
                    for iid, lvl in node.reservoir_level.items():
                        self._csv_buffers.setdefault('res', []).append(
                            [day, nid, iid, float(lvl)])
            else:
                for nid in self._reservoir_source_ids:
                    node = self.network.nodes[nid]
                    for iid, lvl in node.reservoir_level.items():
                        self.reservoir_history.append({
                            "day": day, "node": nid, "item": iid,
                            "reservoir": float(lvl)})

        if self.log_availability:
            if self.streaming_out_dir:
                for nid in self._source_ids:
                    self._csv_buffers.setdefault('avail', []).append(
                        [day, nid, int(self._is_source_available(nid, day))])
            else:
                for nid in self._source_ids:
                    self.availability_history.append({
                        "day": day, "node": nid,
                        "available": int(self._is_source_available(nid, day))})

    def _open_csv_files(self):
        super()._open_csv_files()
        if self.log_reservoir:
            f = open(os.path.join(self.streaming_out_dir,
                                  "reservoir_history.csv"),
                     "w", newline="", buffering=65536)
            w = csv.writer(f)
            w.writerow(["day", "node", "item", "reservoir"])
            self._csv_files['res'] = f
            self._csv_writers['res'] = w
            self._csv_buffers['res'] = []
        if self.log_availability:
            f = open(os.path.join(self.streaming_out_dir,
                                  "source_availability.csv"),
                     "w", newline="", buffering=65536)
            w = csv.writer(f)
            w.writerow(["day", "node", "available"])
            self._csv_files['avail'] = f
            self._csv_writers['avail'] = w
            self._csv_buffers['avail'] = []


# ============================================================================
# Builder — reuse base network/demand, install reservoirs & warehouse caps
# ============================================================================

def build_example_simulation_extended(
    seed: int = 123,
    horizon_days: int = 7300,
    pipeline_multiplier: float = 3.0,
    streaming_out_dir: Optional[str] = None,
    packing: str = "greedy",
    scenario: Optional[dict] = None,
    # Ext 2 (reservoir) — opt-in. Default None reproduces base Node semantics.
    prod_rate_per_item: Optional[float] = None,
    reservoir_cap_per_item: Optional[float] = None,
    init_frac: float = 1.0,
    # Ext 3 (warehouse volumetric cap). None = unlimited.
    warehouse_cap_scale: Optional[float] = None,
    # Ext 1 (Bernoulli source outages).
    disruption_p_onset: float = 0.0,
    disruption_length_days: int = 0,
    # Ext 4 (stochastic edge transit).
    edge_tt_std_frac: float = 0.0,
    # Ext 5 (targeted edge cut window).
    disable_edges: Optional[List[Tuple[str, str]]] = None,
    disable_from_day: int = -1,
    disable_days: int = 0,
    restore_ramp_days: int = 0,
):
    """Build the extended simulation. When every knob is at its
    default (reservoir off, no warehouse cap, no outages, no tt noise,
    no edge cut, no per-tier ss overrides in the scenario dict, no
    n_items trim), the resulting sim reproduces the base
    Supplychaingeo_item50_inv.py behavior byte-for-byte."""
    base_sim, net, items, demand_signals = \
        build_example_simulation_from_adjacency(
            seed=seed,
            horizon_days=horizon_days,
            pipeline_multiplier=pipeline_multiplier,
            streaming_out_dir=None,
            packing=packing,
            scenario=scenario)

    # Apply per-tier (s,S) overrides on top of the base-applied ss_scale.
    # No RNG consumed; no-op unless scenario carries ss_src/ss_hub/ss_tier*.
    _apply_per_tier_ss(net, scenario)

    # Optional catalog trim (I01..I{n_items}). No RNG consumed.
    n_items = (scenario or {}).get("n_items")
    items, demand_signals = _trim_items(net, items, demand_signals, n_items)
    item_ids = sorted(items.keys())

    # Reservoirs are opt-in: only install when user actually asked for one.
    # Leaves sources as plain Node otherwise, matching base behavior.
    if prod_rate_per_item is not None:
        cap = reservoir_cap_per_item if reservoir_cap_per_item is not None \
            else 1e18  # effectively unlimited: acts as a pure
                       # production-rate faucet with no reservoir depletion
        install_reservoirs(
            net, item_ids,
            prod_rate_per_item=prod_rate_per_item,
            reservoir_cap_per_item=cap,
            init_frac=init_frac)

    # Per-node warehouse cap = target-inventory volume * scale.
    # Only applied to non-source, non-destination nodes.
    warehouse_cap: Dict[str, float] = {}
    if warehouse_cap_scale is not None:
        for nid, node in net.nodes.items():
            if node.is_source or node.is_destination:
                continue
            s_vol = sum(node.S_levels.get(iid, 0) * items[iid].volume
                        for iid in item_ids)
            warehouse_cap[nid] = float(s_vol * warehouse_cap_scale)

    # Provenance printout
    n_sources = len(
        [n for n in net.nodes.values() if n.is_source])
    print(f"  Items: {len(item_ids)}  Sources: {n_sources}")
    if prod_rate_per_item is not None:
        total_prod = prod_rate_per_item * len(item_ids) * n_sources
        mean_demand = float(demand_signals.mean() * len(item_ids))
        cap_str = f"{reservoir_cap_per_item:.0f}" \
            if reservoir_cap_per_item is not None else "inf"
        print(f"  Reservoir: cap={cap_str} "
              f"prod={prod_rate_per_item:.1f}/item/day/src  "
              f"aggregate={total_prod:.0f}/day vs "
              f"demand={mean_demand:.0f}/day "
              f"({total_prod/max(mean_demand,1e-9):.2f}x)")
    else:
        print(f"  Reservoir: OFF (sources use base magic (s,S))")
    if warehouse_cap:
        caps_str = ", ".join(f"{k}={v:.0f}" for k, v in warehouse_cap.items())
        print(f"  Warehouse caps (vol, scale={warehouse_cap_scale}): {caps_str}")
    else:
        print(f"  Warehouse caps: unlimited")
    print(f"  Disruption: p_onset={disruption_p_onset} "
          f"length={disruption_length_days}d  "
          f"Edge tt noise: std_frac={edge_tt_std_frac}")
    if disable_edges:
        print(f"  Edge cut: {list(disable_edges)}  "
              f"from day {disable_from_day} for {disable_days}d  "
              f"restore_ramp={restore_ramp_days}d")

    sim = ExtendedSupplyChainSimulation(
        network=net, items=items,
        destination_id="NewYork",
        demand_fn=base_sim.demand_fn,
        horizon_days=horizon_days,
        seed=seed,
        pipeline_multiplier=pipeline_multiplier,
        streaming_out_dir=streaming_out_dir,
        packing=packing,
        warehouse_cap=warehouse_cap,
        disruption_p_onset=disruption_p_onset,
        disruption_length_days=disruption_length_days,
        edge_tt_std_frac=edge_tt_std_frac,
        disable_edges=disable_edges,
        disable_from_day=disable_from_day,
        disable_days=disable_days,
        restore_ramp_days=restore_ramp_days,
    )
    return sim, net, items, demand_signals


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(
        description="Extended supply-chain simulation "
                    "(reservoir + warehouse cap + outages + tt noise + "
                    "edge cut + per-tier ss + n_items)")
    ap.add_argument("--days",          type=int,   default=7300)
    ap.add_argument("--seed",          type=int,   default=2025)
    ap.add_argument("--out_dir",       type=str,   default="test_output_ext")
    ap.add_argument("--pipeline_mult", type=float, default=0.0)
    ap.add_argument("--no_streaming",  action="store_true")

    # Extension 2 (reservoir). Opt-in: unset = plain Node (base semantics).
    ap.add_argument("--prod_rate_per_item",     type=float, default=None,
                    help="Per-item, per-source production rate (units/day). "
                         "Setting this installs ReservoirNode at every "
                         "source. Omit to keep base magic (s,S) behavior.")
    ap.add_argument("--reservoir_cap_per_item", type=float, default=None,
                    help="Per-item reservoir cap. Omit to make the "
                         "reservoir effectively unlimited (pure "
                         "production-rate faucet, no reservoir depletion).")
    ap.add_argument("--init_frac",              type=float, default=1.0)
    ap.add_argument("--source_production",      type=float, default=None,
                    help="Shortcut for the pure faucet model: sets "
                         "--prod_rate_per_item=X and leaves the reservoir "
                         "cap unlimited, so each source produces X units "
                         "per item per day with no depletion dynamic.")

    # Extension 3 (warehouse cap). None → unlimited.
    ap.add_argument("--warehouse_cap_scale",    type=float, default=None,
                    help="Per-node total-volume cap as multiple of "
                         "target-inventory volume (sum S_iid * volume). "
                         "Omit for unlimited.")

    # Extension 1 (Bernoulli supplier disruption)
    ap.add_argument("--disruption_p_onset",     type=float, default=0.0,
                    help="Per-day Bernoulli prob of an outage starting "
                         "at each source (when currently available).")
    ap.add_argument("--disruption_length_days", type=int,   default=0,
                    help="Fixed length of each outage in days.")

    # Extension 4 (stochastic edge transit)
    ap.add_argument("--edge_tt_std_frac",       type=float, default=0.0,
                    help="Per-shipment Gaussian noise on total transit "
                         "time: sampled = base * (1 + N(0, std_frac)), "
                         "clipped to >=1 day.")

    # Extension 5 (targeted edge cut window).
    ap.add_argument("--disable_edges", type=str, default="",
                    help="Semicolon-separated 'u,v' pairs of directed "
                         "edges to zero-capacity during the disruption "
                         "window. e.g. 'Nashville,Atlanta' or "
                         "'Atlanta,Chicago;Nashville,Atlanta'.")
    ap.add_argument("--disable_from_day", type=int, default=-1)
    ap.add_argument("--disable_days",     type=int, default=0)
    ap.add_argument("--restore_ramp_days", type=int, default=0,
                    help="After the cut ends, previously-disabled edges' "
                         "capacity ramps linearly from 0 to full over this "
                         "many days. Default 0 = instant restoration.")

    # Baseline scenario knobs (unchanged from base file)
    ap.add_argument("--phi_lo",             type=float, default=0.9990)
    ap.add_argument("--phi_hi",             type=float, default=0.9996)
    ap.add_argument("--shock_count_scale",  type=float, default=1.0)
    ap.add_argument("--shock_height_scale", type=float, default=1.0)
    ap.add_argument("--seasonal_scale",     type=float, default=1.0)
    ap.add_argument("--containers_scale",   type=float, default=1.0)
    ap.add_argument("--ss_scale",           type=float, default=1.0,
                    help="Global (s,S) scale. Per-tier overrides below take "
                         "precedence when set.")
    # Per-tier (s,S) overrides.
    ap.add_argument("--ss_src",   type=float, default=None,
                    help="(s,S) scale override for source nodes.")
    ap.add_argument("--ss_hub",   type=float, default=None,
                    help="(s,S) scale override for the Hub (Nashville).")
    ap.add_argument("--ss_tier2", type=float, default=None,
                    help="(s,S) scale override for Tier-2 (Atlanta).")
    ap.add_argument("--ss_tier3", type=float, default=None,
                    help="(s,S) scale override for Tier-3 (Chicago, "
                         "Charlotte, Memphis).")
    ap.add_argument("--ss_tier4", type=float, default=None,
                    help="(s,S) scale override for Tier-4 (Columbus, "
                         "Richmond).")
    ap.add_argument("--ss_tier5", type=float, default=None,
                    help="(s,S) scale override for Tier-5 (Philadelphia, "
                         "Baltimore).")
    ap.add_argument("--leadtime_scale",     type=float, default=1.0)
    ap.add_argument("--burst_rate_scale",   type=float, default=1.0)
    ap.add_argument("--burst_height_scale", type=float, default=1.0)
    ap.add_argument("--base_lambda_lo",     type=float, default=80.0)
    ap.add_argument("--base_lambda_hi",     type=float, default=250.0)
    ap.add_argument("--n_items",            type=int,   default=50,
                    help="Number of item SKUs to simulate (default 50); "
                         "smaller values enable small-catalog experiments.")
    ap.add_argument("--scenario_name",      type=str,   default="baseline_ext")
    args = ap.parse_args()

    # --source_production shortcut: sets prod rate, leaves cap unlimited.
    if args.source_production is not None:
        if args.prod_rate_per_item is None:
            args.prod_rate_per_item = args.source_production
        if args.reservoir_cap_per_item is None:
            pass  # None -> unlimited inside build_example_simulation_extended

    # Parse edge-cut list "u,v;u2,v2" -> [("u","v"), ("u2","v2")]
    disable_edges_list: List[Tuple[str, str]] = []
    if args.disable_edges:
        for pair in args.disable_edges.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            u, v = [s.strip() for s in pair.split(",", 1)]
            disable_edges_list.append((u, v))

    scenario = {
        "name": args.scenario_name,
        "phi_lo": args.phi_lo, "phi_hi": args.phi_hi,
        "shock_count_scale": args.shock_count_scale,
        "shock_height_scale": args.shock_height_scale,
        "seasonal_scale": args.seasonal_scale,
        "containers_scale": args.containers_scale,
        "ss_scale": args.ss_scale,
        "leadtime_scale": args.leadtime_scale,
        "burst_rate_scale": args.burst_rate_scale,
        "burst_height_scale": args.burst_height_scale,
        "base_lambda_lo": args.base_lambda_lo,
        "base_lambda_hi": args.base_lambda_hi,
        "seed": args.seed, "days": args.days,
        "n_items": args.n_items,
        # Edge-cut knobs recorded for provenance
        "disable_edges": disable_edges_list,
        "disable_from_day": args.disable_from_day,
        "disable_days": args.disable_days,
        "restore_ramp_days": args.restore_ramp_days,
        # Ext knobs also recorded for provenance
        "prod_rate_per_item": args.prod_rate_per_item,
        "reservoir_cap_per_item": args.reservoir_cap_per_item,
        "init_frac": args.init_frac,
        "warehouse_cap_scale": args.warehouse_cap_scale,
        "disruption_p_onset": args.disruption_p_onset,
        "disruption_length_days": args.disruption_length_days,
        "edge_tt_std_frac": args.edge_tt_std_frac,
    }
    # Only include per-tier ss overrides that the user actually passed.
    for k, v in [("ss_src", args.ss_src), ("ss_hub", args.ss_hub),
                 ("ss_tier2", args.ss_tier2), ("ss_tier3", args.ss_tier3),
                 ("ss_tier4", args.ss_tier4), ("ss_tier5", args.ss_tier5)]:
        if v is not None:
            scenario[k] = float(v)

    streaming = not args.no_streaming and args.days > 500

    print("=== Extended Supply Chain Simulation ===")
    print(f"  Days: {args.days:,} ({args.days/365:.1f} years)  "
          f"Seed: {args.seed}")
    print(f"  Pipeline: {args.pipeline_mult}  Streaming: {streaming}")
    print()

    sim, net, items, dsig = build_example_simulation_extended(
        seed=args.seed,
        horizon_days=args.days,
        pipeline_multiplier=args.pipeline_mult,
        streaming_out_dir=args.out_dir if streaming else None,
        packing="greedy",
        scenario=scenario,
        prod_rate_per_item=args.prod_rate_per_item,
        reservoir_cap_per_item=args.reservoir_cap_per_item,
        init_frac=args.init_frac,
        warehouse_cap_scale=args.warehouse_cap_scale,
        disruption_p_onset=args.disruption_p_onset,
        disruption_length_days=args.disruption_length_days,
        edge_tt_std_frac=args.edge_tt_std_frac,
        disable_edges=disable_edges_list,
        disable_from_day=args.disable_from_day,
        disable_days=args.disable_days,
        restore_ramp_days=args.restore_ramp_days,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "scenario.json"), "w") as _f:
        _json.dump(scenario, _f, indent=2)

    dd, ds, svc, di, db, dt = sim.run()

    print("Saving demand signals...")
    np.save(os.path.join(args.out_dir, "demand_signals.npy"),
            dsig[:args.days])
    with open(os.path.join(args.out_dir, "demand_signals_cols.txt"), "w") as f:
        f.write(",".join(sorted(items.keys())) + "\n")
    print(f"  Saved shape={dsig[:args.days].shape}")

    if not streaming:
        dd.to_csv(os.path.join(args.out_dir, "daily_records.csv"),
                  index=False)
        ds.to_csv(os.path.join(args.out_dir, "shipments.csv"), index=False)
        svc.to_csv(os.path.join(args.out_dir, "service_summary.csv"),
                   index=False)
        if args.days <= 500:
            di.to_csv(os.path.join(args.out_dir, "inventory_history.csv"),
                      index=False)
            db.to_csv(os.path.join(args.out_dir, "backlog_history.csv"),
                      index=False)
            dt.to_csv(os.path.join(args.out_dir, "intransit_history.csv"),
                      index=False)
            if sim.reservoir_history:
                pd.DataFrame(sim.reservoir_history).to_csv(
                    os.path.join(args.out_dir, "reservoir_history.csv"),
                    index=False)
            if sim.availability_history:
                pd.DataFrame(sim.availability_history).to_csv(
                    os.path.join(args.out_dir, "source_availability.csv"),
                    index=False)

    fr = svc['fill_rate_stock_only']
    print(f"\n=== Service Summary ===")
    print(f"  Fill rate: mean={fr.mean():.3f}  "
          f"median={fr.median():.3f}  "
          f"min={fr.min():.3f}  max={fr.max():.3f}")
    print(f"  Total demand:  {svc['total_demand'].sum():,}")
    print(f"  Total served:  {svc['served_from_stock'].sum():,}")
    print(f"  Total backlog: {svc['new_backlog_added'].sum():,}")
    print(f"\nOutputs → {args.out_dir}/")
