"""
Supply Chain Simulation — Extended (four physical-realism knobs).


  1. Supplier disruption  — Bernoulli-onset outages at each source, fixed
     length. While a source is down: reservoir doesn't refill, its (s,S)
     doesn't fire, and _replenish_warehouses cannot pull from it.

  2. Finite source supply — sources hold a per-item raw-material reservoir
     that refills at production_rate/day and caps at reservoir_cap. When
     the source's (s,S) fires, the outbound order is capped at the
     reservoir level, so shortages propagate downstream.

  3. Finite warehouse cap — each non-source, non-destination node has an
     optional total volumetric capacity. `_replenish_warehouses` caps the
     order quantity at (cap - current_used_vol - committed_incoming_vol)
     / item.volume, so shipments never overshoot capacity.

  4. Stochastic edge transit — per-shipment multiplicative Gaussian noise
     on the deterministic edge-sum: sampled_tt = base_tt * (1 + N(0,std)),
     clipped to >=1 day.

Setting {disruption_p_onset=0, prod_rate large, warehouse_cap_scale=None,
edge_tt_std_frac=0} reproduces the base sim
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
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

        self._source_ids = [nid for nid, n in self.network.nodes.items()
                            if n.is_source]
        self._reservoir_source_ids = [
            nid for nid in self._source_ids
            if isinstance(self.network.nodes[nid], ReservoirNode)]

        # Precompute per-source availability (uses self.rng so runs are
        # reproducible per seed).
        self.source_available = _generate_outage_schedule(
            self.rng, self.horizon_days, self._source_ids,
            self.disruption_p_onset, self.disruption_length_days)

        # Summary printout
        for nid, a in self.source_available.items():
            down = int((~a).sum())
            flips = int(np.sum(np.diff(a.astype(int)) < 0))
            n_events = flips + (1 if (not bool(a[0])) else 0)
            print(f"    disruption[{nid}]: {down} down-days "
                  f"across {n_events} outage windows")

        self.reservoir_history: List[dict] = []
        self.availability_history: List[dict] = []

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
    # Extension 2 knobs
    prod_rate_per_item: float = 100.0,
    reservoir_cap_per_item: float = 5000.0,
    init_frac: float = 1.0,
    # Extension 3 knob
    warehouse_cap_scale: Optional[float] = None,
    # Extension 1 knobs
    disruption_p_onset: float = 0.0,
    disruption_length_days: int = 0,
    # Extension 4 knob
    edge_tt_std_frac: float = 0.0,
):
    base_sim, net, items, demand_signals = \
        build_example_simulation_from_adjacency(
            seed=seed,
            horizon_days=horizon_days,
            pipeline_multiplier=pipeline_multiplier,
            streaming_out_dir=None,
            packing=packing,
            scenario=scenario)

    item_ids = sorted(items.keys())

    install_reservoirs(
        net, item_ids,
        prod_rate_per_item=prod_rate_per_item,
        reservoir_cap_per_item=reservoir_cap_per_item,
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

    n_sources = len(
        [n for n in net.nodes.values() if n.is_source])
    total_prod = prod_rate_per_item * len(item_ids) * n_sources
    mean_demand = float(demand_signals.mean() * len(item_ids))
    print(f"  Sources: {n_sources}  reservoir cap={reservoir_cap_per_item:.0f} "
          f"prod={prod_rate_per_item:.1f}/item/day/src")
    print(f"    aggregate prod={total_prod:.0f}/day vs "
          f"demand={mean_demand:.0f}/day "
          f"({total_prod/max(mean_demand,1e-9):.2f}x)")
    if warehouse_cap:
        caps_str = ", ".join(f"{k}={v:.0f}" for k, v in warehouse_cap.items())
        print(f"  Warehouse caps (vol, scale={warehouse_cap_scale}): {caps_str}")
    else:
        print(f"  Warehouse caps: unlimited")
    print(f"  Disruption: p_onset={disruption_p_onset} "
          f"length={disruption_length_days}d  "
          f"Edge tt noise: std_frac={edge_tt_std_frac}")

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
                    "(disruption + reservoir + warehouse cap + tt noise)")
    ap.add_argument("--days",          type=int,   default=7300)
    ap.add_argument("--seed",          type=int,   default=2025)
    ap.add_argument("--out_dir",       type=str,   default="test_output_ext")
    ap.add_argument("--pipeline_mult", type=float, default=0.0)
    ap.add_argument("--no_streaming",  action="store_true")

    # Extension 2 (reservoir)
    ap.add_argument("--prod_rate_per_item",     type=float, default=100.0)
    ap.add_argument("--reservoir_cap_per_item", type=float, default=5000.0)
    ap.add_argument("--init_frac",              type=float, default=1.0)

    # Extension 3 (warehouse cap). None → unlimited.
    ap.add_argument("--warehouse_cap_scale",    type=float, default=None,
                    help="Per-node total-volume cap as multiple of "
                         "target-inventory volume (sum S_iid * volume). "
                         "Omit for unlimited.")

    # Extension 1 (supplier disruption)
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

    # Baseline scenario knobs (unchanged from base file)
    ap.add_argument("--phi_lo",             type=float, default=0.9990)
    ap.add_argument("--phi_hi",             type=float, default=0.9996)
    ap.add_argument("--shock_count_scale",  type=float, default=1.0)
    ap.add_argument("--shock_height_scale", type=float, default=1.0)
    ap.add_argument("--seasonal_scale",     type=float, default=1.0)
    ap.add_argument("--containers_scale",   type=float, default=1.0)
    ap.add_argument("--ss_scale",           type=float, default=1.0)
    ap.add_argument("--leadtime_scale",     type=float, default=1.0)
    ap.add_argument("--burst_rate_scale",   type=float, default=1.0)
    ap.add_argument("--burst_height_scale", type=float, default=1.0)
    ap.add_argument("--base_lambda_lo",     type=float, default=80.0)
    ap.add_argument("--base_lambda_hi",     type=float, default=250.0)
    ap.add_argument("--scenario_name",      type=str,   default="baseline_ext")
    args = ap.parse_args()

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
        # Ext knobs also recorded for provenance
        "prod_rate_per_item": args.prod_rate_per_item,
        "reservoir_cap_per_item": args.reservoir_cap_per_item,
        "init_frac": args.init_frac,
        "warehouse_cap_scale": args.warehouse_cap_scale,
        "disruption_p_onset": args.disruption_p_onset,
        "disruption_length_days": args.disruption_length_days,
        "edge_tt_std_frac": args.edge_tt_std_frac,
    }

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
