"""Synthetic bank world with rented/bought mule-account rings.

Everything in this module is SYNTHETIC.  It exists only to test the
ring-inference idea under controlled noise:

* does a globally consistent (Ising) decision flag fewer innocent accounts than
  simpler baselines at the same recall, and
* after classical preprocessing, is the remaining frustrated core large enough
  to justify an Ising / quantum solver at all?

It is not evidence about real banks.  The story it encodes (rings buying or
renting accounts in real people's names, often students, then operating them
from shared devices and foreign IPs) follows public Vietnamese police and bank
warnings, but every rate below is an assumption exposed in :class:`WorldConfig`
so that results can be swept over it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 1440

__all__ = ["MINUTES_PER_DAY", "WorldConfig", "World", "generate_world", "NOISE_PRESETS"]


@dataclass(frozen=True, slots=True)
class WorldConfig:
    """All generator assumptions.  Rates are per account unless stated."""

    n_persons: int = 2000
    days: int = 60
    observe_day: int = 45

    # legitimate social structure
    household_mean: float = 2.4
    student_frac: float = 0.25
    dorm_size: tuple[int, int] = (15, 55)
    work_frac: float = 0.6
    workplace_size: tuple[int, int] = (5, 50)
    n_cafes: int = 25
    cafe_visit_prob: float = 0.25
    n_nat_pools: int = 10
    shared_device_prob: float = 0.15
    new_phone_prob: float = 0.06
    activity_range: tuple[float, float] = (0.6, 2.0)
    n_merchants: int = 60

    # rings
    n_rings: int = 8
    ring_size: tuple[int, int] = (6, 40)
    ring_student_frac: float = 0.7
    friends_sell_together: float = 0.5
    operator_devices: tuple[int, int] = (2, 5)
    device_spoof_prob: float = 0.3
    operator_ips: tuple[int, int] = (2, 6)
    residential_proxy_prob: float = 0.3
    remote_control_prob: float = 0.2
    owner_keeps_using_prob: float = 0.2
    operator_session_prob: float = 0.7
    front_frac: float = 0.4
    victims_per_front: tuple[int, int] = (1, 8)
    report_prob: float = 0.5
    report_delay_days: tuple[int, int] = (0, 5)
    false_report_rate: float = 0.03

    def with_overrides(self, **changes: Any) -> "WorldConfig":
        return replace(self, **changes)


#: Noise presets used by the validation script.  "realistic" is the default.
NOISE_PRESETS: dict[str, dict[str, Any]] = {
    "clean": dict(
        device_spoof_prob=0.0,
        residential_proxy_prob=0.0,
        remote_control_prob=0.0,
        shared_device_prob=0.08,
        new_phone_prob=0.03,
        owner_keeps_using_prob=0.1,
        friends_sell_together=0.4,
    ),
    "realistic": {},
    "hard": dict(
        device_spoof_prob=0.6,
        residential_proxy_prob=0.6,
        remote_control_prob=0.6,
        shared_device_prob=0.25,
        new_phone_prob=0.12,
        owner_keeps_using_prob=0.4,
        friends_sell_together=0.7,
        cafe_visit_prob=0.35,
    ),
}


@dataclass(frozen=True, slots=True)
class World:
    config: WorldConfig
    seed: int
    n_accounts: int
    sessions: pd.DataFrame
    transfers: pd.DataFrame
    reports: pd.DataFrame
    foreign_ips: frozenset[int]
    ring_of: np.ndarray
    handover_minute: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def observe_minute(self) -> int:
        return int(self.config.observe_day * MINUTES_PER_DAY)

    def active_ring(self, minute: int | None = None) -> np.ndarray:
        """Accounts already operated by a ring at ``minute`` (ground truth)."""

        t = self.observe_minute if minute is None else minute
        return (self.ring_of >= 0) & (self.handover_minute < t)


class _Ids:
    def __init__(self) -> None:
        self.next = 0

    def take(self, count: int = 1) -> np.ndarray:
        values = np.arange(self.next, self.next + count, dtype=np.int64)
        self.next += count
        return values


def _partition(indices: np.ndarray, size_range: tuple[int, int], rng: np.random.Generator):
    groups = np.full(indices.max() + 1 if len(indices) else 0, -1, dtype=np.int64)
    position, group = 0, 0
    while position < len(indices):
        size = int(rng.integers(size_range[0], size_range[1] + 1))
        groups[indices[position : position + size]] = group
        position += size
        group += 1
    return groups, group


def generate_world(config: WorldConfig | None = None, seed: int = 0) -> World:
    cfg = config or WorldConfig()
    rng = np.random.default_rng(seed)
    n, days = cfg.n_persons, cfg.days
    horizon = days * MINUTES_PER_DAY

    # ---- legitimate structure -------------------------------------------------
    sizes: list[int] = []
    while sum(sizes) < n:
        sizes.append(int(min(5, 1 + rng.poisson(cfg.household_mean - 1))))
    household = np.repeat(np.arange(len(sizes)), sizes)[:n]
    n_households = int(household.max()) + 1

    is_student = rng.random(n) < cfg.student_frac
    students = rng.permutation(np.flatnonzero(is_student))
    dorm = np.full(n, -1, dtype=np.int64)
    if len(students):
        groups, n_dorms = _partition(students, cfg.dorm_size, rng)
        dorm[students] = groups[students]
    else:
        n_dorms = 0
    is_worker = (~is_student) & (rng.random(n) < cfg.work_frac)
    workers = rng.permutation(np.flatnonzero(is_worker))
    workplace = np.full(n, -1, dtype=np.int64)
    if len(workers):
        groups, n_workplaces = _partition(workers, cfg.workplace_size, rng)
        workplace[workers] = groups[workers]
    else:
        n_workplaces = 0

    ips = _Ids()
    home_ip = ips.take(n_households)[household]
    dorm_ips = ips.take(max(n_dorms, 1))
    work_ips = ips.take(max(n_workplaces, 1))
    cafe_ips = ips.take(cfg.n_cafes)
    nat_ips = ips.take(cfg.n_nat_pools)
    person_nat = nat_ips[rng.integers(0, cfg.n_nat_pools, n)]
    visits_cafe = rng.random(n) < cfg.cafe_visit_prob
    person_cafe = np.where(visits_cafe, cafe_ips[rng.integers(0, cfg.n_cafes, n)], -1)

    devices = _Ids()
    phone = devices.take(n)
    new_phone_day = np.where(
        rng.random(n) < cfg.new_phone_prob, rng.integers(5, days - 5, n), -1
    )
    phone2 = np.where(new_phone_day >= 0, devices.take(n), -1)
    tablet = np.full(n, -1, dtype=np.int64)
    for h in range(n_households):
        members = np.flatnonzero(household == h)
        if len(members) >= 2 and rng.random() < cfg.shared_device_prob:
            users = rng.choice(members, size=min(len(members), int(rng.integers(2, 4))), replace=False)
            tablet[users] = devices.take(1)[0]

    # ---- rings: recruitment and handover -------------------------------------
    ring_of = np.full(n, -1, dtype=np.int64)
    handover_minute = np.full(n, np.inf)
    keeps_using = np.zeros(n, dtype=bool)
    spoof = rng.random(n) < cfg.device_spoof_prob
    remote = rng.random(n) < cfg.remote_control_prob
    ring_devices: list[np.ndarray] = []
    ring_ips: list[np.ndarray] = []
    ring_proxied: list[bool] = []
    foreign: set[int] = set()
    for r in range(cfg.n_rings):
        size = int(rng.integers(cfg.ring_size[0], cfg.ring_size[1] + 1))
        members: list[int] = []
        recruit_dorms: list[int] = []
        attempts = 0
        while len(members) < size and attempts < size * 50:
            attempts += 1
            same_dorm = recruit_dorms and rng.random() < cfg.friends_sell_together
            if same_dorm:
                d = int(rng.choice(recruit_dorms))
                pool = np.flatnonzero((dorm == d) & (ring_of < 0))
            elif rng.random() < cfg.ring_student_frac:
                pool = np.flatnonzero(is_student & (ring_of < 0))
            else:
                pool = np.flatnonzero((~is_student) & (ring_of < 0))
            if len(pool) == 0:
                continue
            person = int(rng.choice(pool))
            ring_of[person] = r
            members.append(person)
            if dorm[person] >= 0 and dorm[person] not in recruit_dorms:
                recruit_dorms.append(int(dorm[person]))
        start_day = int(rng.integers(3, max(4, cfg.observe_day - 20)))
        for person in members:
            day = int(rng.integers(start_day, min(days - 2, cfg.observe_day + 10)))
            handover_minute[person] = day * MINUTES_PER_DAY + int(rng.integers(0, MINUTES_PER_DAY))
            keeps_using[person] = rng.random() < cfg.owner_keeps_using_prob
        ring_devices.append(devices.take(int(rng.integers(cfg.operator_devices[0], cfg.operator_devices[1] + 1))))
        operator_ips = ips.take(int(rng.integers(cfg.operator_ips[0], cfg.operator_ips[1] + 1)))
        ring_ips.append(operator_ips)
        foreign.update(int(ip) for ip in operator_ips)
        ring_proxied.append(bool(rng.random() < cfg.residential_proxy_prob))

    # ---- legitimate sessions --------------------------------------------------
    activity = rng.uniform(*cfg.activity_range, n)
    counts = rng.poisson(activity[:, None], size=(n, days))
    rows = np.repeat(np.arange(n * days), counts.ravel())
    person = rows // days
    day = rows % days
    minute = day * MINUTES_PER_DAY + rng.integers(7 * 60, 23 * 60, len(rows))
    device = np.where((new_phone_day[person] >= 0) & (day >= new_phone_day[person]), phone2[person], phone[person])
    use_tablet = (tablet[person] >= 0) & (rng.random(len(rows)) < 0.12)
    device = np.where(use_tablet, tablet[person], device)

    u = rng.random(len(rows))
    student, worker = is_student[person], is_worker[person]
    cafe = person_cafe[person]
    ip = home_ip[person].copy()
    # students: dorm 60%, mobile 25%, cafe 10%, home 5%
    s_dorm = student & (u < 0.60)
    s_nat = student & (u >= 0.60) & (u < 0.85)
    s_cafe = student & (u >= 0.85) & (u < 0.95)
    ip = np.where(s_dorm, dorm_ips[np.maximum(dorm[person], 0)], ip)
    ip = np.where(s_nat, person_nat[person], ip)
    ip = np.where(s_cafe & (cafe >= 0), cafe, np.where(s_cafe, person_nat[person], ip))
    # workers: home 45%, work 30%, mobile 20%, cafe 5%
    w_work = worker & (u >= 0.45) & (u < 0.75)
    w_nat = worker & (u >= 0.75) & (u < 0.95)
    w_cafe = worker & (u >= 0.95)
    ip = np.where(w_work, work_ips[np.maximum(workplace[person], 0)], ip)
    ip = np.where(w_nat, person_nat[person], ip)
    ip = np.where(w_cafe & (cafe >= 0), cafe, np.where(w_cafe, person_nat[person], ip))
    # others: home 70%, mobile 25%, cafe 5%
    other = ~(student | worker)
    o_nat = other & (u >= 0.70) & (u < 0.95)
    o_cafe = other & (u >= 0.95)
    ip = np.where(o_nat, person_nat[person], ip)
    ip = np.where(o_cafe & (cafe >= 0), cafe, np.where(o_cafe, person_nat[person], ip))

    # owners stop using sold accounts (unless they keep sharing it)
    sold_after = minute >= handover_minute[person]
    keep = (~sold_after) | (keeps_using[person] & (rng.random(len(rows)) < 0.3))
    legit_sessions = pd.DataFrame(
        {"account": person[keep], "device": device[keep], "ip": ip[keep], "minute": minute[keep]}
    )

    # ---- operator sessions ----------------------------------------------------
    op_rows: list[tuple[int, int, int, int]] = []
    sticky_device = np.full(n, -1, dtype=np.int64)
    spoof_device = np.where(spoof, devices.take(n), -1)
    for r in range(cfg.n_rings):
        members = np.flatnonzero(ring_of == r)
        for person_id in members:
            sticky_device[person_id] = int(rng.choice(ring_devices[r]))
        first_day = int(min(handover_minute[members]) // MINUTES_PER_DAY) if len(members) else days
        for d in range(first_day, days):
            shift = d * MINUTES_PER_DAY + int(rng.integers(8 * 60, 22 * 60))
            active = members[handover_minute[members] < (d + 1) * MINUTES_PER_DAY]
            order = rng.permutation(active)
            offset = 0
            for person_id in order:
                if rng.random() >= cfg.operator_session_prob:
                    continue
                offset += int(rng.integers(1, 5))
                when = shift + offset
                if when < handover_minute[person_id]:
                    continue
                if remote[person_id]:
                    # remote-access tool on the owner's own phone: the bank sees the
                    # owner's device and the owner's usual networks
                    own_phone = phone2[person_id] if 0 <= new_phone_day[person_id] <= d else phone[person_id]
                    dev = int(own_phone)
                    usual = [home_ip[person_id], person_nat[person_id]]
                    if dorm[person_id] >= 0:
                        usual.append(dorm_ips[dorm[person_id]])
                    op_ip = int(rng.choice(usual))
                    op_rows.append((int(person_id), dev, op_ip, int(when)))
                    continue
                dev = spoof_device[person_id] if spoof[person_id] else sticky_device[person_id]
                if ring_proxied[r]:
                    op_ip = int(ips.take(1)[0])
                else:
                    op_ip = int(rng.choice(ring_ips[r]))
                op_rows.append((int(person_id), int(dev), op_ip, int(when)))
    operator_sessions = pd.DataFrame(op_rows, columns=["account", "device", "ip", "minute"])
    sessions = (
        pd.concat([legit_sessions, operator_sessions], ignore_index=True)
        .sort_values("minute", kind="stable")
        .reset_index(drop=True)
    )
    sessions = sessions[sessions.minute < horizon].reset_index(drop=True)

    # ---- transfers --------------------------------------------------------------
    transfer_rows: list[np.ndarray] = []

    def add(src, dst, amount, when):
        transfer_rows.append(
            np.column_stack([np.asarray(src), np.asarray(dst), np.asarray(amount), np.asarray(when)])
        )

    def owner_active(account: np.ndarray, when: np.ndarray) -> np.ndarray:
        return (when < handover_minute[account]) | keeps_using[account]

    # legitimate contacts
    contact_src: list[int] = []
    contact_dst: list[int] = []
    contact_rate: list[float] = []
    for p in range(n):
        same_house = np.flatnonzero(household == household[p])
        for q in same_house:
            if q != p:
                contact_src.append(p), contact_dst.append(int(q)), contact_rate.append(3.0)
        mates_pool = np.flatnonzero(
            ((dorm == dorm[p]) & (dorm[p] >= 0)) | ((workplace == workplace[p]) & (workplace[p] >= 0))
        )
        mates_pool = mates_pool[mates_pool != p]
        if len(mates_pool):
            for q in rng.choice(mates_pool, size=min(3, len(mates_pool)), replace=False):
                contact_src.append(p), contact_dst.append(int(q)), contact_rate.append(1.5)
        for q in rng.integers(0, n, int(rng.integers(1, 3))):
            if q != p:
                contact_src.append(p), contact_dst.append(int(q)), contact_rate.append(0.8)
    contact_src_a = np.asarray(contact_src)
    contact_dst_a = np.asarray(contact_dst)
    per_contact = rng.poisson(np.asarray(contact_rate))
    src = np.repeat(contact_src_a, per_contact)
    dst = np.repeat(contact_dst_a, per_contact)
    when = rng.integers(0, horizon, len(src))
    amount = np.round(rng.lognormal(np.log(4e5), 0.8, len(src)), -3)
    ok = owner_active(src, when)
    add(src[ok], dst[ok], amount[ok], when[ok])

    # merchant payments ("owner life" activity)
    merchants = np.arange(n, n + cfg.n_merchants)
    m_counts = rng.poisson(0.45 * activity[:, None], size=(n, days))
    m_rows = np.repeat(np.arange(n * days), m_counts.ravel())
    m_src = m_rows // days
    m_when = (m_rows % days) * MINUTES_PER_DAY + rng.integers(7 * 60, 23 * 60, len(m_rows))
    m_dst = merchants[rng.integers(0, cfg.n_merchants, len(m_rows))]
    m_amount = np.round(rng.lognormal(np.log(1.5e5), 0.7, len(m_rows)), -3)
    ok = owner_active(m_src, m_when)
    add(m_src[ok], m_dst[ok], m_amount[ok], m_when[ok])

    # scam victims, layering and cash-out
    exit_account = n + cfg.n_merchants
    victim_pool = np.flatnonzero(ring_of < 0)
    report_rows: list[tuple[int, int, int]] = []
    role = np.zeros(n, dtype=np.int8)  # 0 legit, 1 front, 2 layer, 3 collector
    first_use = np.full(n, np.inf)  # first minute the account moved scam money

    def used(account: int, when: int) -> None:
        first_use[account] = min(first_use[account], when)
    for r in range(cfg.n_rings):
        members = rng.permutation(np.flatnonzero(ring_of == r))
        if len(members) == 0:
            continue
        n_fronts = max(1, int(round(cfg.front_frac * len(members))))
        fronts, layers = members[:n_fronts], members[n_fronts:]
        collectors = layers[: max(1, len(layers) // 8)] if len(layers) else np.array([], dtype=np.int64)
        role[fronts] = 1
        role[layers] = 2
        role[collectors] = 3
        for front in fronts:
            n_victims = int(rng.integers(cfg.victims_per_front[0], cfg.victims_per_front[1] + 1))
            start = int(handover_minute[front])
            if start >= horizon - 60:
                continue
            for _ in range(n_victims):
                victim = int(rng.choice(victim_pool))
                t0 = int(rng.integers(start, horizon - 60))
                value = float(np.round(rng.uniform(2e6, 5e7), -3))
                add([victim], [front], [value], [t0])
                used(int(front), t0)
                if rng.random() < cfg.report_prob:
                    delay = int(rng.integers(cfg.report_delay_days[0], cfg.report_delay_days[1] + 1))
                    report_rows.append((int(front), t0 + delay * MINUTES_PER_DAY + int(rng.integers(0, 600)), 0))
                live_layers = layers[handover_minute[layers] < t0] if len(layers) else layers
                t1 = t0 + int(rng.integers(5, 40))
                if len(live_layers) == 0:
                    add([front], [exit_account], [value * 0.97], [t1])
                    continue
                hops = rng.choice(live_layers, size=min(len(live_layers), int(rng.integers(1, 3))), replace=False)
                for hop in hops:
                    part = value * 0.95 / len(hops)
                    add([front], [int(hop)], [part], [t1])
                    used(int(hop), t1)
                    live_collectors = collectors[handover_minute[collectors] < t1] if len(collectors) else collectors
                    t2 = t1 + int(rng.integers(5, 60))
                    if len(live_collectors) and hop not in live_collectors:
                        collector = int(rng.choice(live_collectors))
                        add([int(hop)], [collector], [part * 0.98], [t2])
                        used(collector, t2)
                        add([collector], [exit_account], [part * 0.96], [t2 + int(rng.integers(60, 600))])
                    else:
                        add([int(hop)], [exit_account], [part * 0.97], [t2])

    transfers = pd.DataFrame(np.vstack(transfer_rows), columns=["src", "dst", "amount", "minute"])
    transfers = transfers.astype({"src": np.int64, "dst": np.int64, "amount": float, "minute": np.int64})
    transfers = transfers[transfers.minute < horizon].sort_values("minute", kind="stable").reset_index(drop=True)

    # false reports: legitimate accounts reported by mistake
    true_reports = pd.DataFrame(report_rows, columns=["account", "minute", "is_false"])
    n_true = int(true_reports[true_reports.minute < cfg.observe_day * MINUTES_PER_DAY].account.nunique())
    n_false = int(rng.binomial(max(n_true, 1), cfg.false_report_rate))
    legit_ids = np.flatnonzero(ring_of < 0)
    false_ids = rng.choice(legit_ids, size=min(n_false, len(legit_ids)), replace=False)
    false_reports = pd.DataFrame(
        {
            "account": false_ids,
            "minute": rng.integers(0, cfg.observe_day * MINUTES_PER_DAY, len(false_ids)),
            "is_false": 1,
        }
    )
    reports = pd.concat([true_reports, false_reports], ignore_index=True).sort_values("minute").reset_index(drop=True)

    meta = {
        "household": household,
        "dorm": dorm,
        "workplace": workplace,
        "is_student": is_student,
        "role": role,
        "first_use_minute": first_use,
        "spoof": spoof,
        "remote": remote,
        "keeps_using": keeps_using,
        "ring_proxied": np.asarray(ring_proxied),
        "config": asdict(cfg),
    }
    return World(
        config=cfg,
        seed=seed,
        n_accounts=n,
        sessions=sessions,
        transfers=transfers,
        reports=reports,
        foreign_ips=frozenset(foreign),
        ring_of=ring_of,
        handover_minute=handover_minute,
        meta=meta,
    )
