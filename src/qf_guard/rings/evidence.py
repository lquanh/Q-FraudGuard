"""Observable evidence for ring inference, plus the learned evidence models.

Only data a bank could observe at decision time is used: login sessions
(account, device fingerprint, IP, minute), transfers, geo-IP "foreign/VPN"
flags and reports received through channels such as SIMO.  Ground truth from
the synthetic world is used *only* to build training labels.

Two learned models turn evidence into a pairwise Markov random field:

* a node model ``P(account is operated by a ring | its own features)`` gives
  the per-account log-odds (field);
* a three-class pair model over ``{neither, exactly one, both}`` trained on
  *pair-only* evidence (shared devices/IPs, co-logins, money links, ...), so it
  does not re-count per-account evidence.  Dividing its posteriors by the class
  priors gives log-likelihood ratios ``L0, L1, L2`` of the pair evidence under
  each joint state; they enter the energy exactly (unary shifts on both
  endpoints plus the interaction ``J = (L2 - 2 L1 + L0) / 4``).  ``J > 0`` says
  "same state", ``J < 0`` says "one in, one out".

Every baseline consumes the same model outputs, so comparisons isolate the
inference method, not the features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from qf_guard.rings.synth import MINUTES_PER_DAY, World

__all__ = [
    "EvidenceConfig",
    "Evidence",
    "EvidenceModels",
    "EvidenceScores",
    "build_evidence",
    "fit_models",
    "score_evidence",
    "NODE_FEATURES",
    "PAIR_FEATURES",
]

NODE_FEATURES = [
    "sessions_recent",
    "dev_changed",
    "recent_dev_max_deg",
    "foreign_frac_recent",
    "unique_ips_recent",
    "unique_senders_recent",
    "inflow_recent",
    "forward_ratio_recent",
    "merchant_recent",
    "colog_partners",
]

_PAIR_BASE = [
    "n_shared_dev",
    "min_dev_deg",
    "n_shared_ip",
    "min_ip_deg",
    "shared_foreign",
    "recent_both_dev",
    "colog_dev",
    "colog_ip",
    "n_transfers",
    "n_forward",
    "common_receivers",
    "common_senders",
    "shared_old",
    "shared_recent",
    "divergence",
    "coactive_days",
]
_CROSS = [
    "unique_senders_recent",
    "forward_ratio_recent",
    "dev_changed",
    "foreign_frac_recent",
    "merchant_recent",
]
CROSS_FEATURES = [f"{p}_{c}" for c in _CROSS for p in ("min", "max")]
#: Pair-only evidence used by the MRF pair model (no per-account features).
PAIR_FEATURES = list(_PAIR_BASE)


@dataclass(frozen=True, slots=True)
class EvidenceConfig:
    recent_days: int = 21
    max_device_degree: int = 80
    max_ip_degree: int = 60
    max_counterparty_degree: int = 40
    colog_minutes: int = 10
    colog_lookahead: int = 4
    forward_minutes: int = 30
    coactive_minutes: int = 3


@dataclass(frozen=True, slots=True)
class Evidence:
    n: int
    minute: int
    node: pd.DataFrame
    pairs: pd.DataFrame
    seeds: np.ndarray
    seed_is_false: np.ndarray
    node_label: np.ndarray
    pair_label: np.ndarray


def _pairs_from_groups(df: pd.DataFrame, key: str, max_degree: int, value: str | None = None) -> pd.DataFrame:
    """All unordered account pairs sharing ``key`` (groups of size 2..max_degree)."""

    members = df[["account", key]].drop_duplicates()
    degree = members.groupby(key).account.transform("size")
    members = members[(degree >= 2) & (degree <= max_degree)].assign(deg=degree)
    merged = members.merge(members, on=key, suffixes=("_i", "_j"))
    merged = merged[merged.account_i < merged.account_j]
    return merged.rename(columns={"account_i": "i", "account_j": "j", "deg_i": "deg"})[
        ["i", "j", key, "deg"]
    ]


def _colog(sessions: pd.DataFrame, key: str, degree: pd.Series, cfg: EvidenceConfig) -> pd.DataFrame:
    s = sessions[sessions[key].map(degree).between(2, cfg.max_ip_degree if key == "ip" else cfg.max_device_degree)]
    s = s.sort_values([key, "minute"], kind="stable")
    keys, minutes, accounts = s[key].to_numpy(), s.minute.to_numpy(), s.account.to_numpy()
    out = []
    for k in range(1, cfg.colog_lookahead + 1):
        same = (keys[k:] == keys[:-k]) & (minutes[k:] - minutes[:-k] <= cfg.colog_minutes)
        diff = accounts[k:] != accounts[:-k]
        mask = same & diff
        a, b = accounts[:-k][mask], accounts[k:][mask]
        out.append(np.column_stack([np.minimum(a, b), np.maximum(a, b)]))
    if not out or sum(len(x) for x in out) == 0:
        return pd.DataFrame(columns=["i", "j", "count"])
    pairs = pd.DataFrame(np.vstack(out), columns=["i", "j"])
    return pairs.groupby(["i", "j"]).size().rename("count").reset_index()


def _coactivity(sessions: pd.DataFrame, i: np.ndarray, j: np.ndarray, window: int) -> np.ndarray:
    """Distinct days on which both accounts were active within ``window`` minutes."""

    by_account = {
        int(acc): np.sort(group.to_numpy()) for acc, group in sessions.groupby("account").minute
    }
    out = np.zeros(len(i))
    for k, (u, v) in enumerate(zip(i.tolist(), j.tolist())):
        a, b = by_account.get(u), by_account.get(v)
        if a is None or b is None:
            continue
        pos = np.searchsorted(b, a)
        left = np.abs(a - b[np.clip(pos - 1, 0, len(b) - 1)])
        right = np.abs(b[np.clip(pos, 0, len(b) - 1)] - a)
        close = np.minimum(left, right) <= window
        if close.any():
            out[k] = len(np.unique(a[close] // MINUTES_PER_DAY))
    return out


def build_evidence(world: World, cfg: EvidenceConfig | None = None, minute: int | None = None) -> Evidence:
    cfg = cfg or EvidenceConfig()
    n = world.n_accounts
    t = world.observe_minute if minute is None else int(minute)
    t_recent = t - cfg.recent_days * MINUTES_PER_DAY

    sessions = world.sessions[world.sessions.minute < t]
    transfers = world.transfers[world.transfers.minute < t]
    reports = world.reports[world.reports.minute < t]
    foreign = sessions.ip.isin(world.foreign_ips)

    dev_degree = sessions.groupby("device").account.nunique()
    ip_degree = sessions.groupby("ip").account.nunique()
    old = sessions[sessions.minute < t_recent]
    recent = sessions[sessions.minute >= t_recent]

    # ---- node features -------------------------------------------------------
    node = pd.DataFrame(index=pd.RangeIndex(n, name="account"))
    node["sessions_recent"] = np.log1p(recent.groupby("account").size().reindex(node.index, fill_value=0))

    def primary(frame: pd.DataFrame) -> pd.Series:
        counts = frame.groupby(["account", "device"]).size().rename("c").reset_index()
        idx = counts.sort_values(["account", "c"], ascending=[True, False]).drop_duplicates("account")
        return idx.set_index("account").device

    p_old, p_new = primary(old), primary(recent)
    both = p_old.index.intersection(p_new.index)
    changed = pd.Series(0.0, index=node.index)
    changed.loc[both] = (p_old.loc[both] != p_new.loc[both]).astype(float)
    node["dev_changed"] = changed

    first_seen = sessions.groupby(["account", "device"]).minute.min().rename("first").reset_index()
    first_seen["deg"] = first_seen.device.map(dev_degree)
    recent_devs = first_seen[first_seen["first"] >= t_recent]
    node["recent_dev_max_deg"] = np.log1p(recent_devs.groupby("account").deg.max().reindex(node.index, fill_value=0))

    rec_foreign = recent.ip.isin(world.foreign_ips)
    node["foreign_frac_recent"] = (
        rec_foreign.groupby(recent.account).mean().reindex(node.index, fill_value=0.0)
    )
    node["unique_ips_recent"] = np.log1p(recent.groupby("account").ip.nunique().reindex(node.index, fill_value=0))

    persons = transfers[(transfers.src < n) & (transfers.dst < n)]
    rec_tr = transfers[transfers.minute >= t_recent]
    rec_in = rec_tr[(rec_tr.dst < n) & (rec_tr.src < n)]
    node["unique_senders_recent"] = np.log1p(rec_in.groupby("dst").src.nunique().reindex(node.index, fill_value=0))
    node["inflow_recent"] = np.log1p(
        rec_tr[rec_tr.dst < n].groupby("dst").amount.sum().reindex(node.index, fill_value=0.0) / 1e6
    )

    # forwards: an outgoing transfer within `forward_minutes` after an incoming one
    incoming = transfers[transfers.dst < n][["dst", "minute"]].rename(columns={"dst": "account"})
    incoming = incoming.assign(in_minute=incoming.minute).sort_values("minute")
    outgoing = transfers[transfers.src < n][["src", "dst", "minute"]].rename(columns={"src": "account"})
    outgoing = outgoing.reset_index().sort_values("minute")
    matched = pd.merge_asof(
        outgoing,
        incoming[["account", "minute", "in_minute"]],
        on="minute",
        by="account",
        direction="backward",
        tolerance=cfg.forward_minutes,
        allow_exact_matches=True,
    )
    matched["is_forward"] = matched.in_minute.notna()
    rec_out = matched[matched.minute >= t_recent]
    node["forward_ratio_recent"] = rec_out.groupby("account").is_forward.mean().reindex(node.index, fill_value=0.0)
    merchant = rec_tr[(rec_tr.src < n) & (rec_tr.dst >= n) & (rec_tr.dst < n + world.config.n_merchants)]
    node["merchant_recent"] = np.log1p(merchant.groupby("src").size().reindex(node.index, fill_value=0))

    # ---- pair features ---------------------------------------------------------
    parts: list[pd.DataFrame] = []
    dev_pairs = _pairs_from_groups(sessions, "device", cfg.max_device_degree)
    if len(dev_pairs):
        g = dev_pairs.groupby(["i", "j"]).agg(n_shared_dev=("device", "nunique"), min_dev_deg=("deg", "min"))
        parts.append(g)
        recent_first = first_seen[first_seen["first"] >= t_recent][["account", "device"]]
        recent_set = set(zip(recent_first.account.to_numpy(), recent_first.device.to_numpy()))
        rb = dev_pairs[
            [
                (i, d) in recent_set and (j, d) in recent_set
                for i, j, d in zip(dev_pairs.i.to_numpy(), dev_pairs.j.to_numpy(), dev_pairs.device.to_numpy())
            ]
        ]
        if len(rb):
            parts.append(rb.groupby(["i", "j"]).size().rename("recent_both_dev").to_frame())

    ip_pairs = _pairs_from_groups(sessions.assign(foreign=foreign), "ip", cfg.max_ip_degree)
    if len(ip_pairs):
        ip_pairs["is_foreign"] = ip_pairs.ip.isin(world.foreign_ips)
        g = ip_pairs.groupby(["i", "j"]).agg(
            n_shared_ip=("ip", "nunique"), min_ip_deg=("deg", "min"), shared_foreign=("is_foreign", "max")
        )
        parts.append(g)

    for key, degree, name in (("device", dev_degree, "colog_dev"), ("ip", ip_degree, "colog_ip")):
        cl = _colog(sessions, key, degree, cfg)
        if len(cl):
            parts.append(cl.set_index(["i", "j"])["count"].rename(name).to_frame())

    if len(persons):
        tp = persons.assign(i=np.minimum(persons.src, persons.dst), j=np.maximum(persons.src, persons.dst))
        tp = tp[tp.i != tp.j]
        parts.append(tp.groupby(["i", "j"]).size().rename("n_transfers").to_frame())
        fw = matched[matched.is_forward & (matched.dst < n)]
        if len(fw):
            fw = fw.assign(i=np.minimum(fw.account, fw.dst), j=np.maximum(fw.account, fw.dst))
            parts.append(fw.groupby(["i", "j"]).size().rename("n_forward").to_frame())
        senders_to = persons[["src", "dst"]].drop_duplicates()
        cr = _pairs_from_groups(senders_to.rename(columns={"src": "account"}), "dst", cfg.max_counterparty_degree)
        if len(cr):
            parts.append(cr.groupby(["i", "j"]).size().rename("common_receivers").to_frame())
        cs = _pairs_from_groups(
            senders_to.rename(columns={"dst": "account", "src": "sender"}), "sender", cfg.max_counterparty_degree
        )
        if len(cs):
            parts.append(cs.groupby(["i", "j"]).size().rename("common_senders").to_frame())

    for frame, name in ((old, "shared_old"), (recent, "shared_recent")):
        sp = pd.concat(
            [
                _pairs_from_groups(frame, "device", cfg.max_device_degree)[["i", "j"]],
                _pairs_from_groups(frame, "ip", cfg.max_ip_degree)[["i", "j"]],
            ]
        )
        if len(sp):
            parts.append(sp.groupby(["i", "j"]).size().rename(name).to_frame())

    pairs = pd.concat(parts, axis=1).fillna(0.0).reset_index() if parts else pd.DataFrame(columns=["i", "j"])
    for column in _PAIR_BASE:
        if column not in pairs:
            pairs[column] = 0.0
    ci = node["dev_changed"].to_numpy()[pairs.i.to_numpy()]
    cj = node["dev_changed"].to_numpy()[pairs.j.to_numpy()]
    pairs["divergence"] = ((pairs.shared_old > 0) & (pairs.shared_recent == 0) & (ci != cj)).astype(float)
    pairs["coactive_days"] = _coactivity(
        recent, pairs.i.to_numpy(dtype=np.int64), pairs.j.to_numpy(dtype=np.int64), cfg.coactive_minutes
    )
    for column in ("n_shared_dev", "n_shared_ip", "recent_both_dev", "colog_dev", "colog_ip",
                   "n_transfers", "n_forward", "common_receivers", "common_senders",
                   "shared_old", "shared_recent", "min_dev_deg", "min_ip_deg", "coactive_days"):
        pairs[column] = np.log1p(pairs[column].astype(float))
    pairs["shared_foreign"] = pairs.shared_foreign.astype(float)
    for c in _CROSS:
        vi = node[c].to_numpy()[pairs.i.to_numpy()]
        vj = node[c].to_numpy()[pairs.j.to_numpy()]
        pairs[f"min_{c}"] = np.minimum(vi, vj)
        pairs[f"max_{c}"] = np.maximum(vi, vj)

    colog_counts = pd.concat(
        [pairs.loc[(pairs.colog_dev > 0) | (pairs.colog_ip > 0), "i"],
         pairs.loc[(pairs.colog_dev > 0) | (pairs.colog_ip > 0), "j"]]
    ).value_counts()
    node["colog_partners"] = np.log1p(colog_counts.reindex(node.index, fill_value=0))

    pairs = pairs.astype({"i": np.int64, "j": np.int64}).sort_values(["i", "j"]).reset_index(drop=True)

    active = world.active_ring(t)
    node_label = active.astype(np.int8)
    pair_label = (active[pairs.i.to_numpy()].astype(np.int8) + active[pairs.j.to_numpy()].astype(np.int8))
    seeds_df = reports.groupby("account").is_false.max()
    return Evidence(
        n=n,
        minute=t,
        node=node[NODE_FEATURES].astype(float),
        pairs=pairs[["i", "j", *PAIR_FEATURES, *CROSS_FEATURES]],
        seeds=seeds_df.index.to_numpy(dtype=np.int64),
        seed_is_false=seeds_df.to_numpy(dtype=bool),
        node_label=node_label,
        pair_label=pair_label.astype(np.int8),
    )


@dataclass(slots=True)
class EvidenceModels:
    node_model: HistGradientBoostingClassifier
    pair_model: HistGradientBoostingClassifier
    pair_prior: np.ndarray
    link_model: HistGradientBoostingClassifier
    llr_clip: float = 6.0


@dataclass(frozen=True, slots=True)
class EvidenceScores:
    """Model outputs for one world.

    ``L0, L1, L2`` are log-likelihood ratios of each pair's evidence under the
    joint states (neither, one specific endpoint, both) relative to the prior;
    ``J = (L2 - 2 L1 + L0) / 4`` is the Ising interaction.  ``p_both`` comes
    from a separate pair+account "link" classifier used by link-analysis
    baselines (union-find, PageRank, greedy growth).
    """

    n: int
    node_logit: np.ndarray
    node_prob: np.ndarray
    i: np.ndarray
    j: np.ndarray
    L0: np.ndarray
    L1: np.ndarray
    L2: np.ndarray
    J: np.ndarray
    p_both: np.ndarray
    seeds: np.ndarray


def _hgb(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed
    )


def fit_models(training: list[Evidence], seed: int = 0) -> EvidenceModels:
    X_node = pd.concat([e.node for e in training], ignore_index=True)
    y_node = np.concatenate([e.node_label for e in training])
    X_pair = pd.concat([e.pairs[PAIR_FEATURES] for e in training], ignore_index=True)
    X_link = pd.concat([e.pairs[PAIR_FEATURES + CROSS_FEATURES] for e in training], ignore_index=True)
    y_pair = np.concatenate([e.pair_label for e in training])
    prior = np.bincount(y_pair, minlength=3).astype(float) + 1.0
    return EvidenceModels(
        node_model=_hgb(seed).fit(X_node, y_node),
        pair_model=_hgb(seed).fit(X_pair, y_pair),
        pair_prior=prior / prior.sum(),
        link_model=_hgb(seed).fit(X_link, (y_pair == 2).astype(int)),
    )


def _class_matrix(model: HistGradientBoostingClassifier, X: pd.DataFrame, k: int) -> np.ndarray:
    proba = model.predict_proba(X)
    full = np.zeros((len(proba), k))
    for column, label in enumerate(model.classes_):
        full[:, int(label)] = proba[:, column]
    return full


def score_evidence(evidence: Evidence, models: EvidenceModels, eps: float = 1e-6) -> EvidenceScores:
    p_node = np.clip(models.node_model.predict_proba(evidence.node)[:, 1], eps, 1 - eps)
    if len(evidence.pairs):
        post = np.clip(_class_matrix(models.pair_model, evidence.pairs[PAIR_FEATURES], 3), eps, 1.0)
        post /= post.sum(axis=1, keepdims=True)
        llr = np.log(post / models.pair_prior[None, :])
        llr -= llr[:, [0]]  # relative to "neither": L0 = 0
        llr = np.clip(llr, -models.llr_clip, models.llr_clip)
        L0, L1, L2 = llr[:, 0], llr[:, 1], llr[:, 2]
        J = 0.25 * (L2 - 2.0 * L1 + L0)
        p_both = np.clip(
            _class_matrix(models.link_model, evidence.pairs[PAIR_FEATURES + CROSS_FEATURES], 2)[:, 1], 0.0, 1.0
        )
    else:
        L0 = L1 = L2 = J = p_both = np.zeros(0)
    return EvidenceScores(
        n=evidence.n,
        node_logit=np.log(p_node / (1 - p_node)),
        node_prob=p_node,
        i=evidence.pairs.i.to_numpy(dtype=np.int64),
        j=evidence.pairs.j.to_numpy(dtype=np.int64),
        L0=L0,
        L1=L1,
        L2=L2,
        J=J,
        p_both=p_both,
        seeds=evidence.seeds,
    )
