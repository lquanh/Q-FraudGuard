import numpy as np
import pytest

from qf_guard.rings import baselines as B
from qf_guard.rings.core_solvers import brute_force, local_descent, milp_exact, simulated_annealing
from qf_guard.rings.evidence import build_evidence, fit_models, score_evidence
from qf_guard.rings.inference import explain_account, infer_ring
from qf_guard.rings.model import RingQUBO, build_ring_qubo
from qf_guard.rings.reduce import Core, Reducer, solve_with_reductions
from qf_guard.rings.synth import WorldConfig, generate_world


def random_model(rng, n, density=0.5, n_fixed=2):
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < density]
    if not pairs:
        pairs = [(0, 1)]
    i = np.array([p[0] for p in pairs])
    j = np.array([p[1] for p in pairs])
    fixed = {int(k): 1 for k in rng.choice(n, min(n_fixed, n), replace=False)}
    return RingQUBO(n=n, a=rng.normal(0, 2, n), i=i, j=j, b=rng.normal(0, 2, len(i)), const=0.0, fixed=fixed)


def brute_optimum(model: RingQUBO) -> float:
    X = ((np.arange(2**model.n)[:, None] >> np.arange(model.n)) & 1)
    ok = np.ones(len(X), dtype=bool)
    for k, v in model.fixed.items():
        ok &= X[:, k] == v
    return float(model.energy_many(X[ok]).min())


@pytest.mark.parametrize("seed", range(40))
def test_reductions_preserve_global_optimum(seed):
    rng = np.random.default_rng(seed)
    model = random_model(rng, n=int(rng.integers(3, 12)), density=float(rng.uniform(0.2, 0.9)),
                         n_fixed=int(rng.integers(0, 3)))
    x, stats, _ = solve_with_reductions(model, brute_force)
    assert model.respects_fixed(x)
    assert model.energy(x) == pytest.approx(brute_optimum(model), abs=1e-7)


def test_roof_duality_solves_unfrustrated_models_completely():
    # attractive (b < 0) couplings only: submodular, so the LP is integral
    rng = np.random.default_rng(3)
    n = 30
    i, j = np.triu_indices(n, 1)
    keep = rng.random(len(i)) < 0.2
    model = RingQUBO(n=n, a=rng.normal(0, 1, n), i=i[keep], j=j[keep], b=-rng.uniform(0.1, 2, keep.sum()),
                     const=0.0, fixed={})
    reducer = Reducer(model)
    reducer.roof_duality()
    assert not reducer.free.any()


def test_frustrated_triangle_is_left_as_core():
    # three mutually repulsive accounts that all want to be flagged: no LP/elimination shortcut
    model = RingQUBO(n=3, a=np.full(3, -1.0), i=np.array([0, 0, 1]), j=np.array([1, 2, 2]),
                     b=np.full(3, 2.0), const=0.0, fixed={})
    reducer = Reducer(model)
    reducer.roof_duality()
    assert reducer.free.sum() == 3


def test_core_solvers_agree_with_brute_force():
    rng = np.random.default_rng(7)
    for _ in range(10):
        n = 12
        B_ = rng.normal(0, 2, (n, n)) * (rng.random((n, n)) < 0.5)
        B_ = np.triu(B_, 1)
        B_ = B_ + B_.T
        core = Core(variables=np.arange(n), a=rng.normal(0, 2, n), B=B_)
        best = core.energy(brute_force(core))
        assert core.energy(simulated_annealing(core, sweeps=300, restarts=16)) == pytest.approx(best, abs=1e-7)
        res = milp_exact(core, time_limit=10)
        assert res.optimal and res.objective == pytest.approx(best, abs=1e-6)
        x = local_descent(core, np.zeros(n, dtype=np.int8))
        assert core.energy(x) >= best - 1e-9


def test_qubo_matches_pairwise_likelihood_table():
    # a single pair with L = (0, L1, L2): energies must equal -kappa * L for (00, 10, 11)
    from qf_guard.rings.evidence import EvidenceScores

    L1, L2, kappa = 1.3, -0.4, 0.7
    scores = EvidenceScores(n=2, node_logit=np.zeros(2), node_prob=np.full(2, 0.5), i=np.array([0]),
                            j=np.array([1]), L0=np.zeros(1), L1=np.array([L1]), L2=np.array([L2]),
                            J=np.array([(L2 - 2 * L1) / 4]), p_both=np.zeros(1), seeds=np.array([], dtype=int))
    model = build_ring_qubo(scores, kappa=kappa, beta0=0.0)
    assert model.energy(np.array([0, 0])) == pytest.approx(0.0)
    assert model.energy(np.array([1, 0])) == pytest.approx(-kappa * L1)
    assert model.energy(np.array([1, 1])) == pytest.approx(-kappa * L2)


@pytest.fixture(scope="module")
def small_world_scores():
    cfg = WorldConfig(n_persons=600, n_rings=3, ring_size=(5, 15), n_cafes=10)
    train = [build_evidence(generate_world(cfg, seed=s)) for s in (11, 12)]
    models = fit_models(train)
    world = generate_world(cfg, seed=13)
    evidence = build_evidence(world)
    return world, evidence, score_evidence(evidence, models)


def test_generator_is_deterministic():
    cfg = WorldConfig(n_persons=300, n_rings=2, ring_size=(4, 8), n_cafes=5)
    a, b = generate_world(cfg, seed=5), generate_world(cfg, seed=5)
    assert a.sessions.equals(b.sessions) and a.transfers.equals(b.transfers)
    assert np.array_equal(a.ring_of, b.ring_of)


def test_evidence_uses_only_data_before_decision_time(small_world_scores):
    world, evidence, _ = small_world_scores
    assert evidence.minute == world.observe_minute
    assert set(evidence.seeds.tolist()) <= set(world.reports[world.reports.minute < evidence.minute].account)


def test_inference_keeps_seeds_and_explains(small_world_scores):
    _, evidence, scores = small_world_scores
    result = infer_ring(scores, kappa=0.1, beta0=0.0)
    assert np.all(result.x[evidence.seeds] == 1)
    flagged = [int(k) for k in result.flagged if k not in set(evidence.seeds.tolist())]
    if flagged:
        rows = explain_account(result, flagged[0])
        assert all({"other", "J", "supports_decision"} <= set(r) for r in rows)


def test_baselines_always_include_seeds(small_world_scores):
    _, evidence, scores = small_world_scores
    for flags in (B.node_threshold(scores, 0.99), B.link_boost(scores, 0.99), B.union_find(scores, 0.99),
                  B.greedy_growth(scores, 0.99)):
        assert flags[evidence.seeds].all()
