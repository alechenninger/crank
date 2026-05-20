# crank

**crank** ranks Kubernetes clusters by how much operator attention they deserve. It combines **state** (current health of nodes, pods, and workloads) with **events** (recent warnings, OOMs, evictions, scheduling failures) and layers **ML scoring** plus **keyword/area boosts** so teams can prioritize the right clusters first.

## Why hybrid state + events?

| Signal | Strength | Weakness |
|--------|----------|----------|
| **State** | Shows durable problems (CrashLoopBackOff, NotReady nodes, privileged pods) | Misses flapping or already-recovered incidents |
| **Events** | Captures incident velocity and recent failures | Noisy; decays without tuning |

crank merges both into a single feature vector, then scores with a weighted heuristic (no model required) or a trained **Gradient Boosting** regressor with **Isolation Forest** anomaly boost.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Rank demo clusters (no Kubernetes API needed)
crank rank --demo --format table

# Score your cluster (needs kubeconfig / in-cluster credentials)
crank score --cluster prod-us-east --context my-prod-context

# Rank multiple clusters
crank rank --clusters '{"prod-us":"ctx-prod","staging":"ctx-staging"}' --format json

# Train from operator labels (0–100 attention score per snapshot)
crank train --dataset examples/training_dataset.jsonl --output models/crank.joblib
```

Point `scoring.model_path` in `config/crank.yaml` at the trained model to blend ML with heuristics. Without a model, output uses `scoring_mode: heuristic` and a `base_score` from the weighted feature heuristic (not ML).

Config is auto-discovered from `./config/crank.yaml` or `~/.config/crank/config.yaml` when `--config` is omitted. YAML keyword rules are merged onto built-in defaults (same `pattern` overrides the default).

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ K8s API         │────▶│ FeatureExtractor │────▶│ ClusterScorer   │
│ (state+events)  │     │ (20 features)    │     │ ML + heuristic  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              ▼
│ searchable text │────▶│ KeywordMatcher   │────▶   ClusterRanker
│ (names, events) │     │ (area boosts)    │         (final rank)
└─────────────────┘     └──────────────────┘
```

### Attention areas

Keywords map to operational themes:

- **reliability** — crash loops, prod workloads, payments
- **security** — privileged pods, vault, hostPath
- **capacity** — OOM, eviction, failed scheduling
- **compliance** — PCI, HIPAA, certificate expiry
- **platform** — mesh, GitOps, observability stack

Customize patterns in `config/crank.yaml`.

### Features (ML input)

Ratios and rates include: node NotReady/pressure, pod failure/crash/privilege signals, pending age, event rates (warnings, OOM, evictions, backoff), and workload availability (Deployments, StatefulSets, DaemonSets).

## Configuration

See `config/crank.yaml`. Key knobs:

- `event_window_hours` — how far back to scan events (default 24h)
- `keywords` — substring patterns → area + weight
- `scoring.keyword_boost_cap` — max keyword add-on (default 25)
- `scoring.ml_weight` — blend trained model vs heuristic (default 0.7)

## Training with operator feedback

Export labeled snapshots as JSONL (one object per line):

```json
{"name": "prod-us", "label": 85, "nodes": {"total": 50, "not_ready": 2}, "pods": {"total": 800, "crash_loop_backoff": 4}, "deployments_total": 120, "deployments_unavailable": 5, "events": {"warnings": 80}, "searchable_text": ["prod payment oom"]}
```

`label` is how much attention the cluster deserved (0–100), e.g. from post-incident review or on-call triage. Training requires at least 10 labeled rows and prints validation MAE/R². Retrain periodically so the model reflects your environment.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
mypy
```

## License

MIT
