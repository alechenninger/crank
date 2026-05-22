# crank

**crank** ranks Kubernetes clusters by how much operator attention they deserve. It combines **state** (current health of nodes, pods, and workloads) with **events** (recent warnings, OOMs, evictions, scheduling failures) and layers **ML scoring** plus **keyword/area boosts** so teams can prioritize the right clusters first.

## Why hybrid state + events?

| Signal | Strength | Weakness |
|--------|----------|----------|
| **State** | Shows durable problems (CrashLoopBackOff, NotReady nodes, privileged pods) | Misses flapping or already-recovered incidents |
| **Events** | Captures incident velocity and recent failures | Noisy; decays without tuning |

crank merges both into a single feature vector, then scores with a weighted heuristic (no model required) or a **pairwise-trained linear model**. Keyword area scores are fed directly into the feature vector so the model can learn interactions between cluster health and operator-interest context.

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

# Train from ranked sessions (see "Training with operator feedback" below)
crank train --dataset examples/training_dataset.jsonl --output models/crank.joblib
```

Point `scoring.model_path` in `config/crank.yaml` at the trained model. When a model is loaded, it is used directly (`scoring_mode: ml`); without one, crank falls back to a hand-tuned weighted heuristic (`scoring_mode: heuristic`).

Config is auto-discovered from `./config/crank.yaml` or `~/.config/crank/config.yaml` when `--config` is omitted. YAML keyword rules are merged onto built-in defaults (same `pattern` overrides the default).

## Architecture

```
┌─────────────────┐     ┌──────────────────┐
│ K8s API         │────▶│ FeatureExtractor │──┐
│ (state+events)  │     │ (20 base feat.)  │  │
└─────────────────┘     └──────────────────┘  │  ┌─────────────────┐
                                              ├─▶│ ClusterScorer   │
┌─────────────────┐     ┌──────────────────┐  │  │ ML or heuristic │
│ searchable text │────▶│ KeywordMatcher   │──┘  └────────┬────────┘
│ (names, events) │     │ (5 area scores)  │              │
└─────────────────┘     └──────────────────┘              ▼
                                                    ClusterRanker
                                                     (final rank)
```

Keyword area scores are included in the feature vector (25 features total: 20 base + 5 keyword) so the model learns how operator-interest context interacts with cluster health signals.

### Attention areas

Keywords map to operational themes:

- **reliability** — crash loops, prod workloads, payments
- **security** — privileged pods, vault, hostPath
- **capacity** — OOM, eviction, failed scheduling
- **compliance** — PCI, HIPAA, certificate expiry
- **platform** — mesh, GitOps, observability stack

Customize patterns in `config/crank.yaml`.

### Features (ML input)

**Base features (20):** node NotReady/pressure ratios, pod failure/crash/privilege signals, pending age, event rates (warnings, OOM, evictions, backoff), and workload availability (Deployments, StatefulSets, DaemonSets).

**Keyword features (5):** one per attention area (reliability, security, capacity, compliance, platform), populated from keyword matcher scores.

## Configuration

See `config/crank.yaml`. Key knobs:

- `event_window_hours` — how far back to scan events (default 24h)
- `keywords` — substring patterns → area + weight
- `scoring.keyword_boost_cap` — max aggregate keyword score returned by the matcher (default 25)
- `scoring.model_path` — path to a trained model file (`.joblib`)

## Training with operator feedback

Training uses **ranked sessions**: during a triage session, operators rank the clusters they reviewed by how much attention each deserved. Each JSONL row includes a `session` identifier and an ordinal `rank` (1 = most attention needed).

```json
{"session": "2026-05-19-triage", "rank": 1, "name": "prod-eu-pci", "nodes": {"total": 30, "not_ready": 3}, ...}
{"session": "2026-05-19-triage", "rank": 2, "name": "prod-us-east", "nodes": {"total": 50, "not_ready": 2}, ...}
{"session": "2026-05-20-oncall", "rank": 1, "name": "prod-ca", "nodes": {"total": 15, "not_ready": 1}, ...}
```

Pairs are generated within each session, so rankings from different days or conditions don't contaminate each other. The same cluster can appear in multiple sessions with different ranks as conditions change. A session needs at least 2 clusters; training requires at least 10 total snapshots across all sessions.

Internally, crank fits a logistic regression on feature differences between pairs (pairwise learning-to-rank). The learned coefficient vector becomes the scoring function: `score(cluster) = coef . features(cluster)`. Raw scores are calibrated to 0–100 via min-max scaling on the training data. At inference, each cluster is scored independently and sorted -- no pairwise queries needed.

Training prints `pairwise_accuracy` (fraction of pairs correctly ordered) and `kendall_tau` (rank correlation). Retrain periodically so the model reflects your environment.

The `train` command accepts `--config` so keyword rules used during training match those used at scoring time.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
mypy
```

## License

MIT
