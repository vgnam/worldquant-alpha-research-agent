# API Research Notes

Verified against official WorldQuant-owned properties and the live production frontend on April 7-10, 2026.

## Primary Sources

- Official consultant page: `https://worldquantbrain.com/consultant`
- Production frontend bundle: `https://platform.worldquantbrain.com/static/js/1578.8f1c1c8c.js`
- Platform app shell: `https://platform.worldquantbrain.com/`

## Important Framing

WorldQuant BRAIN does not expose a widely documented public API reference for every endpoint and schema used by the frontend. The client in this repository therefore mixes:

- direct live verification
- official frontend inspection
- conservative implementation choices

Whenever behavior is inferred rather than explicitly documented, the code prefers simple request patterns, explicit polling, and transparent error propagation.

## Verified Endpoint Flow

### 1. Authentication

- Endpoint: `POST /authentication`
- Observed auth mechanism: `Authorization: Basic base64(email:password)`
- Session handling: cookie-backed
- Important caveat: the frontend also contains reCAPTCHA handling, so password-only automation may fail on challenged accounts

#### Observed persona challenge

On a challenged account, `POST /authentication` with correct credentials returns:

```text
401
WWW-Authenticate: persona
{"inquiry": "inq_..."}
```

This is not a wrong-password response. It means the platform wants an interactive challenge solved in the browser before it will mint a session. There is no known way to satisfy it headlessly, so the supported workaround is to copy the browser `Cookie` header into `WQB_COOKIE_HEADER`.

#### Session cookie name

The session cookie observed on the current platform is named `t`, not `sessionid`. `sessionid` does not appear at all. The client forwards the whole cookie string verbatim, so the pragmatic rule is: copy every cookie pair, never a subset.

### 2. Simulation

- Metadata endpoint: `OPTIONS /simulations`
- Start simulation: `POST /simulations`
- Result flow:
  - response may include `Location`
  - response may include `Retry-After`
  - polling continues until a terminal payload is available

### 3. Alpha Inspection

- Alpha detail: `GET /alphas/{id}`
- Submission checks: `GET /alphas/{id}/check`

### 4. Submission

- Submit request: `POST /alphas/{id}/submit`
- If async: poll `GET /alphas/{id}/submit`

Important: the frontend also references `/submissions`, but that is not the same flow as the main alpha submission button used after a normal simulation.

## Header Versioning Observed in Frontend

- General JSON calls: `Accept: application/json;version=2.0`
- `OPTIONS /simulations`: `Accept: application/json;version=3.0`

This repository follows that behavior.

## Regular Simulation Payload

Observed payload shape:

```json
{
  "type": "REGULAR",
  "settings": {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 4,
    "neutralization": "SECTOR",
    "truncation": 0.08,
    "pasteurization": "ON",
    "unitHandling": "VERIFY",
    "nanHandling": "ON",
    "language": "FASTEXPR",
    "visualization": false
  },
  "regular": "rank(ts_delta(close, 5))"
}
```

## Response Detail That Matters in Practice

Two details are especially important for robust automation:

### Simulation payloads are async

The immediate simulation response is not the final alpha detail. The implementation must honor:

- `Location`
- `Retry-After`
- follow-up polling

### Final simulation payload can expose the alpha id in different shapes

In live testing, `simulation["alpha"]` may be:

- a string alpha id
- a nested alpha object

The repo client handles both.

## Endpoints Used by This Repo

The CLI and pipeline use:

- `POST /authentication`
- `OPTIONS /simulations`
- `POST /simulations`
- `GET /alphas/{id}`
- `GET /alphas/{id}/check`
- `POST /alphas/{id}/submit`
- `GET /alphas/{id}/submit`
- `GET /operators`
- `GET /data-fields/summary`

## Live Research Observations

The following behaviors were confirmed live while building the pipeline:

- quality blockers often move in stages:
  - early stage: `LOW_SHARPE`, `LOW_FITNESS`, `CONCENTRATED_WEIGHT`
  - later stage: `SELF_CORRELATION`
  - final stage: `PROD_CORRELATION`
- some expressions that look distinct are still highly correlated with existing production or self history
- small parameter changes can move correlation checks more than quality checks

## Observed Check Inventory

`GET /alphas/{id}/check` currently returns **18** checks, while this repository models only a subset. Observed names:

```text
LOW_SHARPE  LOW_FITNESS  LOW_TURNOVER  HIGH_TURNOVER  CONCENTRATED_WEIGHT
LOW_SUB_UNIVERSE_SHARPE  LOW_2Y_SHARPE  CLUSTER_TEST  DATA_DIVERSITY
SELF_CORRELATION  PROD_CORRELATION  REGULAR_SUBMISSION  POWER_POOL_CORRELATION
MATCHES_PYRAMID  MATCHES_THEMES  OSMOSIS_ALLOCATION
POWER_POOL_DESCRIPTION_LENGTH  POWER_POOL_DESCRIPTION_FORMAT
```

Notes that matter for correctness:

- `LOW_2Y_SHARPE` behaves like a hard quality gate (observed limit `1.58` on `USA`/`TOP3000`, `delay=1`). It was missing from the blocking set and has since been added, otherwise a candidate failing only this check would have been reported as quality-ready.
- `CLUSTER_TEST` and `POWER_POOL_CORRELATION` came back as `WARNING` while exceeding their limits, so they may become blockers later.
- Thresholds are account/configuration specific. `LOW_SHARPE` was observed at `1.58`, which is much stricter than the `1.25`-style thresholds often quoted for research; seeds tuned for Sharpe ~1.3 will never pass.
- Any new check introduced by the platform is penalized only by the generic fallback weight, so `CHECK_SEVERITY_WEIGHTS` and `BLOCKING_CHECKS` should be revisited periodically.

## Caveats

- Endpoint behavior can change without notice.
- Account-specific permissions may limit available settings, data fields, or automation feasibility.
- reCAPTCHA can break headless password login.
- Passing visible research checks is not the same as guaranteed submittability if correlation checks later resolve to failure.
