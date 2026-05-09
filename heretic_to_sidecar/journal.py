"""Read Heretic's optuna JSONL journal and surface trial parameters.

Heretic uses optuna's `JournalFileStorage` (or a similar JournalLog format)
which writes one JSON record per line. The records use op_codes that
correspond to internal optuna events:

    op_code 0  CREATE_STUDY            { study_name, directions }
    op_code 2  SET_STUDY_USER_ATTR     { study_id, user_attr }
    op_code 4  CREATE_TRIAL            { trial_id, ... }
    op_code 5  SET_TRIAL_PARAM         { trial_id, param_name, param_value_internal, distribution }
    op_code 7  SET_TRIAL_USER_ATTR     { trial_id, user_attr }
    op_code 8  SET_TRIAL_STATE_VALUES  { trial_id, state, values }

For the purpose of replaying a trial via Heretic, we need:

    * direction_scope    "global" or "per layer"
    * direction_index    float (only meaningful when direction_scope == "global")
    * <component>.max_weight, max_weight_position, min_weight, min_weight_distance
        for each component in {"attn.o_proj", "mlp.down_proj"} (Gemma-style models)
    * trial outcomes (kl_divergence, refusals) — useful for selecting
      a Pareto trial without recomputing.

The CategoricalDistribution stores choices and the param value is the
INDEX into the choices list (param_value_internal=1 → choices[1]).

Op-code mapping observed in the Gemma-4-E2B-it journal (Heretic 0.x +
optuna JournalFileStorage):

    0  CREATE_STUDY
    2  SET_STUDY_USER_ATTR
    4  CREATE_TRIAL
    5  SET_TRIAL_PARAM
    6  SET_TRIAL_STATE_VALUES   { state, values, datetime_complete }
    8  SET_TRIAL_USER_ATTR

`values` is `[kl_divergence, refusal_fraction]`; refusal_fraction is
`refusals / n_bad_prompts` (so trial 9 with 17/99 refusals records
0.1717..., not 0.17).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrialRecord:
    trial_id: int
    state: str | None = None  # "COMPLETE", "FAIL", etc.
    values: list[float] | None = None  # multi-objective: [kl_divergence, refusals]
    params: dict[str, Any] = field(default_factory=dict)
    user_attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class JournalContents:
    study_name: str | None = None
    settings: dict[str, Any] | None = None  # parsed from study user_attr "settings"
    trials: dict[int, TrialRecord] = field(default_factory=dict)


_OPTUNA_TRIAL_STATE_NAMES = {
    0: "RUNNING",
    1: "COMPLETE",
    2: "PRUNED",
    3: "FAIL",
    4: "WAITING",
}


def _decode_param_value(distribution_json: str, raw_value: float | int) -> Any:
    """Translate a SET_TRIAL_PARAM record's `param_value_internal` to the
    user-visible parameter value, given the optuna distribution metadata.

    For FloatDistribution / IntDistribution, the internal value is the
    user-visible value. For CategoricalDistribution, it's the index into
    the `choices` list — translate to the choice string.
    """
    try:
        distribution = json.loads(distribution_json)
    except json.JSONDecodeError:
        return raw_value
    name = distribution.get("name")
    attrs = distribution.get("attributes", {})
    if name == "CategoricalDistribution":
        choices = attrs.get("choices", [])
        try:
            idx = int(raw_value)
        except (TypeError, ValueError):
            return raw_value
        if 0 <= idx < len(choices):
            return choices[idx]
        return raw_value
    return raw_value


def parse_journal(path: str | Path) -> JournalContents:
    """Walk a Heretic optuna JournalLog jsonl file and reconstruct the
    per-trial parameters and outcomes."""
    out = JournalContents()
    with open(path, "r") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                rec = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            op = rec.get("op_code")

            if op == 0:  # CREATE_STUDY
                out.study_name = rec.get("study_name")

            elif op == 2:  # SET_STUDY_USER_ATTR (carries the JSON-encoded settings blob)
                ua = rec.get("user_attr", {})
                if "settings" in ua and isinstance(ua["settings"], str):
                    try:
                        out.settings = json.loads(ua["settings"])
                    except json.JSONDecodeError:
                        pass

            elif op == 4:  # CREATE_TRIAL
                tid = rec.get("trial_id")
                if tid is None:
                    continue
                out.trials.setdefault(tid, TrialRecord(trial_id=tid))

            elif op == 5:  # SET_TRIAL_PARAM
                tid = rec.get("trial_id")
                if tid is None:
                    continue
                trial = out.trials.setdefault(tid, TrialRecord(trial_id=tid))
                name = rec.get("param_name")
                raw = rec.get("param_value_internal")
                dist = rec.get("distribution", "{}")
                if name is not None:
                    trial.params[name] = _decode_param_value(dist, raw)

            elif op == 6:  # SET_TRIAL_STATE_VALUES
                tid = rec.get("trial_id")
                if tid is None:
                    continue
                trial = out.trials.setdefault(tid, TrialRecord(trial_id=tid))
                state_id = rec.get("state")
                if isinstance(state_id, int):
                    trial.state = _OPTUNA_TRIAL_STATE_NAMES.get(state_id, str(state_id))
                values = rec.get("values")
                if isinstance(values, list):
                    trial.values = list(values)

            elif op == 8:  # SET_TRIAL_USER_ATTR
                tid = rec.get("trial_id")
                if tid is None:
                    continue
                trial = out.trials.setdefault(tid, TrialRecord(trial_id=tid))
                ua = rec.get("user_attr", {})
                trial.user_attrs.update(ua)
    return out


def select_pareto_trials(journal: JournalContents) -> list[TrialRecord]:
    """Return COMPLETE trials in Pareto-optimal order (multi-objective).

    Heretic's two objectives are kl_divergence (minimize) and refusals
    (minimize), both encoded in `values` per the SET_TRIAL_STATE_VALUES
    record. A trial is Pareto-optimal if no other completed trial dominates
    it on both objectives. Sort the survivors by kl + refusals/100 so the
    return order roughly tracks "best refusal-removal under low KL"; tie-
    break with trial_id for determinism.
    """
    completed = [
        t for t in journal.trials.values()
        if t.state == "COMPLETE" and t.values and len(t.values) >= 2
    ]
    pareto = []
    for t in completed:
        dominated = False
        for u in completed:
            if u is t:
                continue
            if (u.values[0] <= t.values[0] and u.values[1] <= t.values[1]
                    and (u.values[0] < t.values[0] or u.values[1] < t.values[1])):
                dominated = True
                break
        if not dominated:
            pareto.append(t)
    pareto.sort(key=lambda t: (t.values[0] + t.values[1] / 100.0, t.trial_id))
    return pareto


def trial_summary(trial: TrialRecord) -> str:
    kl = trial.user_attrs.get("kl_divergence")
    if kl is None and trial.values:
        kl = trial.values[0]
    refusals = trial.user_attrs.get("refusals")
    n_bad = trial.user_attrs.get("n_bad_prompts")
    direction_scope = trial.params.get("direction_scope", "?")
    refusals_str = (
        f"{refusals}/{n_bad}" if refusals is not None and n_bad is not None
        else (f"{refusals}" if refusals is not None else "?")
    )
    return (
        f"trial #{trial.trial_id:>2}  "
        f"KL={kl:.4f}  refusals={refusals_str}  scope={direction_scope}"
    )
