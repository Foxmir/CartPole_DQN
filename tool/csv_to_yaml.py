# CartPole_DQN/tool/csv_to_yaml.py
# Read step2_top_10.csv (in project root) -> write step3_top10_candidates.yaml (in project root)
# No third-party dependencies.
# 使用时可以直接把csv和本代码放到根目录下直接运行，就不会路径报错了

import csv
import re
from pathlib import Path


INPUT_CSV = "step2_top_10.csv"
OUTPUT_YAML = "step3_top10_candidates.yaml"

STEP3_AGENT_NAME = "Step3_DQNAgent"
STEP3_DESCRIPTION = "Step 3 model selection: choose the champion hyperparameter configuration from 10 candidates."


_REQUIRED_COLS = [
    "Name",
    "ID",
    "config_all_train_seed_mean",
    "config_all_train_seed_sd",
    "agent.epsilon_decay",
    "agent.gamma",
    "agent.learning_rate",
    "agent.tau",
    "agent.epsilon_min",
    "agent.epsilon_start",
    "environment.max_episode_steps",
    "environment.name",
    "memory.capacity",
    "network.type",
    "training.batch_size",
    "training.initial_collect_size",
    "training.num_episodes",
]


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def _as_number_for_sort(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return float("-inf")


def _yaml_quote(s: str) -> str:
    # minimal safe quoting for strings
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f"\"{s}\""


def _yaml_scalar(s: str) -> str:
    """
    Return YAML scalar representation:
    - if looks like int/float -> return raw (unquoted) to stay numeric
    - else -> return quoted string
    - empty -> 'null'
    """
    if s is None:
        return "null"
    s = str(s).strip()
    if s == "":
        return "null"
    if _INT_RE.match(s) or _FLOAT_RE.match(s):
        return s
    return _yaml_quote(s)


def main():
    root = Path(__file__).resolve().parent
    in_path = root / INPUT_CSV
    out_path = root / OUTPUT_YAML

    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    rows = []
    with in_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")

        missing = [c for c in _REQUIRED_COLS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}\nGot columns: {reader.fieldnames}")

        for r in reader:
            rows.append(r)

    # sort by mean desc (tie-breaker: ID asc for determinism)
    rows.sort(
        key=lambda r: (-_as_number_for_sort(r["config_all_train_seed_mean"]), str(r["ID"]))
    )

    if len(rows) != 10:
        # 你说是 Top10，这里做强检查，避免误把更多/更少候选写进 Step3
        raise ValueError(f"Expected 10 rows (Top10), but got {len(rows)} rows.")

    lines = []
    lines.append("# Auto-generated from step2_top_10.csv")
    lines.append("# Format: top-level keys are '01'..'10' (ranked by config_all_train_seed_mean desc)")
    lines.append("")

    for idx, r in enumerate(rows, start=1):
        key = f"{idx:02d}"

        # comment for traceability (won't affect config parsing)
        lines.append(
            f"# {key}: mean={r['config_all_train_seed_mean']}, sd={r['config_all_train_seed_sd']}, "
            f"wandb_id={r['ID']}, wandb_name={r['Name']}"
        )

        lines.append(f"{key}:")
        # description
        lines.append(f"  description: {_yaml_quote(STEP3_DESCRIPTION)}")

        # environment
        lines.append("  environment:")
        lines.append(f"    name: {_yaml_scalar(r['environment.name'])}")
        lines.append(f"    max_episode_steps: {_yaml_scalar(r['environment.max_episode_steps'])}")
        lines.append("    render_mode: null")

        # network (even if hardcoded, keeping type is usually harmless; remove if你确定代码完全不读它)
        lines.append("  network:")
        lines.append(f"    type: {_yaml_scalar(r['network.type'])}")

        # memory
        lines.append("  memory:")
        lines.append(f"    capacity: {_yaml_scalar(r['memory.capacity'])}")

        # agent
        lines.append("  agent:")
        lines.append(f"    name: {_yaml_quote(STEP3_AGENT_NAME)}")
        lines.append(f"    gamma: {_yaml_scalar(r['agent.gamma'])}")
        lines.append(f"    learning_rate: {_yaml_scalar(r['agent.learning_rate'])}")
        lines.append(f"    epsilon_start: {_yaml_scalar(r['agent.epsilon_start'])}")
        lines.append(f"    epsilon_decay: {_yaml_scalar(r['agent.epsilon_decay'])}")
        lines.append(f"    epsilon_min: {_yaml_scalar(r['agent.epsilon_min'])}")
        lines.append(f"    tau: {_yaml_scalar(r['agent.tau'])}")

        # training
        lines.append("  training:")
        lines.append(f"    num_episodes: {_yaml_scalar(r['training.num_episodes'])}")
        lines.append(f"    batch_size: {_yaml_scalar(r['training.batch_size'])}")
        lines.append(f"    initial_collect_size: {_yaml_scalar(r['training.initial_collect_size'])}")

        lines.append("")  # blank line between candidates

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
