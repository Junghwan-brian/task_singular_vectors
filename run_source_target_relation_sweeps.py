"""
소스↔타깃 태스크 관련성(유사/비유사)에 따른 성능 비교 실험 런처.

- Similar source -> target:
  Source: {CUB200, Flowers102, OxfordIIITPet}
  Target: FGVCAircraft
- Dissimilar source -> target:
  Source: {MNIST, EMNIST, FashionMNIST}
  Target: FGVCAircraft

각 조건에서 k-shot = 2,4,8 그리고 adapter = none, tip, lp++를 스윕하고
최종 accuracy를 JSON/테이블로 정리한다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


TARGET = "FGVCAircraft"
K_LIST = [4]
ADAPTERS = ["none"]
MODEL = "ViT-B-32"


@dataclass(frozen=True)
class Condition:
    name: str
    config_file: str
    sources: List[str]


CONDITIONS = [
    Condition(
        name="similar",
        config_file=os.path.join("config", "config_reverse_similar_aircraft.yaml"),
        sources=["CUB200", "Flowers102", "OxfordIIITPet"],
    ),
    Condition(
        name="dissimilar",
        config_file=os.path.join("config", "config_reverse_dissimilar_aircraft.yaml"),
        sources=["MNIST", "EMNIST", "FashionMNIST"],
    ),
]


def _checkpoint_root() -> str:
    return os.path.join(".", "models", "checkpoints", MODEL)


def _exists_required_checkpoints(cond: Condition) -> Tuple[bool, List[str]]:
    """energy_train_reverse.py가 기대하는 최소 파일이 존재하는지 점검."""
    missing: List[str] = []
    root = _checkpoint_root()
    zeroshot = os.path.join(root, "nonlinear_zeroshot.pt")
    if not os.path.exists(zeroshot):
        missing.append(zeroshot)
    for ds in cond.sources:
        ft = os.path.join(root, f"{ds}Val", "nonlinear_finetuned.pt")
        if not os.path.exists(ft):
            # get_finetuned_path가 finetuned.pt도 허용하지만, 통일성 위해 둘 다 확인
            alt = os.path.join(root, f"{ds}Val", "finetuned.pt")
            if not os.path.exists(alt):
                missing.append(ft)
    return (len(missing) == 0), missing


def _run_one(config_file: str, k: int, adapter: str) -> str:
    cmd = [
        sys.executable,
        "energy_train_reverse.py",
        "--config_file",
        config_file,
        "--model",
        MODEL,
        "--test_dataset",
        TARGET,
        "--k",
        str(k),
        "--adapter",
        adapter,
    ]
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)}")
    # 결과 파일은 실행 로그에 출력되지만, 여기서는 deterministic 경로로 찾는다.
    # energy_train_reverse.py는 cfg.save_dir=cfg.model_location/cfg.model (기본값) 를 사용함.
    save_dir = os.path.join(".", "models", "checkpoints", MODEL)
    dataset_val = TARGET + "Val"
    # config_tag는 config 파일에 고정값으로 넣어둠.
    base = os.path.basename(config_file).lower()
    if "dissimilar" in base:
        config_tag = "src_dissimilar_to_aircraft"
    elif "similar" in base:
        config_tag = "src_similar_to_aircraft"
    else:
        raise ValueError(f"Unknown config file naming (expected similar/dissimilar): {config_file}")
    shot_folder = f"{k}shots"
    # adapter 파일명 규칙: energy_results_{adapter_tag}.json (none/tip/lp++)
    # lp++는 파일명에 '+'가 들어감.
    result_path = os.path.join(save_dir, dataset_val, config_tag, shot_folder, f"energy_results_{adapter}.json")
    if not os.path.exists(result_path):
        # 에너지 코드에서 adapter_result_tag가 조금 달라질 수 있어(예: lp++), fallback
        # 1) lp++는 그대로, 2) none/tip은 그대로
        raise FileNotFoundError(f"Expected results JSON not found: {result_path}")
    return result_path


def _extract_final_accuracy(result_json: Dict) -> float:
    # adapter가 있으면 adapter_results.final_accuracy를 우선 사용
    adapter_results = result_json.get("adapter_results")
    if isinstance(adapter_results, dict) and "final_accuracy" in adapter_results:
        return float(adapter_results["final_accuracy"])
    return float(result_json["final_accuracy"])


def main() -> None:
    all_results: Dict[str, Dict[int, Dict[str, float]]] = {}

    for cond in CONDITIONS:
        ok, missing = _exists_required_checkpoints(cond)
        if not ok:
            raise SystemExit(
                "Missing required checkpoints for condition '{}':\n{}".format(
                    cond.name, "\n".join("  - " + p for p in missing)
                )
            )

        cond_table: Dict[int, Dict[str, float]] = {}
        for k in K_LIST:
            row: Dict[str, float] = {}
            for adapter in ADAPTERS:
                path = _run_one(cond.config_file, k=k, adapter=adapter)
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                row[adapter] = _extract_final_accuracy(payload)
            cond_table[k] = row

        all_results[cond.name] = cond_table

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "source_target_relation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Print markdown tables
    def _print_table(name: str, table: Dict[int, Dict[str, float]]) -> None:
        print("\n" + "=" * 80)
        print(f"Condition: {name}")
        print("=" * 80)
        print("| k-shot | energy (none) | energy+TIP | energy+LP++ |")
        print("|---:|---:|---:|---:|")
        for k in sorted(table.keys()):
            r = table[k]
            print(
                f"| {k} | {100*r['none']:.2f} | {100*r['tip']:.2f} | {100*r['lp++']:.2f} |"
            )

    _print_table("similar", all_results["similar"])
    _print_table("dissimilar", all_results["dissimilar"])
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

