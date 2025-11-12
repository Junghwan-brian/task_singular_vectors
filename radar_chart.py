#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import matplotlib.patches as mpatches

# ------------------------------------------------------------
# CSV 경로
# ------------------------------------------------------------
csv_path = "results/comprehensive_results_table.csv"
if os.path.exists(csv_path):
    df_raw = pd.read_csv(csv_path)
else:
    raise FileNotFoundError(f"파일이 존재하지 않습니다: {csv_path}")
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "DejaVu Sans",   # 대부분 환경에서 기본 제공
    "font.size": 11,
    "mathtext.default": "regular"
})
# ------------------------------------------------------------
# 전처리
# ------------------------------------------------------------
df = df_raw.copy()
df["Model"] = df["Model"].ffill()
df["Shot"]  = df["Shot"].ffill()
df["Shot"]  = df["Shot"].astype(str).str.extract(r"(\d+)").astype(float)

non_dataset_cols = {"Model", "Shot", "Method", "Average"}
dataset_cols = [c for c in df.columns if c not in non_dataset_cols]
# 이름이 너무 길면 원이랑 겹치므로 직접 설정함.
if 'CIFAR10' in dataset_cols:
    dataset_cols = ['CIFAR10', 'RenderedSST2', 'Country211', 'CIFAR100', 'CUB200', 'DTD', 'EMNIST', 'FGVCAircraft', 'FashionMNIST', 'Flowers102', 'OxfordIIITPet', 'Food101', 'GTSRB', 'MNIST', 'PCAM', 'STL10', 'SVHN']
# ------------------------------------------------------------
# 메서드 목록(번호 포함)
# ------------------------------------------------------------
def list_methods(model: str, shot: int):
    sub = df[(df["Model"] == model) & (df["Shot"] == shot)].copy()
    methods = (
        sub["Method"].dropna().astype(str).unique().tolist()
    )
    methods = sorted(methods, key=lambda s: s.lower())
    print("Available methods:")
    for i, m in enumerate(methods):
        print(f"[{i}] {m}")
    return methods

# ------------------------------------------------------------
# Legend만 저장
# ------------------------------------------------------------
def save_legend_only(methods, colors, save_path="figures/legend_only.png", fontsize=10):
    """
    methods : list[str]  - 플롯에서 실제 사용된 메서드명 순서
    colors  : list       - 각 메서드에 대응되는 색(플롯에서 실제 사용된 순서)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    handles = []
    for method, color in zip(methods, colors):
        patch = mpatches.FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle="round,pad=0.25,rounding_size=0.15",
            facecolor=color, edgecolor=color, linewidth=0
        )
        patch.set_label(method)
        handles.append(patch)

    # ✅ 한 줄로 표시하기 위해 ncol을 methods 개수로 설정
    ncol = len(methods)

    fig_legend = plt.figure(figsize=(max(5, len(methods)*1.5), 1.2))
    fig_legend.legend(
        handles=handles,
        loc="center",
        ncol=ncol,
        frameon=False,
        handlelength=1.6,
        handleheight=1.0,
        borderaxespad=0.2,
        fontsize=fontsize,
    )
    fig_legend.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig_legend)
    print(f"✅ Legend saved to: {save_path}")

# ------------------------------------------------------------
# 레이더 플롯 (추천: 반환값으로 methods/colors를 돌려받아 legend 저장에 재사용)
# ------------------------------------------------------------
def plot_radar_for_selection(model: str, shot: int,
                             methods_to_show=None, method_idx=None,
                             color_list=None,
                             # ▼ 새 파라미터들 (원하는 대로 조절)
                             label_fs_min: float = 8.0,
                             label_fs_max: float = 12.0,
                             label_fs_gamma: float = 1.0,
                             value_sim_thresh: float = 0.10,   # 값이 이 이내로 같으면 하나만 표기
                             label_min_rad_gap: float = 0.035, # 같은 각도 내 레이블 간 최소 간격
                             angle_jitter_deg: float = 3.0,    # 겹치면 각도를 ±로 살짝 비틀어 배치
                              jitter_attempts: int = 4,
                              int_label_t_thresh: float = 0.35,
                              outer_t_thresh: float = 0.50,          # 바깥 영역(3~5번째 링 부근)
                              outer_value_sim_thresh: float = 0.05,  # 바깥에서 매우 가까운 값 판정
                              outer_label_min_rad_gap: float = 0.050 # 바깥 영역 스택 간격(더 넉넉)
                              ):
    """
    반환:
        used_methods, used_colors  (플롯에서 실제 사용된 순서/색)
    """
    # 서브셋
    sub_all = df[(df["Model"] == model) & (df["Shot"] == shot)].copy()
    if sub_all.empty:
        raise ValueError(f"No rows found for Model={model}, Shot={shot}")

    # 선택 로직: indices -> names
    avail_methods = sorted(
        sub_all["Method"].dropna().astype(str).unique().tolist(),
        key=lambda s: s.lower()
    )
    if methods_to_show is None and method_idx is not None:
        methods_to_show = [avail_methods[i] for i in method_idx]
    if methods_to_show is None:
        methods_to_show = avail_methods

    sub = sub_all[sub_all["Method"].astype(str).isin(methods_to_show)].copy()
    if sub.empty:
        raise ValueError("No matching methods found for given selection.")

    # 축별 min/max, 정렬(낮은→높은 성능)
    data_min = sub[dataset_cols].min(axis=0)
    data_max = sub[dataset_cols].max(axis=0)
    sub["MeanAcc"] = sub[dataset_cols].mean(axis=1)
    sub = sub.sort_values("MeanAcc", ascending=True).reset_index(drop=True)
    norm = (sub[dataset_cols] - data_min) / (data_max - data_min + 1e-8)

    # 기본 극좌표 세팅
    labels = dataset_cols
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks([])             # 각도 라벨 제거
    ax.set_yticklabels([])

    # 기본 스파인/그리드 제거 (커스텀)
    ax.spines['polar'].set_visible(False)
    ax.grid(False)

    # 배경 링 + 시작 오프셋
    rings = np.linspace(0, 1, 6)          # 5개 구간
    base_radius_offset = rings[1]         # 꼴찌 시작 반경(가장 작은 원)
    theta_bg = np.linspace(0, 2*np.pi, 720)

    # 교차 링 배경
    for i in range(5):
        color_bg = "white" if i % 2 == 0 else "#e0e0e0"
        ax.fill_between(theta_bg, rings[i], rings[i+1],
                        color=color_bg, alpha=0.30, zorder=-2)

    # 내부 링 경계선 살짝 진하게
    ring_edge_color = "#9a9a9a"
    for rr in rings[1:-1]:  # 내부 경계만
        ax.plot(theta_bg, np.full_like(theta_bg, rr),
                color=ring_edge_color, linewidth=0.9, alpha=0.9, zorder=-1)

    ax.set_ylim(0, 1.05)

    # 라디얼 가이드(방사선) 진하게/굵게
    guide_color = "#7f7f7f"
    for a in angles[:-1]:
        ax.plot([a, a], [0, 1.0], color=guide_color, linewidth=1.2,
                alpha=0.95, zorder=-1)

    # 바깥 원
    outer_eps = 0.010
    ax.plot(theta_bg, np.ones_like(theta_bg) - outer_eps,
            color=guide_color, linewidth=0.9, alpha=0.8,
            solid_capstyle="round", zorder=-5)

    # 색상 팔레트
    if color_list is None:
        color_list = plt.cm.tab20(np.linspace(0, 1, 10))[::2]
    elif len(color_list) < len(sub):
        raise ValueError(f"color_list length ({len(color_list)}) is smaller than methods ({len(sub)}).")

    # --- 레이블 중복/겹침 관리용 레지스트리 ---
    # 각도 index별로 이미 배치된 레이블의 (표시값, raw, angle, r_text) 기록
    angle_registry = {i: [] for i in range(N)}
    jitter = np.deg2rad(angle_jitter_deg)
    # 예: 0, +1, -1, +2, -2 단계로 각도 지터 시도
    jitter_steps = [0]
    for k in range(1, jitter_attempts + 1):
        jitter_steps += [k, -k]

    # 각 각도(데이터셋)별 텍스트를 지연 배치하기 위한 버퍼
    labels_by_angle = {i: [] for i in range(N)}

    used_methods, used_colors = [], []
    for idx, (_, row) in enumerate(sub.iterrows()):
        method = str(row["Method"])
        vals = norm.loc[row.name].tolist()
        vals = [v * (1 - base_radius_offset) + base_radius_offset for v in vals]
        vals += vals[:1]

        raw_vals = sub.loc[row.name, dataset_cols].tolist()
        raw_vals += raw_vals[:1]

        color = color_list[idx]
        used_methods.append(method)
        used_colors.append(color)

        # 내부 채움(옅게) → 선(진하게)
        ax.fill(angles, vals, color=color, alpha=0.25, linewidth=0, zorder=2)
        ax.plot(angles, vals, linewidth=2.0, color=color, alpha=0.95, zorder=3)

        # 꼭짓점 마커
        ax.scatter(angles, vals, s=30, facecolors="white", edgecolors=[color],
                   linewidths=1.2, zorder=4)

        # ===== 숫자 텍스트 수집 =====
        for k, (a, r, raw) in enumerate(zip(angles[:-1], vals[:-1], raw_vals[:-1])):
            angle_idx = k
            # 반경 따라 폰트 크기 스케일 (안쪽일수록 작게)
            # t = 0 (base_radius_offset) -> fs_min, t = 1 (바깥) -> fs_max
            t = (r - base_radius_offset) / max(1e-8, (1.0 - base_radius_offset))
            t = min(max(t, 0.0), 1.0)
            fs = label_fs_min + (label_fs_max - label_fs_min) * (t ** label_fs_gamma)
            labels_by_angle[angle_idx].append({
                "a": a, "r": r, "raw": raw,
                "t": t, "fs": fs,
                "r_text": r - 0.04
            })

    # ===== 각 각도별 텍스트 실제 배치 =====
    for angle_idx in range(N):
        all_entries = labels_by_angle[angle_idx]
        if not all_entries:
            continue
        # 1) 값/반경이 매우 가까운 항목들은 큰 값만 남김 (raw 내림차순으로 선택)
        candidates = sorted(all_entries, key=lambda e: e["raw"], reverse=True)
        kept = []
        for e in candidates:
            suppress = False
            for kk in kept:
                # 값이 매우 비슷하거나, 반경 위치가 너무 가까우면(겹칠 우려) 낮은 값을 제거
                rad_thresh = outer_label_min_rad_gap if (e["t"] >= outer_t_thresh or kk["t"] >= outer_t_thresh) else label_min_rad_gap
                if abs(e["raw"] - kk["raw"]) <= value_sim_thresh or abs(e["r_text"] - kk["r_text"]) < rad_thresh:
                    suppress = True
                    break
            if not suppress:
                kept.append(e)

        a_axis = angles[angle_idx]
        deg = np.degrees(a_axis)
        # 읽기 편하게 뒤집힘 최소화(숫자가 뒤집히지 않도록 각도 보정)
        if 90 < deg < 270:
            deg -= 180

        # 2) 바깥/안쪽을 나눠서 배치
        outer_entries = [e for e in kept if e["t"] >= outer_t_thresh]
        inner_entries = [e for e in kept if e["t"] < outer_t_thresh]

        # 2-a) 바깥: 축 정렬 배치(회전), 기본은 각자 r_text, 충돌 시 약간 안쪽으로 이동
        for e in sorted(outer_entries, key=lambda x: x["r_text"], reverse=True):
            raw = e["raw"]; fs = e["fs"]; t = e["t"]; r_place = e["r_text"]
            text_str = f"{int(round(raw))}" if t <= int_label_t_thresh else f"{raw:.1f}"
            # 기존 배치와 간격 보장
            while any(abs(r_place - rec["r"]) < outer_label_min_rad_gap for rec in angle_registry[angle_idx]) and r_place > base_radius_offset + 0.02:
                r_place -= outer_label_min_rad_gap * 0.5
            r_place = max(r_place, base_radius_offset + 0.02)
            ax.text(a_axis, r_place, text_str,
                    ha="center", va="center",
                    rotation=deg, rotation_mode="anchor",
                    fontsize=fs, color="black", zorder=6)
            angle_registry[angle_idx].append({"text": text_str, "raw": raw, "angle": a_axis, "r": r_place})

        # 2-b) 안쪽: 기존 지터 방식으로 배치
        for e in inner_entries:
            raw = e["raw"]; fs = e["fs"]; t = e["t"]; r_text = e["r_text"]; a = e["a"]
            text_str = f"{int(round(raw))}" if t <= int_label_t_thresh else f"{raw:.1f}"
            placed = False
            for step in jitter_steps:
                a_try = a + step * jitter
                radial_conflict = any(abs(r_text - rec["r"]) < label_min_rad_gap for rec in angle_registry[angle_idx])
                if not radial_conflict or step != 0:
                    ax.text(a_try, r_text, text_str,
                            ha="center", va="center",
                            fontsize=fs, color="black", zorder=6)
                    angle_registry[angle_idx].append({"text": text_str, "raw": raw, "angle": a_try, "r": r_text})
                    placed = True
                    break
            if not placed:
                ax.text(a, r_text - 0.02, text_str,
                        ha="center", va="center",
                        fontsize=max(label_fs_min, fs - 1),
                        color="black", zorder=6)
                angle_registry[angle_idx].append({"text": text_str, "raw": raw, "angle": a, "r": r_text - 0.02})

    # 데이터셋 라벨(원 밖)
    label_radius = 1.10
    for angle, label in zip(angles[:-1], labels):
        ax.text(angle, label_radius, label, fontsize=11, fontweight="bold",
                ha="center", va="center", zorder=6)

    # (중앙부 회색 최솟값 텍스트는 제거된 상태)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("figures/radar_chart.png", bbox_inches="tight", dpi=300)
    plt.show()

    return used_methods, used_colors
# --------------------- 사용 예시 ---------------------
if __name__ == "__main__":
    # 1) (선택) 가능한 메서드 보기
    # _ = list_methods("ViT-B-16", 4)

    # 2) 플롯 실행 (메서드 선택 + 색 팔레트)
    methods_to_use = ["Energy (best)", "LinearProbe (best config)", "TIP (best config)", "Atlas"]
    custom_colors   = plt.cm.tab20(np.linspace(0, 1, 28))[::3]
    used_methods, used_colors = plot_radar_for_selection(
        "ViT-B-32", 16,
        methods_to_show=methods_to_use,
        color_list=custom_colors
    )
    used_methods = [method.replace(" (best config)","").replace(" (best)", "").replace("Energy", "BOLT").replace("LinearProbe", "LP") for method in used_methods]

    # 3) 플롯에서 실제 사용된 순서/색으로 legend만 별도 저장
    save_legend_only(used_methods, used_colors, save_path="figures/legend_only.png", fontsize=10)
