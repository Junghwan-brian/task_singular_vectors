#%%
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
                             color_list=None):
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
    rings = np.linspace(0, 1, 6)
    base_radius_offset = rings[1]     # 꼴찌 시작 반경(가장 작은 원)
    theta_bg = np.linspace(0, 2*np.pi, 720)

    # 교차 링 배경 (아래 레이어)
    for i in range(5):
        color_bg = "white" if i % 2 == 0 else "#e0e0e0"
        ax.fill_between(theta_bg, rings[i], rings[i+1], color=color_bg, alpha=0.3, zorder=-2)

    ax.set_ylim(0, 1.05)

    # 라디얼 가이드 (옅은 선)
    guide_color = "#bfbfbf"
    for a in angles[:-1]:
        ax.plot([a, a], [0, 1.0], color=guide_color, linewidth=0.6, alpha=0.8, zorder=-1)

    # 바깥 원을 아래 레이어에 (숫자/마커 가리지 않도록)
    outer_eps = 0.010
    ax.plot(theta_bg, np.ones_like(theta_bg) - outer_eps,
            color=guide_color, linewidth=0.7, alpha=0.65,
            solid_capstyle="round", zorder=-5)

    # 색상 팔레트 준비 (입력 없으면 tab20에서 필요한 만큼)
    if color_list is None:
        color_list = plt.cm.tab20(np.linspace(0, 1, len(sub)))
    elif len(color_list) < len(sub):
        raise ValueError(f"color_list length ({len(color_list)}) is smaller than methods ({len(sub)}).")

    # 채움/선/포인트/점수
    used_methods, used_colors = [], []
    for idx, (_, row) in enumerate(sub.iterrows()):
        method = str(row["Method"])
        vals = norm.loc[row.name].tolist()
        vals = [v * (1 - base_radius_offset) + base_radius_offset for v in vals]
        vals += vals[:1]
        raw_vals = sub.loc[row.name, dataset_cols].tolist() + [sub.loc[row.name, dataset_cols].tolist()[0]]

        color = color_list[idx]
        used_methods.append(method)
        used_colors.append(color)

        # 내부 채움(옅게) → 선(진하게)
        ax.fill(angles, vals, color=color, alpha=0.25, linewidth=0, zorder=2)
        ax.plot(angles, vals, linewidth=1.8, color=color, alpha=0.9, zorder=3)

        # 꼭짓점: 내부 흰색, 테두리 해당 색
        ax.scatter(angles, vals, s=28, facecolors="white", edgecolors=[color],
                   linewidths=1.2, zorder=4)

        # 점수 텍스트(조금 안쪽)
        for a, r, raw in zip(angles[:-1], vals[:-1], raw_vals[:-1]):
            ax.text(
                a, r - 0.04, f"{raw:.1f}",
                ha="center", va="center",
                fontsize=11,                      # ← 크기 키움 (예: 10~12)
                # fontweight="bold",                # ← 진하게
                color="black",
                zorder=6
            )

    # 데이터셋 라벨(원 밖)
    label_radius = 1.10
    for angle, label in zip(angles[:-1], labels):
        ax.text(angle, label_radius, label, fontsize=11, fontweight="bold",
                ha="center", va="center", zorder=6)

    # 축별 최소값 (중심쪽)
    for ang, dmin in zip(angles[:-1], data_min.tolist()):
        ax.text(ang, base_radius_offset - 0.05, f"{dmin:.1f}",
                ha="center", va="center", fontsize=8, color="gray", zorder=6)

    # 범례는 본 플롯에서 그리지 않음(별도 저장을 위해)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # ✅ 위쪽 여유 확보
    plt.show()

    return used_methods, used_colors

# --------------------- 사용 예시 ---------------------
if __name__ == "__main__":
    # 1) (선택) 가능한 메서드 보기
    # _ = list_methods("ViT-B-16", 4)

    # 2) 플롯 실행 (메서드 선택 + 색 팔레트)
    methods_to_use = ["Energy (best)", "LinearProbe (best config)", "TIP (best config)", "LP++ (best config)", "Atlas"]
    custom_colors   = plt.cm.tab20(np.linspace(0, 1, 6))
    used_methods, used_colors = plot_radar_for_selection(
        "ViT-B-16", 4,
        methods_to_show=methods_to_use,
        color_list=custom_colors
    )
    used_methods = [method.replace(" (best config)","").replace("(best)", "") for method in used_methods]

    # 3) 플롯에서 실제 사용된 순서/색으로 legend만 별도 저장
    save_legend_only(used_methods, used_colors, save_path="figures/legend_only.png", fontsize=10)
