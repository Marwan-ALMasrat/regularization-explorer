
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")
 
# ── 1. Load Data ─────────────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\User\M5\regularization-explorer\data\telecom_churn.csv")
 
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(2))
 
# ── 2. Identify Target Column ────────────────────────────────────────────────
# Common names for churn target column
churn_candidates = [c for c in df.columns if "churn" in c.lower()]
target_col = churn_candidates[0] if churn_candidates else df.columns[-1]
print(f"\nUsing target column: '{target_col}'")
 
# ── 3. Preprocess ────────────────────────────────────────────────────────────
# Encode target if it's boolean / string
y_raw = df[target_col]
if y_raw.dtype == object or y_raw.dtype == bool:
    y = y_raw.map({True: 1, False: 0,
                   "True": 1, "False": 0,
                   "Yes": 1, "No": 0,
                   "yes": 1, "no": 0,
                   1: 1, 0: 0}).fillna(y_raw.astype(int))
else:
    y = y_raw.astype(int)
 
# Select only numeric feature columns
X_raw = df.drop(columns=[target_col])
X_num = X_raw.select_dtypes(include=[np.number])
 
# Drop columns with zero variance or all-NaN
X_num = X_num.dropna(axis=1, how="all")
X_num = X_num.loc[:, X_num.std() > 0]
X_num = X_num.fillna(X_num.median())
 
feature_names = X_num.columns.tolist()
print(f"\nFeatures used ({len(feature_names)}): {feature_names}")
 
# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X_num)
 
# ── 4. Generate 20 C Values (Log Scale) ──────────────────────────────────────
C_values = np.logspace(-3, 2, 20)   # 0.001 → 100
print(f"\nC values: {np.round(C_values, 4)}")
 
# ── 5. Train Models & Record Coefficients ────────────────────────────────────
coefs_l1 = []
coefs_l2 = []
 
for C in C_values:
    # L1
    model_l1 = LogisticRegression(
        penalty="l1", C=C, solver="saga",
        max_iter=5000, random_state=42
    )
    model_l1.fit(X, y)
    coefs_l1.append(model_l1.coef_[0])
 
    # L2
    model_l2 = LogisticRegression(
        penalty="l2", C=C, solver="lbfgs",
        max_iter=5000, random_state=42
    )
    model_l2.fit(X, y)
    coefs_l2.append(model_l2.coef_[0])
 
coefs_l1 = np.array(coefs_l1)   # shape: (20, n_features)
coefs_l2 = np.array(coefs_l2)
 
# ── 6. Identify Features Zeroed Out Under L1 ─────────────────────────────────
# A feature is "zeroed" when its absolute coefficient drops below a threshold
ZERO_THRESH = 1e-4
 
zeroed_at = {}   # feature_name → C value where first zeroed (smallest C = strongest reg)
for i, fname in enumerate(feature_names):
    for j, C in enumerate(C_values):                       # C_values is ascending
        if abs(coefs_l1[j, i]) < ZERO_THRESH:
            zeroed_at[fname] = C
            break
 
zeroed_features = list(zeroed_at.keys())
non_zeroed     = [f for f in feature_names if f not in zeroed_features]
 
# Sort zeroed features by the C at which they first reach zero (ascending = zeroed soonest)
zeroed_sorted = sorted(zeroed_at.items(), key=lambda x: x[1])
print("\nL1 zero-out order (feature → first C where coef ≈ 0):")
for fname, c_val in zeroed_sorted:
    print(f"  {fname:35s} C = {c_val:.4f}")
 
# ── 7. Plot ───────────────────────────────────────────────────────────────────
# Color palette — one color per feature
n_features = len(feature_names)
cmap       = plt.cm.tab20 if n_features <= 20 else plt.cm.hsv
colors     = [cmap(i / n_features) for i in range(n_features)]
feat_color = {f: colors[i] for i, f in enumerate(feature_names)}
 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
fig.patch.set_facecolor("#0f1117")
for ax in (ax1, ax2):
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
 
LABEL_STYLE = dict(color="#c9d1d9", fontsize=10)
TITLE_STYLE = dict(color="#e6edf3", fontsize=13, fontweight="bold", pad=12)
TICK_STYLE  = dict(colors="#8b949e", labelsize=9)
 
# ── Subplot 1: L1 (Lasso) ───────────────────────────────────────────────────
for i, fname in enumerate(feature_names):
    alpha = 1.0 if fname not in zeroed_features else 0.7
    lw    = 1.8 if fname not in zeroed_features else 1.2
    ax1.plot(C_values, coefs_l1[:, i],
             color=feat_color[fname], alpha=alpha, lw=lw, label=fname)
 
# Annotate first-zeroed feature
if zeroed_sorted:
    first_zeroed_name, first_zero_C = zeroed_sorted[0]
    fi = feature_names.index(first_zeroed_name)
    # Find the y-value just before zeroing
    for j, C in enumerate(C_values):
        if abs(coefs_l1[j, fi]) < ZERO_THRESH:
            y_ann = coefs_l1[max(0, j-1), fi]
            break
    ax1.annotate(
        f"← First to zero:\n  {first_zeroed_name}",
        xy=(first_zero_C, 0),
        xytext=(first_zero_C * 3, y_ann + 0.05),
        fontsize=8, color="#f85149",
        arrowprops=dict(arrowstyle="->", color="#f85149", lw=0.8),
    )
 
ax1.axhline(0, color="#484f58", lw=0.8, ls="--")
ax1.set_xscale("log")
ax1.set_xlabel("C  (regularization strength ↑ as C → 0)", **LABEL_STYLE)
ax1.set_ylabel("Standardized Coefficient", **LABEL_STYLE)
ax1.set_title("L1 Regularization (Lasso) — Coefficient Paths", **TITLE_STYLE)
ax1.tick_params(axis="both", **TICK_STYLE)
ax1.xaxis.set_tick_params(which="both", colors="#8b949e")
 
# ── Subplot 2: L2 (Ridge) ───────────────────────────────────────────────────
for i, fname in enumerate(feature_names):
    ax2.plot(C_values, coefs_l2[:, i],
             color=feat_color[fname], alpha=0.85, lw=1.8, label=fname)
 
ax2.axhline(0, color="#484f58", lw=0.8, ls="--")
ax2.set_xscale("log")
ax2.set_xlabel("C  (regularization strength ↑ as C → 0)", **LABEL_STYLE)
ax2.set_ylabel("Standardized Coefficient", **LABEL_STYLE)
ax2.set_title("L2 Regularization (Ridge) — Coefficient Paths", **TITLE_STYLE)
ax2.tick_params(axis="both", **TICK_STYLE)
 
# ── Shared Legend ─────────────────────────────────────────────────────────────
handles, labels = ax1.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=min(n_features, 6),
    bbox_to_anchor=(0.5, -0.14),
    framealpha=0.15,
    facecolor="#161b22",
    edgecolor="#30363d",
    fontsize=8,
    labelcolor="#c9d1d9",
)
 
# ── Annotation Box ───────────────────────────────────────────────────────────
if zeroed_features:
    note_lines = ["L1 features zeroed out (strongest → weakest reg):"]
    for fname, c_val in zeroed_sorted[:6]:
        note_lines.append(f"  • {fname}  (C≈{c_val:.3f})")
    if len(zeroed_sorted) > 6:
        note_lines.append(f"  ... and {len(zeroed_sorted)-6} more")
    note_text = "\n".join(note_lines)
    ax1.text(
        0.02, 0.02, note_text,
        transform=ax1.transAxes,
        fontsize=7.5, color="#c9d1d9",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d",
                  edgecolor="#30363d", alpha=0.9),
    )
 
plt.suptitle(
    "Regularization Path Explorer — Telecom Churn Dataset",
    color="#e6edf3", fontsize=15, fontweight="bold", y=1.02
)
 
plt.tight_layout()
output_path = r"C:\Users\User\M5\regularization-explorer\regularization_path.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"\nPlot saved to: {output_path}")
plt.show()