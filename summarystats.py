# ============================================================
# TrackMan Pitching Outputs (Box Scores + Strike Zone + Pitch Usage Per Outing)
# For: Nick Bonn + Cordon Pettey
# Copy/paste into a Jupyter notebook cell and run.
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

# ----------------------------
# 0) CONFIG
# ----------------------------
DATA_DIR = "./trackman_files"

# (kept, but nothing is saved unless you choose to later)
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_PITCHERS = ["Bonn", "Pettey"]

SZ_LEFT, SZ_RIGHT = -0.83, 0.83
SZ_BOT, SZ_TOP = 1.5, 3.5

HIT_RESULTS = {"Single", "Double", "Triple", "HomeRun"}

# ----------------------------
# FIXED PITCH COLORS (BASED ON YOUR ACTUAL DATA)
# ----------------------------
PITCH_COLOR_MAP = {
    "Four-Seam": "#C8102E",   # FF - Red
    "Sinker": "#FF8C00",      # SI - Bright Orange
    "Changeup": "#1FB82E",    # CH - Green
    "Curveball": "#2CB1C9",   # CU - Cyan
    "Slider": "#F4EA00",      # SL - Yellow
    "Splitter": "#3E9C9C",    # FS - Teal
    "Cutter": "#8B2E1E",      # FC - Brown
    "Unknown": "white"
}

# ----------------------------
# 1) LOAD ALL CSVs
# ----------------------------
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in: {DATA_DIR}")

dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    df["__source_file__"] = os.path.basename(f)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

for col in ["Date", "UTCDate"]:
    if col in data.columns:
        data[col] = pd.to_datetime(data[col], errors="coerce")

num_cols = [
    "OutsOnPlay", "RunsScored", "PlateLocSide", "PlateLocHeight",
    "RelSpeed", "ZoneSpeed", "SpinRate"
]
for c in num_cols:
    if c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")

# ----------------------------
# 2) FIND THE EXACT PITCHER NAME STRINGS PRESENT
# ----------------------------
if "Pitcher" not in data.columns:
    raise ValueError("Column 'Pitcher' not found. Verify CSV headers match TrackMan export.")

unique_pitchers = sorted([p for p in data["Pitcher"].dropna().unique()])
matches = [p for p in unique_pitchers if any(t.lower() in str(p).lower() for t in TARGET_PITCHERS)]

print("Pitchers in data matching TARGET_PITCHERS:")
for p in matches:
    print(" -", p)

if not matches:
    raise ValueError("No pitchers matched. Update TARGET_PITCHERS to match names in your CSVs.")

PITCHER_LIST = matches

# ----------------------------
# 3) HELPER FUNCTIONS
# ----------------------------
def opponent_for_row(r):
    ht, at, pt = r.get("HomeTeam"), r.get("AwayTeam"), r.get("PitcherTeam")
    if pd.isna(ht) or pd.isna(at) or pd.isna(pt):
        return np.nan
    return at if pt == ht else ht

def ip_from_outs(outs: int) -> str:
    if outs is None or (isinstance(outs, float) and np.isnan(outs)):
        return "0.0"
    outs = int(outs)
    whole = outs // 3
    rem = outs % 3
    return f"{whole}.{rem}"

def is_strike_call(pitch_call: str) -> bool:
    if not isinstance(pitch_call, str):
        return False
    if pitch_call in {"StrikeCalled", "StrikeSwinging", "InPlay"}:
        return True
    if pitch_call.startswith("FoulBall"):
        return True
    return False

def is_ball_call(pitch_call: str) -> bool:
    if not isinstance(pitch_call, str):
        return False
    return pitch_call in {"BallCalled", "BallinDirt", "BallIntentional"}

def is_swing(pitch_call: str) -> bool:
    if not isinstance(pitch_call, str):
        return False
    return (
        pitch_call == "StrikeSwinging"
        or pitch_call.startswith("FoulBall")
        or pitch_call == "InPlay"
    )

def pick_pitch_type(row):
    apt = row.get("AutoPitchType")
    tpt = row.get("TaggedPitchType")
    if isinstance(apt, str) and apt.strip() and apt != "Undefined":
        return apt
    if isinstance(tpt, str) and tpt.strip() and tpt != "Undefined":
        return tpt
    return "Unknown"

# Add opponent + pitch_type columns
data["Opponent"] = data.apply(opponent_for_row, axis=1)
data["PitchType"] = data.apply(pick_pitch_type, axis=1)

# ----------------------------
# 4) BOX SCORE GAME-BY-GAME FOR EACH PITCHER (+ 1st pitch strike + whiff)
# ----------------------------
# ============================================================
# DROP-IN REPLACEMENT: BOX SCORE SECTION (Fixes IP using Outs state transitions)
# Paste this to REPLACE your entire "4) BOX SCORE..." section
# (from "# ---------------------------- # 4) (1) BOX SCORE..." down through the display(box[...]) line)
# ============================================================

# ============================================================
# DROP-IN REPLACEMENT: BOX SCORE SECTION (Fixes IP using event-based outs)
# Uses: Strikeouts + OutsOnPlay on balls in play (fallback to PlayResult)
# ============================================================

from IPython.display import display

# Filter to our pitchers
pdat = data[data["Pitcher"].isin(PITCHER_LIST)].copy()

# ---------- Per-pitch flags ----------
pdat["is_K"] = (pdat.get("KorBB") == "Strikeout")
pdat["is_BB"] = (pdat.get("KorBB") == "Walk")
pdat["is_HBP"] = (pdat.get("PitchCall") == "HitByPitch")
pdat["is_hit"] = pdat.get("PlayResult").isin(HIT_RESULTS) if "PlayResult" in pdat.columns else False
pdat["is_HR"] = (pdat.get("PlayResult") == "HomeRun") if "PlayResult" in pdat.columns else False
pdat["is_strike"] = pdat.get("PitchCall").apply(is_strike_call) if "PitchCall" in pdat.columns else False
pdat["is_ball"] = pdat.get("PitchCall").apply(is_ball_call) if "PitchCall" in pdat.columns else False

# First pitch strike
pdat["is_first_pitch"] = (pdat["PitchofPA"] == 1)
pdat["is_first_pitch_strike"] = pdat["is_first_pitch"] & (pdat["is_strike"] == True)

# Whiff (swing & miss)
pdat["is_swing"] = pdat["PitchCall"].apply(is_swing) if "PitchCall" in pdat.columns else False
pdat["is_whiff"] = (pdat["PitchCall"] == "StrikeSwinging") if "PitchCall" in pdat.columns else False

# ---------- Outs recorded (event-based, robust) ----------
# This is the key fix: compute outs credited to the pitcher from events.
# Priority:
#   1) Strikeout => 1 out
#   2) InPlay with OutsOnPlay => use OutsOnPlay (captures DP as 2)
#   3) InPlay with PlayResult "Out"/"Sacrifice"/"FieldersChoice" => assume 1 out
#   4) Else => 0
OUT_PLAYRESULTS_1OUT = {"Out", "Sacrifice", "FieldersChoice"}  # adjust if your data uses more labels

def outs_recorded_on_pitch(row) -> int:
    # Strikeout always 1 out
    if row.get("KorBB") == "Strikeout":
        return 1

    pitch_call = row.get("PitchCall")
    play_result = row.get("PlayResult")
    outs_on_play = row.get("OutsOnPlay")

    # If ball put in play, outs may happen
    if isinstance(pitch_call, str) and pitch_call == "InPlay":
        # Prefer OutsOnPlay if available and valid
        if outs_on_play is not None and not (isinstance(outs_on_play, float) and np.isnan(outs_on_play)):
            try:
                o = int(outs_on_play)
                # OutsOnPlay can be 0-3; keep sane range
                return max(0, min(3, o))
            except Exception:
                pass

        # Fallback: infer 1 out from PlayResult labels that indicate an out
        if isinstance(play_result, str) and play_result in OUT_PLAYRESULTS_1OUT:
            return 1

    return 0

# Make sure OutsOnPlay is numeric if present
if "OutsOnPlay" in pdat.columns:
    pdat["OutsOnPlay"] = pd.to_numeric(pdat["OutsOnPlay"], errors="coerce")

pdat["OutsRecorded"] = pdat.apply(outs_recorded_on_pitch, axis=1)

# ---------- Grouping keys ----------
group_cols = ["Pitcher", "GameID"]
for c in ["Date", "Stadium", "Level", "League", "HomeTeam", "AwayTeam", "Opponent", "__source_file__"]:
    if c in pdat.columns:
        group_cols.append(c)

# ---------- Aggregate boxscore ----------
agg = {
    "OutsRecorded": "sum",
    "is_K": "sum",
    "is_BB": "sum",
    "is_HBP": "sum",
    "is_hit": "sum",
    "is_HR": "sum",
    "RunsScored": "sum" if "RunsScored" in pdat.columns else "size",
    "PitchUID": "count" if "PitchUID" in pdat.columns else "size",
    "is_strike": "sum",
    "is_ball": "sum",
    "is_first_pitch": "sum",
    "is_first_pitch_strike": "sum",
    "is_swing": "sum",
    "is_whiff": "sum",
}

box = pdat.groupby(group_cols, dropna=False).agg(agg).reset_index()

# Rename columns
box = box.rename(columns={
    "OutsRecorded": "Outs",
    "is_K": "K",
    "is_BB": "BB",
    "is_HBP": "HBP",
    "is_hit": "H",
    "is_HR": "HR",
    "RunsScored": "R",
    "PitchUID": "Pitches",
    "is_strike": "Strikes",
    "is_ball": "Balls"
})

# IP from Outs
box["IP"] = box["Outs"].apply(ip_from_outs)

# Rates
box["Strike%"] = np.where(box["Pitches"] > 0, (box["Strikes"] / box["Pitches"]) * 100, np.nan).round(1)
box["1stPitchStrike%"] = np.where(
    box["is_first_pitch"] > 0,
    (box["is_first_pitch_strike"] / box["is_first_pitch"]) * 100,
    np.nan
).round(1)
box["Whiff%"] = np.where(
    box["is_swing"] > 0,
    (box["is_whiff"] / box["is_swing"]) * 100,
    np.nan
).round(2)

# Sort nicely
sort_cols = ["Pitcher"]
if "Date" in box.columns:
    sort_cols.append("Date")
sort_cols.append("GameID")
box = box.sort_values(sort_cols).reset_index(drop=True)

# Display
print("\n=== BOX SCORE SUMMARY (game-by-game) ===")
display_cols = [c for c in [
    "Pitcher", "Date", "GameID", "Opponent", "IP",
    "Pitches", "Strikes", "Strike%", "1stPitchStrike%", "Whiff%",
    "K", "BB", "HBP", "H", "HR", "R",
    "Stadium", "Level"
] if c in box.columns]
display(box[display_cols])
# ----------------------------
# 5) PITCH USAGE BREAKDOWN TABLE PER OUTING
# ----------------------------
def pitch_usage_breakdown_table(df_outing):
    d = df_outing.copy()

    if "PitchType" not in d.columns:
        d["PitchType"] = "Unknown"
    if "RelSpeed" not in d.columns:
        d["RelSpeed"] = np.nan
    if "is_swing" not in d.columns:
        d["is_swing"] = False
    if "is_whiff" not in d.columns:
        d["is_whiff"] = False

    total_pitches = len(d)

    out = (
        d.groupby("PitchType", dropna=False)
         .agg(
            Pitches=("PitchType", "size"),
            Swings=("is_swing", "sum"),
            Whiffs=("is_whiff", "sum"),
            AvgVelo=("RelSpeed", "mean")
         )
         .reset_index()
    )

    out["Usage%"] = (out["Pitches"] / total_pitches * 100).round(1)
    out["Whiff%"] = np.where(out["Swings"] > 0, (out["Whiffs"] / out["Swings"] * 100), np.nan).round(1)
    out["AvgVelo"] = out["AvgVelo"].round(1)

    out = out.sort_values(["Usage%", "PitchType"], ascending=[False, True]).reset_index(drop=True)
    out = out[["PitchType", "Pitches", "Usage%", "AvgVelo", "Swings", "Whiffs", "Whiff%"]]
    return out

# ----------------------------
# 6) STRIKE ZONE PLOTS PER OUTING
#   - Balls = hollow circles
#   - Strikes = solid circles
#   - Hits = X
#   - Color-coded by pitch type
#   - Solid white zone lines
#   - NO SAVING (inline only)
# ----------------------------
def draw_strikezone(ax, left=SZ_LEFT, right=SZ_RIGHT, bot=SZ_BOT, top=SZ_TOP):
    ax.plot([left, right, right, left, left], [bot, bot, top, top, bot], linewidth=2.5, color="white")
    x1 = left + (right-left)/3
    x2 = left + 2*(right-left)/3
    y1 = bot + (top-bot)/3
    y2 = bot + 2*(top-bot)/3
    ax.plot([x1, x1], [bot, top], linewidth=1.5, color="white")
    ax.plot([x2, x2], [bot, top], linewidth=1.5, color="white")
    ax.plot([left, right], [y1, y1], linewidth=1.5, color="white")
    ax.plot([left, right], [y2, y2], linewidth=1.5, color="white")

def plot_outing(df_outing, pitcher_name, title_suffix):
    d = df_outing.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    if d.empty:
        print(f"Skipping (no PlateLocSide/Height): {pitcher_name} {title_suffix}")
        return

    # ----------------------------
    # FIXED COLOR MAP (CONSISTENT)
    # ----------------------------
    PITCH_COLOR_MAP = {
        "Four-Seam": "#C8102E",   # Red
        "Sinker": "#FF8C00",      # Orange
        "Changeup": "#1FB82E",    # Green
        "Curveball": "#2CB1C9",   # Cyan
        "Slider": "#F4EA00",      # Yellow
        "Splitter": "#3E9C9C",    # Teal
        "Cutter": "#8B2E1E",      # Brown
        "Unknown": "white"
    }

    plt.figure(figsize=(14, 7))
    ax = plt.gca()
    ax.set_facecolor("black")
    plt.gcf().patch.set_facecolor("black")
    ax.grid(True, alpha=0.18, color="white")

    # Use fixed mapping instead of dynamic colors
    pitch_types = sorted(d["PitchType"].fillna("Unknown").unique())
    color_map = {pt: PITCH_COLOR_MAP.get(pt, "white") for pt in pitch_types}

    for pt in pitch_types:
        sub = d[d["PitchType"] == pt]

        hits = sub[sub["is_hit"] == True]
        non_hits = sub[sub["is_hit"] == False]

        strikes = non_hits[non_hits["is_strike"] == True]
        balls = non_hits[non_hits["is_strike"] == False]

        # Balls: hollow circles
        if not balls.empty:
            ax.scatter(
                balls["PlateLocSide"], balls["PlateLocHeight"],
                s=160,
                marker="o",
                facecolors="none",
                edgecolors=color_map[pt],
                linewidths=2,
                label=f"{pt} (Ball)"
            )

        # Strikes: solid circles
        if not strikes.empty:
            ax.scatter(
                strikes["PlateLocSide"], strikes["PlateLocHeight"],
                s=160,
                marker="o",
                facecolors=color_map[pt],
                edgecolors=color_map[pt],
                linewidths=1,
                label=f"{pt} (Strike)"
            )

        # Hits: X
        if not hits.empty:
            ax.scatter(
                hits["PlateLocSide"], hits["PlateLocHeight"],
                s=220,
                marker="X",
                color=color_map[pt],
                linewidths=2,
                label=f"{pt} (Hit)"
            )

    draw_strikezone(ax)

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(1.0, 4.0)
    ax.set_xlabel("Horizontal Location (ft)", color="white", fontsize=12)
    ax.set_ylabel("Vertical Location (ft)", color="white", fontsize=12)
    ax.tick_params(colors="white")

    ax.set_title(
        f"{pitcher_name} — Pitch Locations (Outing: {title_suffix})",
        color="white",
        fontsize=16,
        loc="left",
        pad=18
    )

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    new_h, new_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            new_h.append(h)
            new_l.append(l)

    leg = ax.legend(
        new_h,
        new_l,
        title="Pitch Type",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    plt.setp(leg.get_texts(), color="white")
    plt.setp(leg.get_title(), color="white")

    plt.tight_layout()
    plt.show()

# ----------------------------
# 7) RUN PER-OUTING OUTPUTS (TABLE + PLOT)
# ----------------------------
outing_groups = ["Pitcher", "GameID"]
if "Date" in pdat.columns:
    outing_groups.append("Date")

for (pitcher, gameid, *rest), g in pdat.groupby(outing_groups, dropna=False):
    date_str = ""
    if rest:
        dt = rest[0]
        if pd.notna(dt):
            date_str = pd.to_datetime(dt).strftime("%Y-%m-%d")

    opp = g["Opponent"].dropna().unique()
    opp_str = opp[0] if len(opp) else "UnknownOpp"
    suffix = f"{date_str} vs {opp_str} ({gameid})".strip()

    print("\n" + "=" * 90)
    print(f"Pitch Usage Breakdown — {pitcher} — {suffix}")
    display(pitch_usage_breakdown_table(g))

        # ----------------------------
    # Show Box Score For This Outing Only
    # ----------------------------
    box_row = box[
        (box["Pitcher"] == pitcher) &
        (box["GameID"] == gameid)
    ]

    print("\nBOX SCORE SUMMARY")
    display_cols = [c for c in [
        "IP", "Pitches", "Strikes", "Strike%",
        "1stPitchStrike%", "Whiff%",
        "K", "BB", "HBP", "H", "HR", "R"
    ] if c in box_row.columns]

    display(box_row[display_cols])

    plot_outing(g, pitcher, suffix)

print("\nFinished: box score + pitch usage tables + strike zone plots (inline only).")
