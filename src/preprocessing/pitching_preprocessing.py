import numpy as np
import pandas as pd


def add_starter_indicator_pitchlevel(
    df: pd.DataFrame,
    game_id_col: str = "game_id",
    pitcher_col: str = "pitcher_name",
    game_date_col: str = "game_date",
    inning_col: str = "inning",
    outs_when_up_col: str = "outs_when_up",
    pitch_number_col: str = "pitch_number",
    pitching_team_col: str = "pitching_team",
    inning_topbot_col: str = "inning_topbot",
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
) -> pd.DataFrame:
    """
    Pitch-level starter flag (safe + simple):

    - Ensures `pitching_team` exists (derives from inning_topbot/home_team/away_team if missing).
    - Defines starter as the pitcher on the earliest pitch for each (game_id, pitching_team).
    - Enforces: ALL rows for that pitcher in that (game_id, pitching_team) get is_starter=1 (no flip-backs).
    """
    out = df.copy()

    # keep tidy; not required for starter logic
    if game_date_col in out.columns:
        out[game_date_col] = pd.to_datetime(out[game_date_col], errors="coerce")

    # Ensure pitching_team exists
    if pitching_team_col not in out.columns:
        topbot = out[inning_topbot_col].astype("string")
        out[pitching_team_col] = out[home_team_col]
        out.loc[topbot.eq("Bot"), pitching_team_col] = out.loc[topbot.eq("Bot"), away_team_col]

    # Deterministic ordering to find the first pitch for each team in each game
    sort_cols = [game_id_col, pitching_team_col, inning_col, outs_when_up_col, pitch_number_col]
    out = out.sort_values(sort_cols, kind="mergesort")

    # Identify starter pitcher per (game_id, pitching_team)
    starter_lookup = (
        out.groupby([game_id_col, pitching_team_col], sort=False)[pitcher_col]
           .first()
           .rename("starter_pitcher")
           .reset_index()
    )

    # Enforce is_starter based on pitcher identity (no flip-backs possible)
    out = out.merge(starter_lookup, on=[game_id_col, pitching_team_col], how="left")
    out["is_starter"] = (out[pitcher_col] == out["starter_pitcher"]).astype(int)

    return out.drop(columns=["starter_pitcher"])


def add_starter_full_game_indicator(
    df: pd.DataFrame,
    game_id_col: str = "game_id",
    pitching_team_col: str = "pitching_team",
    pitcher_col: str = "pitcher_name",
) -> pd.DataFrame:
    """
    Adds `starter_full_game` indicator at pitch-level.

    Definition:
        starter_full_game = 1 if the pitching_team used exactly
        ONE pitcher in that game (i.e., no bullpen usage).

    This is the most robust definition for validating missing bullpen metrics.

    Returns:
        Original dataframe with new column:
            - starter_full_game (0/1)
    """
    out = df.copy()

    # Count distinct pitchers used per (game_id, pitching_team)
    pitcher_counts = (
        out.groupby([game_id_col, pitching_team_col])[pitcher_col]
           .nunique()
           .rename("num_pitchers_used")
           .reset_index()
    )

    # Full game = exactly one pitcher used
    pitcher_counts["starter_full_game"] = (
        pitcher_counts["num_pitchers_used"] == 1
    ).astype(int)

    # Merge back
    out = out.merge(
        pitcher_counts[[game_id_col, pitching_team_col, "starter_full_game"]],
        on=[game_id_col, pitching_team_col],
        how="left"
    )

    return out


def add_pitching_indicators(
    pa_df: pd.DataFrame,
    events_col: str = "events",
    inning_topbot_col: str = "inning_topbot",
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
) -> pd.DataFrame:
    """
    Add minimal PA-level indicators for pitching (keep the PA table light).

    Designed for pitcher metric aggregation later (FIP/WHIP/K9/HR9, bullpen vs starter splits, etc.).
    Uses `events` (PA outcome), NOT pitch-level `description`.

    Adds:
      - pitching_team: team on defense for this PA
      - did_not_end_pa: 1 if events is NA/None (fallback row / not a real PA-ending event)
      - is_k, is_bb, is_hbp, is_hr, is_h (hit)
      - outs: outs recorded on the PA (0/1/2/3), suitable for IP = outs/3
      - is_pa_countable: drop truncated/NA PAs cleanly later
    """
    out = pa_df.copy()
    ev = out[events_col].astype("string")

    # Team on defense (pitching team) for each PA
    mask_bot = out[inning_topbot_col].astype("string").eq("Bot")
    out["pitching_team"] = out[home_team_col]
    out.loc[mask_bot, "pitching_team"] = out.loc[mask_bot, away_team_col]

    # Flag rows that are NOT true PA-ending events (events is missing)
    out["did_not_end_pa"] = ev.isna().astype(int)

    # Completed PA flag (countable for rate stats)
    out["is_pa_countable"] = (~ev.eq("truncated_pa")) & ev.notna()

    # Core FIP components (NA-safe)
    out["is_hr"]  = ev.eq("home_run").fillna(False).astype(int)
    out["is_bb"]  = ev.isin(["walk", "intent_walk"]).fillna(False).astype(int)
    out["is_hbp"] = ev.eq("hit_by_pitch").fillna(False).astype(int)
    out["is_k"]   = ev.isin(["strikeout", "strikeout_double_play"]).fillna(False).astype(int)

    # Hits allowed (for WHIP; HR is also a hit) (NA-safe)
    out["is_h"] = ev.isin(["single", "double", "triple", "home_run"]).fillna(False).astype(int)

    # Outs on the play (0/1/2/3)
    outs_map = {
        # 1 out
        "field_out": 1,
        "force_out": 1,
        "fielders_choice_out": 1,
        "strikeout": 1,
        "sac_fly": 1,
        "sac_bunt": 1,

        # 2 outs
        "double_play": 2,
        "grounded_into_double_play": 2,
        "strikeout_double_play": 2,
        "sac_fly_double_play": 2,

        # 3 outs
        "triple_play": 3,
    }
    out["outs"] = ev.map(outs_map).fillna(0).astype(int)

    return out


def split_starter_bullpen(
    df: pd.DataFrame,
    is_starter_col: str = "is_starter",
    countable_col: str | None = "is_pa_countable",
    fallback_col: str | None = "did_not_end_pa",   # NEW
    game_id_col: str = "game_id",
    pitching_team_col: str = "pitching_team",
    pitcher_col: str = "pitcher_name",
    validate: bool = True,
):
    """
    Returns (starter_df, bullpen_df).

    If countable_col exists and is not None, filters to:
      - countable PAs OR
      - fallback rows (did_not_end_pa == 1), if fallback_col provided.

    Validation:
      1) Split partitions filtered rows
      2) Exactly one starter pitcher identity per (game_id, pitching_team)
    """
    out = df.copy()

    if countable_col is not None and countable_col in out.columns:
        keep = out[countable_col].astype(bool)

        # keep fallback rows too (so starters with no PA-ending events don't disappear)
        if fallback_col is not None and fallback_col in out.columns:
            keep = keep | (out[fallback_col].astype(int) == 1)

        out = out.loc[keep].copy()

    starter_df = out.loc[out[is_starter_col].astype(int) == 1].copy()
    bullpen_df = out.loc[out[is_starter_col].astype(int) == 0].copy()

    if validate:
        if len(starter_df) + len(bullpen_df) != len(out):
            raise ValueError(
                f"Split does not partition rows cleanly: "
                f"starter({len(starter_df)}) + bullpen({len(bullpen_df)}) != out({len(out)})"
            )

        nunique = (
            starter_df.groupby([game_id_col, pitching_team_col])[pitcher_col]
            .nunique(dropna=False)
        )
        bad = nunique[nunique != 1]
        if not bad.empty:
            raise ValueError(
                f"Found {len(bad)} (game_id, pitching_team) groups with starter pitcher nunique != 1. "
                f"Example:\n{bad.head(10)}"
            )

    return starter_df, bullpen_df


def aggregate_pitching_game_lines(
    pa_pitching: pd.DataFrame,
    pitcher_id_col: str | None = None,     # e.g., "pitcher" or "pitcher_id"
    pitcher_name_col: str | None = None    # optional, e.g., "player_name"
) -> pd.DataFrame:
    """
    Aggregate PA-level pitching data to game lines.

    Two modes:
    1) Team-by-game (default): one row per (game_id, pitching_team, is_home_team, pitcher_role)
    2) Pitcher-by-game: if pitcher_id_col is provided, one row per
       (game_id, pitching_team, is_home_team, pitcher_role, pitcher_id)

    Always returns:
      game_id, game_date, pitching_team, home_team, away_team, is_home_team, pitcher_role,
      [pitcher_id], [pitcher_name], IP, H, BB, HBP, K, HR,
      did_not_end_pa,
      starter_full_game

    Notes:
    - Stats come from is_pa_countable == True rows
    - pitcher_role = "starter" if is_starter == 1 else "bullpen"
    - IP computed from outs: IP = sum(outs)/3
    - did_not_end_pa is a QC flag aggregated from ALL rows (1 if any did_not_end_pa rows exist)
    - starter_full_game is carried through if present (team used exactly one pitcher that game)
    """
    df = pa_pitching.copy()

    required = [
        "game_id", "game_date", "pitching_team", "home_team", "away_team",
        "is_starter", "is_pa_countable",
        "outs", "is_h", "is_bb", "is_hbp", "is_k", "is_hr"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if pitcher_id_col is not None and pitcher_id_col not in df.columns:
        raise ValueError(
            f"pitcher_id_col='{pitcher_id_col}' not found in columns. "
            f"Pass the correct pitcher id column name."
        )

    if pitcher_name_col is not None and pitcher_name_col not in df.columns:
        raise ValueError(
            f"pitcher_name_col='{pitcher_name_col}' was provided but not found in columns."
        )

    # Normalize role
    df["is_starter"] = pd.to_numeric(df["is_starter"], errors="coerce").fillna(0).astype(int)
    df["pitcher_role"] = np.where(df["is_starter"] == 1, "starter", "bullpen")

    # Home-team indicator (1/0)
    df["is_home_team"] = (df["pitching_team"] == df["home_team"]).astype(int)

    # Grouping grain
    group_cols = ["game_id", "pitching_team", "is_home_team", "pitcher_role"]
    if pitcher_id_col is not None:
        group_cols.append(pitcher_id_col)
        if pitcher_name_col is not None:
            group_cols.append(pitcher_name_col)

    # --- 1) Build skeleton from ALL rows (countable + fallback) ---
    agg_dict = {
        "game_date": ("game_date", "min"),
        "home_team": ("home_team", "first"),
        "away_team": ("away_team", "first"),
    }
    if "starter_full_game" in df.columns:
        agg_dict["starter_full_game"] = ("starter_full_game", "max")

    skeleton = (
        df.groupby(group_cols, as_index=False)
          .agg(**agg_dict)
    )

    # --- 2) Aggregate stats from countable PA rows only ---
    stats_df = df[df["is_pa_countable"] == True].copy()

    num_cols = ["outs", "is_h", "is_bb", "is_hbp", "is_k", "is_hr"]
    for c in num_cols:
        stats_df[c] = pd.to_numeric(stats_df[c], errors="coerce").fillna(0)

    stats = (
        stats_df.groupby(group_cols, as_index=False)
          .agg(
              outs=("outs", "sum"),
              H=("is_h", "sum"),
              BB=("is_bb", "sum"),
              HBP=("is_hbp", "sum"),
              K=("is_k", "sum"),
              HR=("is_hr", "sum"),
          )
    )

    # --- 3) Aggregate did_not_end_pa as a binary QC flag from ALL rows (if present) ---
    if "did_not_end_pa" in df.columns:
        df["did_not_end_pa"] = pd.to_numeric(df["did_not_end_pa"], errors="coerce").fillna(0).astype(int)
        qc_flag = (
            df.groupby(group_cols, as_index=False)
              .agg(did_not_end_pa=("did_not_end_pa", "max"))  # 1 if any exist, else 0
        )
    else:
        qc_flag = None

    # --- 4) Merge skeleton + stats (+ QC flag), fill missing with 0 ---
    game_lines = skeleton.merge(stats, on=group_cols, how="left")
    for c in ["outs", "H", "BB", "HBP", "K", "HR"]:
        game_lines[c] = game_lines[c].fillna(0)

    if qc_flag is not None:
        game_lines = game_lines.merge(qc_flag, on=group_cols, how="left")
        game_lines["did_not_end_pa"] = game_lines["did_not_end_pa"].fillna(0).astype(int)
    else:
        game_lines["did_not_end_pa"] = 0

    # Ensure starter_full_game exists (default 0 if absent upstream)
    if "starter_full_game" not in game_lines.columns:
        game_lines["starter_full_game"] = 0
    else:
        game_lines["starter_full_game"] = pd.to_numeric(
            game_lines["starter_full_game"], errors="coerce"
        ).fillna(0).astype(int)

    game_lines["IP"] = game_lines["outs"] / 3.0
    game_lines = game_lines.drop(columns=["outs"])

    # Column order
    cols = [
        "game_id", "game_date",
        "pitching_team", "home_team", "away_team",
        "is_home_team", "pitcher_role",
    ]
    if pitcher_id_col is not None:
        cols.append(pitcher_id_col)
        if pitcher_name_col is not None:
            cols.append(pitcher_name_col)

    cols += [
        "IP", "H", "BB", "HBP", "K", "HR",
        "did_not_end_pa",
        "starter_full_game",
    ]

    game_lines = (
        game_lines[cols]
        .sort_values(["game_date", "game_id", "pitching_team", "is_home_team", "pitcher_role"])
        .reset_index(drop=True)
    )

    # Uniqueness guardrail at the intended grain
    key_cols = ["game_id", "pitching_team", "is_home_team", "pitcher_role"]
    if pitcher_id_col is not None:
        key_cols.append(pitcher_id_col)

    dupes = game_lines.duplicated(key_cols).sum()
    if dupes:
        raise ValueError(f"Expected unique rows by {key_cols} but found {dupes} duplicates.")

    return game_lines


def add_rolling_pitching_counts(
    df: pd.DataFrame,
    windows: tuple[str, ...] = ("3D", "7D"),          # accepts ("3D","7D") or ("3G","7G") or ("3","7")
    pitcher_col: str = "pitcher_name",
    team_col: str = "pitching_team",
    date_col: str = "game_date",
    game_id_col: str = "game_id",
    role_col: str = "pitcher_role",
    mode: str = "auto",                              # "auto" | "starter" | "bullpen"
    include_current_game: bool = False,              # False = only prior games in the window
    starter_tag: str = "starter",
    bullpen_tag: str = "bullpen",
    base_stats: tuple[str, ...] = ("IP", "H", "BB", "HBP", "K", "HR"),
    output_window_suffix: str = "G",                 # always label outputs as games: roll_3G_...
) -> pd.DataFrame:
    """
    Add rolling pitching COUNT stats over the last N GAMES (not calendar days).

    Key behavior
    - windows accepts labels like ("3D","7D") for backward-compatibility, BUT is interpreted as N games.
      Output columns are labeled with games by default (e.g., roll_3G_starter_IP).
    - include_current_game=False prevents leakage by excluding the current game
      from the rolling window (uses a shift(1) before rolling).

    Mode
    - mode="starter": filter to rows where role_col == "starter" (if role_col exists),
      group by pitcher_name, output tag starter_tag
    - mode="bullpen": filter to rows where role_col == "bullpen" (if role_col exists),
      group by pitching_team, output tag bullpen_tag
    - mode="auto": if role_col exists and ANY starter rows exist -> starter grouping,
      else bullpen grouping (team)

    Output columns
    - starters: roll_{n}{output_window_suffix}_{starter_tag}_{STAT}
    - bullpen:  roll_{n}{output_window_suffix}_{bullpen_tag}_{STAT}

    Rolling stats are SUMS over the last N games in the group.
    """

    def _parse_window_to_n_and_label(w: str) -> tuple[int, str]:
        """
        Convert window label to (n_games, output_label).
        Accepts: '3D', '3G', '3', '03D', etc.
        Always outputs '{n}{output_window_suffix}' as the label.
        """
        s = str(w).strip().upper()
        if s.endswith(("D", "G")):
            s_num = s[:-1]
        else:
            s_num = s

        try:
            n = int(s_num)
        except Exception as e:
            raise ValueError(f"Invalid window label '{w}'. Expected like '3D', '3G', or '3'.") from e

        if n <= 0:
            raise ValueError(f"Window must be positive. Got '{w}'.")

        return n, f"{n}{output_window_suffix}"

    if mode not in {"auto", "starter", "bullpen"}:
        raise ValueError('mode must be one of: "auto", "starter", "bullpen"')

    out = df.copy()

    # --- basic checks ---
    for c in [team_col, date_col, game_id_col]:
        if c not in out.columns:
            raise ValueError(f"Missing required column: '{c}'")

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if out[date_col].isna().any():
        bad = out.loc[out[date_col].isna(), [date_col, game_id_col]].head(5)
        raise ValueError(f"Found non-parseable {date_col} values. Examples:\n{bad}")

    # Validate required stat columns
    missing_base = [c for c in base_stats if c not in out.columns]
    if missing_base:
        raise ValueError(f"Missing required stat columns for rolling: {missing_base}")

    # Coerce base stats numeric
    for c in base_stats:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # --- decide mode + grouping ---
    has_pitcher = pitcher_col in out.columns
    has_role = role_col in out.columns

    if mode == "starter":
        if not has_pitcher:
            raise ValueError(f"mode='starter' requires column '{pitcher_col}'")
        if has_role:
            out = out.loc[out[role_col].astype("string") == "starter"].copy()
        group_keys = [pitcher_col]
        sort_cols = [pitcher_col, date_col, game_id_col]
        tag = starter_tag

    elif mode == "bullpen":
        if has_role:
            out = out.loc[out[role_col].astype("string") == "bullpen"].copy()
        group_keys = [team_col]
        sort_cols = [team_col, date_col, game_id_col]
        tag = bullpen_tag

    else:
        # auto
        is_starter_mode = False
        if has_pitcher and has_role:
            is_starter_mode = out[role_col].astype("string").eq("starter").any()

        if is_starter_mode:
            group_keys = [pitcher_col]
            sort_cols = [pitcher_col, date_col, game_id_col]
            tag = starter_tag
        else:
            group_keys = [team_col]
            sort_cols = [team_col, date_col, game_id_col]
            tag = bullpen_tag

    # Sort before rolling (deterministic)
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    def _add_rolls(g: pd.DataFrame) -> pd.DataFrame:
        # Ensure group sorted by time
        g = g.sort_values([date_col, game_id_col], kind="mergesort").reset_index(drop=True).copy()

        # Guardrail: within a group, each (game_date, game_id) should be unique (1 row per game)
        if g.duplicated([date_col, game_id_col]).any():
            examples = g.loc[g.duplicated([date_col, game_id_col]), [date_col, game_id_col]].head(5)
            raise ValueError(
                "Found duplicate (game_date, game_id) rows within a rolling group; expected one row per game.\n"
                f"Examples:\n{examples}"
            )

        src = g[list(base_stats)].copy()

        # Exclude current game to avoid leakage
        if not include_current_game:
            src = src.shift(1)

        for w in windows:
            n, out_label = _parse_window_to_n_and_label(w)

            rolled = src.rolling(window=n, min_periods=1).sum()
            rolled.columns = [f"roll_{out_label}_{tag}_{c}" for c in rolled.columns]

            g = pd.concat([g, rolled.reset_index(drop=True)], axis=1)

        return g

    out = (
        out.groupby(group_keys, group_keys=False, sort=False)
           .apply(_add_rolls)
           .reset_index(drop=True)
    )

    return out


def add_rate_metrics_from_rolled_counts(
    df: pd.DataFrame,
    windows: tuple[str, ...] = ("3D", "7D"),
    include_hbp_in_whip: bool = True,
    fip_constant: float | None = None,
    bullpen_tag: str = "bullpen",
    exclude_when_did_not_end_pa: bool = True,
    did_not_end_pa_col: str = "did_not_end_pa",
) -> pd.DataFrame:
    """
    Add WHIP, K9, HR9, and (raw) FIP using *rolled count columns*.

    Supports:
      - starter: roll_{w}_starter_IP/H/BB/HBP/K/HR -> roll_{w}_starter_WHIP/K9/HR9/FIP
      - bullpen: roll_{w}_bullpen_IP/H/BB/HBP/K/HR -> roll_{w}_bullpen_WHIP/K9/HR9/FIP

    Optional:
      - If exclude_when_did_not_end_pa is True and did_not_end_pa_col exists,
        sets the derived rate metrics to NaN where did_not_end_pa == 1.
        (Note: this does NOT remove those rows from the rolled counts upstream.)
    """
    out = df.copy()

    # Row-level exclusion mask (only affects the *current row's* derived rate metrics)
    exclude_mask = None
    if exclude_when_did_not_end_pa and did_not_end_pa_col in out.columns:
        exclude_mask = pd.to_numeric(out[did_not_end_pa_col], errors="coerce").fillna(0).astype(int).eq(1)

    def _compute_for_prefix(prefix: str, label: str) -> None:
        ip  = f"{prefix}IP"
        h   = f"{prefix}H"
        bb  = f"{prefix}BB"
        hbp = f"{prefix}HBP"
        k   = f"{prefix}K"
        hr  = f"{prefix}HR"

        needed = [ip, h, bb, k, hr]
        if include_hbp_in_whip:
            needed.append(hbp)

        if not all(col in out.columns for col in needed):
            return

        for col in set([ip, h, bb, hbp, k, hr]):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        ip_vals = out[ip]
        whip_num = out[h] + out[bb] + (out[hbp] if (include_hbp_in_whip and hbp in out.columns) else 0)

        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"{label}_WHIP"] = np.where(ip_vals > 0, whip_num / ip_vals, np.nan)
            out[f"{label}_K9"]   = np.where(ip_vals > 0, (out[k]  * 9.0) / ip_vals, np.nan)
            out[f"{label}_HR9"]  = np.where(ip_vals > 0, (out[hr] * 9.0) / ip_vals, np.nan)

            fip_num = 13.0 * out[hr] + 3.0 * (out[bb] + (out[hbp] if hbp in out.columns else 0)) - 2.0 * out[k]
            fip = np.where(ip_vals > 0, fip_num / ip_vals, np.nan)
            if fip_constant is not None:
                fip = fip + float(fip_constant)
            out[f"{label}_FIP"] = fip

        # If this row is a fallback/non-PA-ending row, blank the derived rates
        if exclude_mask is not None:
            for metric in ["WHIP", "K9", "HR9", "FIP"]:
                colname = f"{label}_{metric}"
                if colname in out.columns:
                    out.loc[exclude_mask, colname] = np.nan

    for w in windows:
        _compute_for_prefix(prefix=f"roll_{w}_starter_", label=f"roll_{w}_starter")
        _compute_for_prefix(prefix=f"roll_{w}_{bullpen_tag}_", label=f"roll_{w}_{bullpen_tag}")

    return out


def combine_game_level_pitching_rolling_rates(
    starter_df: pd.DataFrame,
    bullpen_df: pd.DataFrame,
    windows: tuple[str, ...] = ("3D", "7D"),
    metrics: tuple[str, ...] = ("WHIP", "K9", "HR9", "FIP"),
    game_id_col: str = "game_id",
    date_col: str = "game_date",
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
    is_home_col: str = "is_home_team",
    pitcher_name_col: str = "pitcher_name",
    starter_tag: str = "starter",
    bullpen_tag: str = "bullpen",
) -> pd.DataFrame:
    """
    Combine starter + bullpen rolling RATE metrics into ONE row per game_id.

    Key behavior:
    - Robust to games where one side has no bullpen row (complete games, data wrinkles).
    - Robust to rare cases where one side starter row is missing.
    - Preserves game_date, home_team, away_team.

    Strategy:
    - Split into home/away using is_home_team, then WIDEN with OUTER join.
    - Merge bullpen onto starters with LEFT join (starters are the game spine).
    """

    # -------------------------
    # Helpers
    # -------------------------
    def _require_cols(df: pd.DataFrame, cols: list[str], label: str) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{label} is missing columns: {missing}")

    def _make_wide(
        df: pd.DataFrame,
        value_cols: list[str],
        role_label: str,          # used only for error messages
        keep_pitcher_name: bool,  # True for starters
        rename_value_cols: bool,  # True if we want to rewrite roll_* columns
    ) -> pd.DataFrame:
        """
        Convert long (one row per team-game-side) -> wide (one row per game_id)
        using OUTER join so missing sides don't drop games.

        Returns columns:
          game_id, game_date, home_team, away_team,
          [starter_pitcher_name_home/away],
          <value_col>_home, <value_col>_away
        """
        base_cols = [game_id_col, date_col, home_team_col, away_team_col, is_home_col]
        extra_cols = [pitcher_name_col] if keep_pitcher_name else []
        use_cols = base_cols + extra_cols + value_cols
        _require_cols(df, use_cols, f"{role_label}")

        tmp = df[use_cols].copy()
        tmp[is_home_col] = tmp[is_home_col].astype(int)

        # Guardrail: for a given role, we expect at most one row per (game_id, side)
        dups = tmp.duplicated([game_id_col, is_home_col]).sum()
        if dups:
            raise ValueError(f"{role_label} has {dups} duplicates of (game_id, is_home_team).")

        home = tmp[tmp[is_home_col] == 1].drop(columns=[is_home_col])
        away = tmp[tmp[is_home_col] == 0].drop(columns=[is_home_col])

        # pitcher names only for starters
        if keep_pitcher_name:
            home = home.rename(columns={pitcher_name_col: "starter_pitcher_name_home"})
            away = away.rename(columns={pitcher_name_col: "starter_pitcher_name_away"})

        # Rename value columns with side suffix
        home = home.rename(columns={c: f"{c}_home" for c in value_cols})
        away = away.rename(columns={c: f"{c}_away" for c in value_cols})

        # Keep date/home/away identifiers only once (prefer home side if present)
        away = away.drop(columns=[date_col, home_team_col, away_team_col], errors="ignore")

        # CRITICAL FIX: OUTER join so missing home/away side doesn't drop the game
        wide = home.merge(away, on=game_id_col, how="outer")

        return wide

    # -------------------------
    # Expected rate columns
    # -------------------------
    starter_rate_cols = [f"roll_{w}_{starter_tag}_{m}" for w in windows for m in metrics]
    bullpen_rate_cols = [f"roll_{w}_{bullpen_tag}_{m}" for w in windows for m in metrics]

    # -------------------------
    # Validate structural columns
    # -------------------------
    _require_cols(
        starter_df,
        [game_id_col, date_col, home_team_col, away_team_col, is_home_col, pitcher_name_col],
        "starter_df",
    )
    _require_cols(
        bullpen_df,
        [game_id_col, date_col, home_team_col, away_team_col, is_home_col],
        "bullpen_df",
    )

    # Validate metric columns
    missing_st = [c for c in starter_rate_cols if c not in starter_df.columns]
    missing_bp = [c for c in bullpen_rate_cols if c not in bullpen_df.columns]
    if missing_st:
        raise ValueError(f"starter_df missing rolling RATE columns: {missing_st}")
    if missing_bp:
        raise ValueError(f"bullpen_df missing rolling RATE columns: {missing_bp}")

    # -------------------------
    # Widen to game-level
    # -------------------------
    starter_wide = _make_wide(
        starter_df,
        value_cols=starter_rate_cols,
        role_label="starter_df",
        keep_pitcher_name=True,
        rename_value_cols=False,
    )

    bullpen_wide = _make_wide(
        bullpen_df,
        value_cols=bullpen_rate_cols,
        role_label="bullpen_df",
        keep_pitcher_name=False,
        rename_value_cols=False,
    )

    # -------------------------
    # Merge bullpen onto starters (starter spine)
    # -------------------------
    bullpen_wide = bullpen_wide.drop(columns=[date_col, home_team_col, away_team_col], errors="ignore")

    # CRITICAL FIX: LEFT join so bullpen missing games don't drop starter games
    out = starter_wide.merge(bullpen_wide, on=game_id_col, how="left")

    # -------------------------
    # Sort + guardrails
    # -------------------------
    if date_col in out.columns:
        out = out.sort_values([date_col, game_id_col]).reset_index(drop=True)
    else:
        # If date_col ended up missing, something is wrong with the input structure
        raise ValueError(
            f"'{date_col}' missing after merge. This usually means starter_wide lost it unexpectedly."
        )

    # Uniqueness by game_id
    if out.duplicated([game_id_col]).any():
        dupes = out[out.duplicated([game_id_col], keep=False)][[game_id_col]].head(10)
        raise ValueError(f"Output is not unique by {game_id_col}. Example dupes:\n{dupes}")

    return out


def make_pitching_delta_df(
    game_pitching_rates: pd.DataFrame,
    windows: tuple[str, ...] = ("3D", "7D"),
    game_id_col: str = "game_id",
    date_col: str = "game_date",
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
    home_name_col: str = "starter_pitcher_name_home",
    away_name_col: str = "starter_pitcher_name_away",
) -> pd.DataFrame:
    """
    Create a NEW dataframe with only game identifiers, teams, starter names, and pitching deltas (home - away).

    Produces columns for each window:
      Δstarter_FIP_{w}, Δstarter_WHIP_{w}, Δstarter_K9_{w}, Δstarter_HR9_{w}
      Δbullpen_FIP_{w}

    Does NOT modify the input dataframe.
    """
    df = game_pitching_rates.copy()

    # Required base columns
    required_base = [
        game_id_col, date_col, home_team_col, away_team_col, home_name_col, away_name_col
    ]
    missing = [c for c in required_base if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required base columns: {missing}")

    out = df[required_base].copy()

    def _get_col(w: str, role: str, metric: str, side: str) -> str:
        """
        Prefer clean naming:
          roll_{w}_{role}_{metric}_{side}

        Fallback if combine step double-tagged:
          roll_{w}_{role}_{metric}_{role}_{side}
        """
        clean = f"roll_{w}_{role}_{metric}_{side}"
        if clean in df.columns:
            return clean

        double = f"roll_{w}_{role}_{metric}_{role}_{side}"
        if double in df.columns:
            return double

        raise ValueError(
            f"Missing expected column(s) for {role} {metric} {w} ({side}). "
            f"Tried: '{clean}' and '{double}'."
        )

    for w in windows:
        # --- starter deltas ---
        for metric in ("FIP", "WHIP", "K9", "HR9"):
            home_col = _get_col(w=w, role="starter", metric=metric, side="home")
            away_col = _get_col(w=w, role="starter", metric=metric, side="away")
            out[f"Δstarter_{metric}_{w}"] = df[home_col] - df[away_col]

        # --- bullpen deltas (FIP only) ---
        home_col = _get_col(w=w, role="bullpen", metric="FIP", side="home")
        away_col = _get_col(w=w, role="bullpen", metric="FIP", side="away")
        out[f"Δbullpen_FIP_{w}"] = df[home_col] - df[away_col]

    # Nice ordering
    delta_cols = [c for c in out.columns if c.startswith("Δ")]
    out = out[required_base + delta_cols].copy()

    return out


'''
Move to correct place later
'''
def summarize_pitching_rates(
    df: pd.DataFrame,
    kind: str = "starter",  # "starter" or "bullpen"
    pitcher_col: str = "pitcher_name",
    team_col: str = "pitching_team",
    ip_col: str = "IP",
    h_col: str = "H",
    bb_col: str = "BB",
    hbp_col: str = "HBP",
    k_col: str = "K",
    hr_col: str = "HR",
    include_hbp_in_whip: bool = True,
    fip_constant: float | None = None,  # None = raw FIP
    overall_label: str | None = None,   # optional custom label for the top row
) -> pd.DataFrame:
    """
    Aggregate pitching counts and compute WHIP, K/9, HR/9, FIP from totals.

    Output: ONE dataframe with the overall (innings-weighted, from totals) row at the top,
    followed by group rows:
      - kind="starter": group by pitcher_col
      - kind="bullpen": group by team_col

    Notes:
    - Rates are computed from aggregated totals (recommended), NOT averaged per-game or per-pitcher rates.
    - Groups with IP == 0 get NaN for rate stats.
    - No unweighted mean row is included.
    """
    kind = kind.lower().strip()
    if kind not in {"starter", "bullpen"}:
        raise ValueError('kind must be "starter" or "bullpen"')

    group_col = pitcher_col if kind == "starter" else team_col
    if group_col not in df.columns:
        raise ValueError(f"Expected grouping column '{group_col}' not found in df.")

    required = [ip_col, h_col, bb_col, hbp_col, k_col, hr_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = df.copy()
    for c in required:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0)

    # --- group totals ---
    by = (
        x.groupby(group_col, as_index=False)
         .agg(
             IP=(ip_col, "sum"),
             H=(h_col, "sum"),
             BB=(bb_col, "sum"),
             HBP=(hbp_col, "sum"),
             K=(k_col, "sum"),
             HR=(hr_col, "sum"),
         )
    )

    # --- compute rates from totals ---
    ip = by["IP"].to_numpy(dtype=float)
    h  = by["H"].to_numpy(dtype=float)
    bb = by["BB"].to_numpy(dtype=float)
    hbp= by["HBP"].to_numpy(dtype=float)
    k  = by["K"].to_numpy(dtype=float)
    hr = by["HR"].to_numpy(dtype=float)

    whip_num = h + bb + (hbp if include_hbp_in_whip else 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        by["WHIP"] = np.where(ip > 0, whip_num / ip, np.nan)
        by["K9"]   = np.where(ip > 0, (k  * 9.0) / ip, np.nan)
        by["HR9"]  = np.where(ip > 0, (hr * 9.0) / ip, np.nan)

        fip_num = 13.0 * hr + 3.0 * (bb + hbp) - 2.0 * k
        fip = np.where(ip > 0, fip_num / ip, np.nan)
        if fip_constant is not None:
            fip = fip + float(fip_constant)
        by["FIP"] = fip

    # --- overall (innings-weighted) totals-based row ---
    totals = by[["IP", "H", "BB", "HBP", "K", "HR"]].sum(numeric_only=True)
    IPt, Ht, BBt, HBPt, Kt, HRt = [float(totals[c]) for c in ["IP", "H", "BB", "HBP", "K", "HR"]]

    if overall_label is None:
        overall_label = f"ALL_{kind.upper()}"

    whip_num_t = Ht + BBt + (HBPt if include_hbp_in_whip else 0.0)

    overall = {
        group_col: overall_label,
        "IP": IPt,
        "H": Ht,
        "BB": BBt,
        "HBP": HBPt,
        "K": Kt,
        "HR": HRt,
        "WHIP": (whip_num_t / IPt) if IPt > 0 else np.nan,
        "K9": (Kt * 9.0 / IPt) if IPt > 0 else np.nan,
        "HR9": (HRt * 9.0 / IPt) if IPt > 0 else np.nan,
        "FIP": ((13.0 * HRt + 3.0 * (BBt + HBPt) - 2.0 * Kt) / IPt + (float(fip_constant) if fip_constant is not None else 0.0))
               if IPt > 0 else np.nan,
        "mean_type": "weighted_by_IP (from totals)",
    }

    # sort groups (optional): most IP first
    by = by.sort_values(["IP", group_col], ascending=[False, True]).reset_index(drop=True)

    # final: overall row on top
    out = pd.concat([pd.DataFrame([overall]), by.assign(mean_type="group_totals")], ignore_index=True)

    # column order
    out = out[[group_col, "IP", "H", "BB", "HBP", "K", "HR", "WHIP", "K9", "HR9", "FIP", "mean_type"]]

    return out


'''
Move to correct place later
'''
def impute_pitching_roll_rates_from_prev_season(
    season_df: pd.DataFrame,
    prev_summary_df: pd.DataFrame,
    kind: str = "starter",  # "starter" or "bullpen"
    windows: tuple[str, ...] = ("3D", "7D"),
    metrics: tuple[str, ...] = ("WHIP", "K9", "HR9", "FIP"),
    pitcher_col: str = "pitcher_name",
    team_col: str = "pitching_team",
    roll_prefix: str = "roll_",
    starter_tag: str = "starter",
    bullpen_tag: str = "bullpen",
) -> pd.DataFrame:
    """
    Impute missing rolling RATE features in season t using season t-1 summary values.

    Starter case:
      - Expected columns: roll_{w}_starter_{metric}
      - Key: pitcher_name
      - Fallback: if pitcher not in prev summary OR prev value is NaN -> league mean row (ALL_STARTER)

    Bullpen case:
      - Expected columns: roll_{w}_bullpen_{metric}
      - Key: pitching_team
      - Fallback: if team not in prev summary -> league mean row (ALL_BULLPEN)

    Returns a copy of season_df with NaNs filled. (No extra indicator columns.)
    """
    kind = kind.lower().strip()
    if kind not in {"starter", "bullpen"}:
        raise ValueError('kind must be "starter" or "bullpen"')

    out = season_df.copy()

    if kind == "starter":
        group_col = pitcher_col
        overall_label = "ALL_STARTER"
        col_template = f"{roll_prefix}{{w}}_{starter_tag}_{{m}}"
    else:
        group_col = team_col
        overall_label = "ALL_BULLPEN"
        col_template = f"{roll_prefix}{{w}}_{bullpen_tag}_{{m}}"

    if group_col not in out.columns:
        raise ValueError(f"season_df missing grouping column '{group_col}'")

    if group_col not in prev_summary_df.columns:
        raise ValueError(f"prev_summary_df missing grouping column '{group_col}'")

    for m in metrics:
        if m not in prev_summary_df.columns:
            raise ValueError(f"prev_summary_df missing metric column '{m}'")

    prev = prev_summary_df[[group_col, *metrics]].copy()
    prev_map = prev.set_index(group_col)

    if overall_label not in prev_map.index:
        raise ValueError(
            f"prev_summary_df must contain overall row '{overall_label}' in column '{group_col}'."
        )

    league_vals = prev_map.loc[overall_label, list(metrics)].to_dict()

    def _prev_or_league(entity: str, metric: str) -> float:
        if entity in prev_map.index:
            v = prev_map.at[entity, metric]
            if pd.notna(v):
                return float(v)

        lv = league_vals.get(metric, np.nan)
        return float(lv) if pd.notna(lv) else np.nan

    roll_cols = []
    for w in windows:
        for m in metrics:
            c = col_template.format(w=w, m=m)
            if c in out.columns:
                roll_cols.append((w, m, c))

    if not roll_cols:
        example = col_template.format(w="3D", m=metrics[0])
        raise ValueError(
            f"No matching roll columns found in season_df. Example expected: '{example}'."
        )

    entities = out[group_col].astype("string").fillna("")

    for _, m, c in roll_cols:
        miss = out[c].isna()
        if miss.any():
            out.loc[miss, c] = entities.loc[miss].map(lambda e: _prev_or_league(str(e), m)).to_numpy()

    return out

'''
Move to correct place later
'''
def combine_game_level_pitching_rolling_rates(
    starter_df: pd.DataFrame,
    bullpen_df: pd.DataFrame,
    windows: tuple[str, ...] = ("3D", "7D"),
    metrics: tuple[str, ...] = ("WHIP", "K9", "HR9", "FIP"),
    game_id_col: str = "game_id",
    date_col: str = "game_date",
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
    is_home_col: str = "is_home_team",
    pitcher_name_col: str = "pitcher_name",
    starter_tag: str = "starter",
    bullpen_tag: str = "bullpen",
    did_not_end_pa_col: str = "did_not_end_pa",
    starter_full_game_col: str = "starter_full_game",
) -> pd.DataFrame:
    """
    Combine starter + bullpen rolling RATE metrics into ONE row per game_id.

    Key behavior:
    - Widen home vs away with OUTER (never drops a game just because one side is missing)
    - Merge bullpen onto starter with LEFT (starter_wide is the backbone)
    - Keeps home_team + away_team + game_date from starter_wide (preferred source)

    Keeps (if present):
    - starter_full_game as side-specific columns:
        starter_full_game_home, starter_full_game_away
    - did_not_end_pa as side-specific columns:
        did_not_end_pa_home, did_not_end_pa_away

    NOTE:
    - This version does NOT blank any rolling rate columns based on did_not_end_pa.
    """

    def _require_cols(df: pd.DataFrame, cols: list[str], label: str) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{label} is missing columns: {missing}")

    def _make_wide(
        df: pd.DataFrame,
        value_cols: list[str],
        role_tag: str,
        keep_pitcher_name: bool,
        keep_full_game: bool,
    ) -> pd.DataFrame:
        base_cols = [game_id_col, date_col, home_team_col, away_team_col, is_home_col]
        extra_cols = [pitcher_name_col] if keep_pitcher_name else []
        maybe_flag = [did_not_end_pa_col] if did_not_end_pa_col in df.columns else []
        maybe_full = [starter_full_game_col] if (keep_full_game and starter_full_game_col in df.columns) else []
        use_cols = base_cols + extra_cols + maybe_full + maybe_flag + value_cols
        _require_cols(df, use_cols, f"{role_tag}_df")

        tmp = df[use_cols].copy()
        tmp[is_home_col] = tmp[is_home_col].astype(int)

        # Guardrail: should be <=1 row per (game_id, side) at this stage
        dups = tmp.duplicated([game_id_col, is_home_col]).sum()
        if dups:
            raise ValueError(f"{role_tag}_df has {dups} duplicates of (game_id, is_home_team).")

        home = tmp[tmp[is_home_col] == 1].drop(columns=[is_home_col], errors="ignore")
        away = tmp[tmp[is_home_col] == 0].drop(columns=[is_home_col], errors="ignore")

        # Rename starter pitcher names
        if keep_pitcher_name:
            home = home.rename(columns={pitcher_name_col: "starter_pitcher_name_home"})
            away = away.rename(columns={pitcher_name_col: "starter_pitcher_name_away"})

        # Keep starter_full_game as side-specific columns (starter df only)
        if keep_full_game and starter_full_game_col in home.columns:
            home = home.rename(columns={starter_full_game_col: "starter_full_game_home"})
        if keep_full_game and starter_full_game_col in away.columns:
            away = away.rename(columns={starter_full_game_col: "starter_full_game_away"})

        # Keep did_not_end_pa as side-specific columns (prevents _x/_y suffix mess)
        if did_not_end_pa_col in home.columns:
            home = home.rename(columns={did_not_end_pa_col: "did_not_end_pa_home"})
        if did_not_end_pa_col in away.columns:
            away = away.rename(columns={did_not_end_pa_col: "did_not_end_pa_away"})

        # Rename metric columns to include side
        home = home.rename(columns={c: f"{c}_home" for c in value_cols})
        away = away.rename(columns={c: f"{c}_away" for c in value_cols})

        # We want exactly one copy of date/home_team/away_team in the merged output.
        away = away.drop(columns=[date_col, home_team_col, away_team_col], errors="ignore")

        wide = home.merge(away, on=game_id_col, how="outer", validate="one_to_one")
        return wide

    starter_rate_cols = [f"roll_{w}_{starter_tag}_{m}" for w in windows for m in metrics]
    bullpen_rate_cols = [f"roll_{w}_{bullpen_tag}_{m}" for w in windows for m in metrics]

    # structural cols
    starter_required = [game_id_col, date_col, home_team_col, away_team_col, is_home_col, pitcher_name_col]
    bullpen_required = [game_id_col, date_col, home_team_col, away_team_col, is_home_col]

    # starter_full_game is optional but if present we will carry it through
    if starter_full_game_col in starter_df.columns:
        starter_required = starter_required + [starter_full_game_col]

    _require_cols(starter_df, starter_required, "starter_df")
    _require_cols(bullpen_df, bullpen_required, "bullpen_df")

    # rate cols exist
    missing_st = [c for c in starter_rate_cols if c not in starter_df.columns]
    missing_bp = [c for c in bullpen_rate_cols if c not in bullpen_df.columns]
    if missing_st:
        raise ValueError(f"starter_df missing rate cols: {missing_st}")
    if missing_bp:
        raise ValueError(f"bullpen_df missing rate cols: {missing_bp}")

    # widen
    keep_full_game = starter_full_game_col in starter_df.columns
    starter_wide = _make_wide(
        starter_df, starter_rate_cols, role_tag=starter_tag, keep_pitcher_name=True, keep_full_game=keep_full_game
    )
    bullpen_wide = _make_wide(
        bullpen_df, bullpen_rate_cols, role_tag=bullpen_tag, keep_pitcher_name=False, keep_full_game=False
    )

    # merge bullpen onto starters
    out = starter_wide.merge(
        bullpen_wide.drop(columns=[date_col, home_team_col, away_team_col], errors="ignore"),
        on=game_id_col,
        how="left",
        validate="one_to_one",
    )

    out = out.sort_values([date_col, game_id_col], na_position="last").reset_index(drop=True)

    if out.duplicated([game_id_col]).any():
        raise ValueError("Output is not unique by game_id (unexpected).")

    return out


def carry_forward_bullpen_rolls_on_full_games(
    df: pd.DataFrame,
    windows: tuple[str, ...] = ("3D", "7D"),
    metrics: tuple[str, ...] = ("WHIP", "K9", "HR9", "FIP"),
    date_col: str = "game_date",
    game_id_col: str = "game_id",
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
    starter_full_game_home_col: str = "starter_full_game_home",
    starter_full_game_away_col: str = "starter_full_game_away",
    bullpen_tag: str = "bullpen",
) -> pd.DataFrame:
    """
    Post-merge patch: carry forward bullpen rolling RATE metrics for teams in games
    where the starter pitched the full game (bullpen unused), which yields NA bullpen metrics.

    Strategy:
      - Only fills bullpen metrics when they are NA AND the corresponding side's
        starter_full_game_* == 1.
      - Fill is done within-team over time:
          home side filled within home_team sequence
          away side filled within away_team sequence

    Returns a copy of df with bullpen rolling metrics carry-forward filled.
    """
    out = df.copy()

    # Ensure proper ordering for forward-fill
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.sort_values([date_col, game_id_col]).reset_index(drop=True)

    # Build bullpen rate columns explicitly (avoid accidentally grabbing non-rate bullpen cols)
    bullpen_rate_cols = []
    for w in windows:
        for m in metrics:
            bullpen_rate_cols.append(f"roll_{w}_{bullpen_tag}_{m}_home")
            bullpen_rate_cols.append(f"roll_{w}_{bullpen_tag}_{m}_away")

    # Keep only cols that actually exist
    bullpen_rate_cols = [c for c in bullpen_rate_cols if c in out.columns]
    if not bullpen_rate_cols:
        return out

    # Side-specific lists
    home_cols = [c for c in bullpen_rate_cols if c.endswith("_home")]
    away_cols = [c for c in bullpen_rate_cols if c.endswith("_away")]

    # Masks: which rows should be filled (NA + full game on that side)
    home_full_mask = (
        out.get(starter_full_game_home_col, 0)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0).astype(int).eq(1)
    )
    away_full_mask = (
        out.get(starter_full_game_away_col, 0)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0).astype(int).eq(1)
    )

    # Fill home bullpen metrics within each home team timeline
    if home_cols:
        home_ffill = (
            out.groupby(home_team_col, sort=False)[home_cols]
               .ffill()
        )
        # only fill where NA and full-game
        for c in home_cols:
            need = home_full_mask & out[c].isna()
            out.loc[need, c] = home_ffill.loc[need, c]

    # Fill away bullpen metrics within each away team timeline
    if away_cols:
        away_ffill = (
            out.groupby(away_team_col, sort=False)[away_cols]
               .ffill()
        )
        for c in away_cols:
            need = away_full_mask & out[c].isna()
            out.loc[need, c] = away_ffill.loc[need, c]

    return out