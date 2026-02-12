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

    # Completed PA flag
    out["is_pa_countable"] = (~ev.eq("truncated_pa")) & ev.notna()

    # Core FIP components
    out["is_hr"]  = ev.eq("home_run").astype(int)
    out["is_bb"]  = ev.isin(["walk", "intent_walk"]).astype(int)
    out["is_hbp"] = ev.eq("hit_by_pitch").astype(int)
    out["is_k"]   = ev.isin(["strikeout", "strikeout_double_play"]).astype(int)

    # Hits allowed (for WHIP; HR is also a hit)
    out["is_h"] = ev.isin(["single", "double", "triple", "home_run"]).astype(int)

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
    game_id_col: str = "game_id",
    pitching_team_col: str = "pitching_team",
    pitcher_col: str = "pitcher_name",
    validate: bool = True,
):
    """
    Returns (starter_df, bullpen_df).

    If countable_col exists and is not None, filters to countable PAs first.

    Optional validation (recommended):
      1) Every (game_id, pitching_team) has exactly 1 starter pitcher identity
      2) Starter + bullpen rows == filtered rows
    """
    out = df.copy()

    if countable_col is not None and countable_col in out.columns:
        out = out[out[countable_col].astype(bool)].copy()

    starter_df = out[out[is_starter_col].astype(int) == 1].copy()
    bullpen_df = out[out[is_starter_col].astype(int) == 0].copy()

    if validate:
        # (A) partition check
        if len(starter_df) + len(bullpen_df) != len(out):
            raise ValueError(
                f"Split does not partition rows cleanly: "
                f"starter({len(starter_df)}) + bullpen({len(bullpen_df)}) != out({len(out)})"
            )

        # (B) identity uniqueness: exactly one starter pitcher per (game_id, pitching_team)
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
      game_id, game_date, pitching_team, is_home_team, pitcher_role,
      [pitcher_id], [pitcher_name], IP, H, BB, HBP, K, HR

    Notes:
    - Filters to is_pa_countable == True
    - pitcher_role = "starter" if is_starter == 1 else "bullpen"
    - IP computed from outs: IP = sum(outs)/3
    """
    df = pa_pitching.copy()

    required = [
        "game_id", "game_date", "pitching_team", "home_team",
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

    # Keep only countable PA rows
    df = df[df["is_pa_countable"] == True].copy()

    # Normalize role
    df["is_starter"] = pd.to_numeric(df["is_starter"], errors="coerce").fillna(0).astype(int)
    df["pitcher_role"] = np.where(df["is_starter"] == 1, "starter", "bullpen")

    # Home-team indicator (1/0)
    df["is_home_team"] = (df["pitching_team"] == df["home_team"]).astype(int)

    # Coerce numeric flags
    num_cols = ["outs", "is_h", "is_bb", "is_hbp", "is_k", "is_hr"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    group_cols = ["game_id", "pitching_team", "is_home_team", "pitcher_role"]
    if pitcher_id_col is not None:
        group_cols.append(pitcher_id_col)
        if pitcher_name_col is not None:
            group_cols.append(pitcher_name_col)

    game_lines = (
        df.groupby(group_cols, as_index=False)
          .agg(
              game_date=("game_date", "min"),
              outs=("outs", "sum"),
              H=("is_h", "sum"),
              BB=("is_bb", "sum"),
              HBP=("is_hbp", "sum"),
              K=("is_k", "sum"),
              HR=("is_hr", "sum"),
          )
    )

    game_lines["IP"] = game_lines["outs"] / 3.0
    game_lines = game_lines.drop(columns=["outs"])

    # Column order (keeps is_home_team between pitching_team and pitcher_role)
    cols = ["game_id", "game_date", "pitching_team", "is_home_team", "pitcher_role"]
    if pitcher_id_col is not None:
        cols.append(pitcher_id_col)
        if pitcher_name_col is not None:
            cols.append(pitcher_name_col)
    cols += ["IP", "H", "BB", "HBP", "K", "HR"]

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
    windows: tuple[str, ...] = ("3D", "7D"),
    pitcher_col: str = "pitcher_name",
    team_col: str = "pitching_team",
    date_col: str = "game_date",
    game_id_col: str = "game_id",
    role_col: str = "pitcher_role",
    include_current_game: bool = False,  # False = only prior games in the window
    starter_tag: str = "starter",
    bullpen_tag: str = "bullpen",
) -> pd.DataFrame:
    """
    Add rolling (time-based) pitching COUNT stats for 3D and 7D windows.

    Behavior:
      - Starter mode: rolls by pitcher_name and outputs columns like roll_3D_starter_IP, ...
        (Does NOT create starter_* duplicate base columns.)

      - Bullpen/team mode: rolls by pitching_team and outputs columns like roll_3D_bullpen_IP, ...

    Rolling stats are SUMS over the window (not ratios).
    include_current_game=False shifts by 1 to avoid leakage.
    """
    out = df.copy()

    # --- basic checks ---
    for c in [team_col, date_col, game_id_col]:
        if c not in out.columns:
            raise ValueError(f"Missing required column: '{c}'")

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if out[date_col].isna().any():
        bad = out.loc[out[date_col].isna(), [date_col, game_id_col]].head(5)
        raise ValueError(f"Found non-parseable {date_col} values. Examples:\n{bad}")

    # Detect whether this is starter-style data
    has_pitcher = pitcher_col in out.columns
    has_role_starter = (role_col in out.columns) and (out[role_col].astype("string").eq("starter").any())
    is_starter_mode = has_pitcher and has_role_starter

    # Stats we want to roll (raw counts only)
    base_stats = ["IP", "H", "BB", "HBP", "K", "HR"]
    missing_base = [c for c in base_stats if c not in out.columns]
    if missing_base:
        raise ValueError(f"Missing required stat columns for rolling: {missing_base}")

    # Coerce base stats numeric
    for c in base_stats:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if is_starter_mode:
        group_keys = [pitcher_col]
        sort_cols = [pitcher_col, date_col, game_id_col]
        tag = starter_tag
    else:
        group_keys = [team_col]
        sort_cols = [team_col, date_col, game_id_col]
        tag = bullpen_tag

    # Sort before rolling
    out = out.sort_values(sort_cols).reset_index(drop=True)

    def _add_rolls(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values([date_col, game_id_col]).copy()
        g_indexed = g.set_index(date_col)

        for w in windows:
            rolled = g_indexed[base_stats].rolling(window=w, min_periods=1).sum()

            if not include_current_game:
                rolled = rolled.shift(1)

            # Name columns exactly how you want:
            # starters: roll_3D_starter_IP
            # bullpen:  roll_3D_bullpen_IP
            rolled.columns = [f"roll_{w}_{tag}_{c}" for c in rolled.columns]
            rolled = rolled.reset_index(drop=True)

            g = pd.concat([g.reset_index(drop=True), rolled], axis=1)

        return g

    out = out.groupby(group_keys, group_keys=False).apply(_add_rolls).reset_index(drop=True)
    return out


def add_rate_metrics_from_rolled_counts(
    df: pd.DataFrame,
    windows: tuple[str, ...] = ("3D", "7D"),
    include_hbp_in_whip: bool = True,
    fip_constant: float | None = None,
    bullpen_tag: str = "bullpen",
) -> pd.DataFrame:
    """
    Add WHIP, K9, HR9, and (raw) FIP using *rolled count columns*.

    Now supports:
      - starter: roll_{w}_starter_IP/H/BB/HBP/K/HR -> roll_{w}_starter_WHIP/K9/HR9/FIP
      - bullpen: roll_{w}_bullpen_IP/H/BB/HBP/K/HR -> roll_{w}_bullpen_WHIP/K9/HR9/FIP
    """
    import numpy as np
    import pandas as pd

    out = df.copy()

    def _compute_for_prefix(prefix: str, label: str) -> None:
        ip = f"{prefix}IP"
        h  = f"{prefix}H"
        bb = f"{prefix}BB"
        hbp= f"{prefix}HBP"
        k  = f"{prefix}K"
        hr = f"{prefix}HR"

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

    for w in windows:
        # starter
        _compute_for_prefix(prefix=f"roll_{w}_starter_", label=f"roll_{w}_starter")
        # bullpen (tagged)
        _compute_for_prefix(prefix=f"roll_{w}_{bullpen_tag}_", label=f"roll_{w}_{bullpen_tag}")

    return out


def combine_game_level_pitching_rolling_rates(
    starter_df: pd.DataFrame,
    bullpen_df: pd.DataFrame,
    windows: tuple[str, ...] = ("3D", "7D"),
    metrics: tuple[str, ...] = ("WHIP", "K9", "HR9", "FIP"),
    game_id_col: str = "game_id",
    date_col: str = "game_date",
    is_home_col: str = "is_home_team",
    pitcher_name_col: str = "pitcher_name",
    starter_tag: str = "starter",
    bullpen_tag: str = "bullpen",
) -> pd.DataFrame:
    """
    Combine starter + bullpen rolling RATE metrics into ONE row per game_id.

    Expects:
      starter_df contains: roll_{w}_{starter_tag}_{metric} + pitcher_name + is_home_team
      bullpen_df contains: roll_{w}_{bullpen_tag}_{metric} + is_home_team

    Output columns:
      starter_pitcher_name_home, starter_pitcher_name_away,
      roll_3D_starter_WHIP_home, ... roll_7D_starter_FIP_away,
      roll_3D_bullpen_WHIP_home, ... roll_7D_bullpen_FIP_away
    """

    def _require_cols(df: pd.DataFrame, cols: list[str], label: str) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{label} is missing columns: {missing}")

    def _make_wide(
        df: pd.DataFrame,
        value_cols: list[str],
        prefix: str,              # "starter" or "bullpen" (used in the final column names)
        keep_pitcher_name: bool,  # True for starters only
    ) -> pd.DataFrame:
        base_cols = [game_id_col, date_col, is_home_col]
        extra_cols = [pitcher_name_col] if keep_pitcher_name else []
        use_cols = base_cols + extra_cols + value_cols
        _require_cols(df, use_cols, f"{prefix} df")

        tmp = df[use_cols].copy()
        tmp[is_home_col] = tmp[is_home_col].astype(int)

        # Guardrail: one row per (game_id, side)
        dups = tmp.duplicated([game_id_col, is_home_col]).sum()
        if dups:
            raise ValueError(f"{prefix} df has {dups} duplicates of (game_id, is_home_team).")

        home = tmp[tmp[is_home_col] == 1].drop(columns=[is_home_col])
        away = tmp[tmp[is_home_col] == 0].drop(columns=[is_home_col])

        if keep_pitcher_name:
            home = home.rename(columns={pitcher_name_col: "starter_pitcher_name_home"})
            away = away.rename(columns={pitcher_name_col: "starter_pitcher_name_away"})

        # Final desired naming is: <value_col>_<prefix>_<side>
        # Example: roll_7D_bullpen_WHIP_home (prefix="bullpen")
        home = home.rename(columns={c: f"{c}_{prefix}_home" for c in value_cols})
        away = away.rename(columns={c: f"{c}_{prefix}_away" for c in value_cols})

        # Keep date once (prefer home)
        if date_col in away.columns:
            away = away.drop(columns=[date_col])

        return home.merge(away, on=game_id_col, how="inner")

    # ---- expected rate columns ----
    starter_rate_cols = [f"roll_{w}_{starter_tag}_{m}" for w in windows for m in metrics]
    bullpen_rate_cols = [f"roll_{w}_{bullpen_tag}_{m}" for w in windows for m in metrics]

    # ---- validate structural columns ----
    _require_cols(starter_df, [game_id_col, date_col, is_home_col, pitcher_name_col], "starter_df")
    _require_cols(bullpen_df, [game_id_col, date_col, is_home_col], "bullpen_df")

    # ---- validate rate columns exist ----
    missing_st = [c for c in starter_rate_cols if c not in starter_df.columns]
    missing_bp = [c for c in bullpen_rate_cols if c not in bullpen_df.columns]
    if missing_st:
        raise ValueError(
            "starter_df is missing some starter rolling RATE metric columns.\n"
            f"Missing: {missing_st}\n"
            "Did you run add_rate_metrics_from_rolled_counts on the starter rolling df?"
        )
    if missing_bp:
        raise ValueError(
            "bullpen_df is missing some bullpen rolling RATE metric columns.\n"
            f"Missing: {missing_bp}\n"
            "Did you run add_rate_metrics_from_rolled_counts on the bullpen rolling df?"
        )

    # ---- wide starter + wide bullpen ----
    starter_wide = _make_wide(starter_df, starter_rate_cols, prefix=starter_tag, keep_pitcher_name=True)
    bullpen_wide = _make_wide(bullpen_df, bullpen_rate_cols, prefix=bullpen_tag, keep_pitcher_name=False)

    # ---- merge to one row per game ----
    out = starter_wide.merge(
        bullpen_wide.drop(columns=[date_col]) if date_col in bullpen_wide.columns else bullpen_wide,
        on=game_id_col,
        how="inner",
    )

    # ---- rename cleanup (remove accidental double tags if they ever occur) ----
    # Example bad: roll_7D_starter_K9_starter_home -> roll_7D_starter_K9_home
    rename_map = {}
    for w in windows:
        for m in metrics:
            for side in ("home", "away"):
                bad = f"roll_{w}_{starter_tag}_{m}_{starter_tag}_{side}"
                good = f"roll_{w}_{starter_tag}_{m}_{side}"
                if bad in out.columns:
                    rename_map[bad] = good
    if rename_map:
        out = out.rename(columns=rename_map)

    # ---- final sort + guardrail ----
    out = out.sort_values([date_col, game_id_col]).reset_index(drop=True)
    if out.duplicated([game_id_col]).any():
        raise ValueError("Output is not unique by game_id (unexpected).")

    return out


def make_pitching_delta_df(
    game_pitching_rates: pd.DataFrame,
    windows: tuple[str, ...] = ("3D", "7D"),
    game_id_col: str = "game_id",
    date_col: str = "game_date",
    home_name_col: str = "starter_pitcher_name_home",
    away_name_col: str = "starter_pitcher_name_away",
) -> pd.DataFrame:
    """
    Create a NEW dataframe with only game identifiers, starter names, and pitching deltas (home - away).

    Produces columns for each window:
      Δstarter_FIP_{w}, Δstarter_WHIP_{w}, Δstarter_K9_{w}, Δstarter_HR9_{w}
      Δbullpen_FIP_{w}

    Does NOT modify the input dataframe.
    """
    df = game_pitching_rates.copy()

    # Required base columns
    required_base = [game_id_col, date_col, home_name_col, away_name_col]
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
