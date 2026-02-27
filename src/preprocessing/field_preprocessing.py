import numpy as np
import pandas as pd


def add_fielding_indicators(
    pa_df: pd.DataFrame,
    events_col: str = "events",
    topbot_col: str = "inning_topbot",
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
) -> pd.DataFrame:
    """
    Add PA-level indicators needed for a BIP-based fielding metric and create fielding_team.

    - Drops rows where `events_col` is missing (None/NaN/<NA>).
    - Creates `fielding_team` as the *defense* (opposite of batting_team based on inning_topbot).
    - Adds flags for K/BB/HBP/SF/HR/H (in-play hit proxy) and BIP eligibility.
    - Places `fielding_team` after `batting_team` if present, otherwise after away_team.
    """
    out = pa_df.copy()

    # Drop rows with missing events
    out = out[out[events_col].notna()].copy()

    ev = out[events_col].astype("string")

    # Basic PA outcome indicators (needed to define BIP universe)
    out["is_bb"]  = ev.isin(["walk", "intent_walk"]).astype(int)
    out["is_hbp"] = ev.eq("hit_by_pitch").astype(int)
    out["is_sf"]  = ev.isin(["sac_fly", "sac_fly_double_play"]).astype(int)
    out["is_k"]   = ev.isin(["strikeout", "strikeout_double_play", "strikeout_triple_play"]).astype(int)

    # Hits / HR
    out["is_hr"] = ev.eq("home_run").astype(int)
    out["is_h"]  = ev.isin(["single", "double", "triple", "home_run"]).astype(int)

    # Hits on balls in play (exclude HR)
    out["is_bip_hit"] = ((out["is_h"] == 1) & (out["is_hr"] == 0)).astype(int)

    # BIP indicator: ball in play (exclude BB/HBP/K/HR/SF)
    out["is_bip"] = (
        (out["is_bb"] == 0)
        & (out["is_hbp"] == 0)
        & (out["is_k"] == 0)
        & (out["is_hr"] == 0)
        & (out["is_sf"] == 0)
    ).astype(int)

    # Create fielding_team (defense): Top = home fields, Bot = away fields
    topbot = out[topbot_col].astype("string").str.lower()
    out["fielding_team"] = out[home_team_col]
    out.loc[topbot == "bot", "fielding_team"] = out[away_team_col]

    # Place fielding_team after batting_team if it exists, else after away_team
    cols = list(out.columns)
    cols.remove("fielding_team")
    if "batting_team" in cols:
        insert_at = cols.index("batting_team") + 1
    else:
        insert_at = cols.index(away_team_col) + 1
    cols.insert(insert_at, "fielding_team")
    out = out[cols]

    return out


def make_game_fielding_bip_counts(
    pa_field_df: pd.DataFrame,
    game_id_col: str = "game_id",
    date_col: str = "game_date",
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
    fielding_team_col: str = "fielding_team",
    bip_col: str = "is_bip",
    bip_hit_col: str = "is_bip_hit",
) -> pd.DataFrame:
    """
    Aggregate PA-level fielding indicators to game-level defense for each team.

    Output: one row per (game_id, fielding_team) with:
      - BIP: balls in play faced by that defense
      - BIP_H: hits on balls in play (non-HR hits)

    Note: We intentionally do NOT compute BIP_out_rate here.
    We'll compute the rate later from rolled totals (recommended).
    """
    df = pa_field_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    g = (
        df.groupby([game_id_col, date_col, home_team_col, away_team_col, fielding_team_col], as_index=False)
          .agg(
              BIP=(bip_col, "sum"),
              BIP_H=(bip_hit_col, "sum"),
          )
    )

    return g


def add_rolling_bip_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (3, 7),
    team_col: str = "fielding_team",
    date_col: str = "game_date",
    game_id_col: str = "game_id",
    bip_col: str = "BIP",
    bip_hit_col: str = "BIP_H",
) -> pd.DataFrame:
    """
    Add rolling BIP features for the FIELDING team using PRIOR games only (shifted by 1).

    For each team:
      1) shift BIP and BIP_H by 1 so the current game is excluded
      2) rolling-sum those shifted counts over last N games
      3) compute rolling BIP outs rate from rolled totals:
           roll_BIP_out_rate = 1 - roll_BIP_H / roll_BIP

    Creates for each N in `windows`:
      - roll_{N}G_BIP
      - roll_{N}G_BIP_H
      - roll_{N}G_BIP_out_rate

    Also reorders columns so roll_3G_BIP_out_rate and roll_7G_BIP_out_rate
    appear next to each other (and similarly groups the rolled BIP and BIP_H columns).
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    out[bip_col] = pd.to_numeric(out[bip_col], errors="coerce").fillna(0)
    out[bip_hit_col] = pd.to_numeric(out[bip_hit_col], errors="coerce").fillna(0)

    out = out.sort_values([team_col, date_col, game_id_col]).reset_index(drop=True)
    g = out.groupby(team_col, sort=False)

    # Prior-games-only shift
    out["_BIP_prev"] = g[bip_col].shift(1)
    out["_BIP_H_prev"] = g[bip_hit_col].shift(1)

    # Compute rolling totals + derived rate
    for n in windows:
        rbip  = f"roll_{n}G_BIP"
        rbh   = f"roll_{n}G_BIP_H"
        rrate = f"roll_{n}G_BIP_out_rate"

        out[rbip] = g["_BIP_prev"].rolling(window=n, min_periods=1).sum().reset_index(level=0, drop=True)
        out[rbh]  = g["_BIP_H_prev"].rolling(window=n, min_periods=1).sum().reset_index(level=0, drop=True)

        with np.errstate(divide="ignore", invalid="ignore"):
            out[rrate] = np.where(out[rbip] > 0, 1.0 - (out[rbh] / out[rbip]), np.nan)

    out = out.drop(columns=["_BIP_prev", "_BIP_H_prev"])

    # ---- reorder: keep rolled columns grouped and put out-rates adjacent ----
    def _move_after(cols_to_move: list[str], after: str) -> None:
        nonlocal out
        cols_to_move = [c for c in cols_to_move if c in out.columns]
        if not cols_to_move or after not in out.columns:
            return
        cols = list(out.columns)
        for c in cols_to_move:
            cols.remove(c)
        insert_at = cols.index(after) + 1
        cols[insert_at:insert_at] = cols_to_move
        out = out[cols]

    # Preserve window order as provided
    bip_roll_cols   = [f"roll_{n}G_BIP" for n in windows]
    bh_roll_cols    = [f"roll_{n}G_BIP_H" for n in windows]
    rate_roll_cols  = [f"roll_{n}G_BIP_out_rate" for n in windows]

    # Place all rolled features right after BIP_H (or after BIP if BIP_H missing)
    anchor = bip_hit_col if bip_hit_col in out.columns else (bip_col if bip_col in out.columns else team_col)
    _move_after(bip_roll_cols + bh_roll_cols + rate_roll_cols, after=anchor)

    return out


def calculate_mean_bip_out_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-team aggregate totals and mean BIP_out_rate at the game level.

    Intended use:
        - Generate prior-season team-level BIP_out_rate summaries
        - Provide fallback values for imputing missing rolling BIP_out_rate
          features early in a season

    Method:
        - Compute BIP_out_rate at the game level:
              BIP_out_rate = 1 - (BIP_H / BIP)
        - If BIP == 0 for a game, that game's rate is set to NA
        - Return per-team:
              * aggregated totals (BIP, BIP_H)
              * mean of game-level BIP_out_rate (simple average across games)

    Notes:
        - Mean is computed as the average of per-game rates (not totals-based).
        - NA game-level rates are excluded from the mean.
    """    
    
    out = df.copy()

    out["BIP_out_rate"] = 1.0 - (out["BIP_H"] / out["BIP"].replace(0, pd.NA))

    summary = (
        out.groupby("fielding_team", as_index=False)
           .agg(
               games=("game_id", "nunique"),
               BIP=("BIP", "sum"),
               BIP_H=("BIP_H", "sum"),
               mean_BIP_out_rate=("BIP_out_rate", "mean"),
           )
    )

    return summary


def fill_missing_rolling_bip_out_rate_from_prior_year(
    team_game_df: pd.DataFrame,
    prior_year_means: pd.DataFrame,
    rolling_bip_out_rate_cols: tuple[str, ...] = ("roll_3G_BIP_out_rate", "roll_7G_BIP_out_rate"),
    team_col: str = "fielding_team",
) -> pd.DataFrame:
    """
    Fill missing rolling BIP_out_rate columns in season t using prior season's mean_BIP_out_rate
    for the same team (computed at the game level).
    """
    df = team_game_df.copy()

    # team -> mean_BIP_out_rate
    rate_map = prior_year_means.set_index(team_col)["mean_BIP_out_rate"].to_dict()

    for c in rolling_bip_out_rate_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[team_col].map(rate_map))

    return df


def make_game_level_fielding_out_rate_wide(
    fielding_df: pd.DataFrame,
    game_id_col: str = "game_id",
    date_col: str = "game_date",
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
    fielding_team_col: str = "fielding_team",
    keep_cols: tuple[str, ...] = ("roll_3G_BIP_out_rate", "roll_7G_BIP_out_rate"),
) -> pd.DataFrame:
    """
    Convert team-level fielding rows (one row per game_id x fielding_team) into one row per game_id with:
      roll_3G_BIP_out_rate_home, roll_3G_BIP_out_rate_away,
      roll_7G_BIP_out_rate_home, roll_7G_BIP_out_rate_away

    Output order is window-grouped: 3G (home, away) then 7G (home, away).
    """
    df = fielding_df.copy()

    required = [game_id_col, date_col, home_team_col, away_team_col, fielding_team_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"fielding_df missing required columns: {missing}")

    keep_cols = tuple(c for c in keep_cols if c in df.columns)
    if not keep_cols:
        raise ValueError("No rolling out-rate columns found to widen (keep_cols not present).")

    df["is_home_defense"] = (df[fielding_team_col] == df[home_team_col]).astype(int)

    dups = df.duplicated([game_id_col, "is_home_defense"]).sum()
    if dups:
        raise ValueError(f"Found {dups} duplicates of (game_id, is_home_defense).")

    base = [game_id_col, date_col, home_team_col, away_team_col]
    use = base + ["is_home_defense"] + list(keep_cols)
    tmp = df[use].copy()

    home = tmp[tmp["is_home_defense"] == 1].drop(columns=["is_home_defense"])
    away = tmp[tmp["is_home_defense"] == 0].drop(columns=["is_home_defense", date_col, home_team_col, away_team_col])

    home = home.rename(columns={c: f"{c}_home" for c in keep_cols})
    away = away.rename(columns={c: f"{c}_away" for c in keep_cols})

    out = home.merge(away, on=game_id_col, how="outer", validate="one_to_one")
    out = out.sort_values([date_col, game_id_col]).reset_index(drop=True)

    # ---- final column order: 3G home/away, then 7G home/away ----
    ordered_rates = []
    if "roll_3G_BIP_out_rate" in keep_cols:
        ordered_rates += ["roll_3G_BIP_out_rate_home", "roll_3G_BIP_out_rate_away"]
    if "roll_7G_BIP_out_rate" in keep_cols:
        ordered_rates += ["roll_7G_BIP_out_rate_home", "roll_7G_BIP_out_rate_away"]

    out = out[[game_id_col, date_col, home_team_col, away_team_col, *ordered_rates]]

    return out


def make_fielding_out_rate_deltas(
    df: pd.DataFrame,
    game_id_col: str = "game_id",
    date_col: str = "game_date",
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
    col_3g_home: str = "roll_3G_BIP_out_rate_home",
    col_3g_away: str = "roll_3G_BIP_out_rate_away",
    col_7g_home: str = "roll_7G_BIP_out_rate_home",
    col_7g_away: str = "roll_7G_BIP_out_rate_away",
) -> pd.DataFrame:
    """
    Create a new dataframe with only home-away deltas
    for rolling BIP out-rate fielding features.
    """
    required = [
        game_id_col, date_col, home_team_col, away_team_col,
        col_3g_home, col_3g_away, col_7g_home, col_7g_away
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df[[game_id_col, date_col, home_team_col, away_team_col]].copy()

    out["ΔBIP_out_rate_3G"] = df[col_3g_home] - df[col_3g_away]
    out["ΔBIP_out_rate_7G"] = df[col_7g_home] - df[col_7g_away]

    return out
