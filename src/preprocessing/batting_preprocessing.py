import numpy as np
import pandas as pd


def add_batting_indicators(pa_df: pd.DataFrame, events_col: str = "events") -> pd.DataFrame:
    """
    Add minimal PA-level indicators (keep the PA table light).
    We'll compute AB/TB/etc. later during aggregation.
    """
    out = pa_df.copy()
    ev = out[events_col].astype("string")

    # Walk / HBP / Sac / Catcher's interference
    out["is_bb"]  = ev.isin(["walk", "intent_walk"]).astype(int)
    out["is_hbp"] = ev.eq("hit_by_pitch").astype(int)
    out["is_sf"]  = ev.isin(["sac_fly", "sac_fly_double_play"]).astype(int)
    out["is_sh"]  = ev.eq("sac_bunt").astype(int)
    out["is_ci"]  = ev.eq("catcher_interf").astype(int)

    # Hits
    out["is_1b"] = ev.eq("single").astype(int)
    out["is_2b"] = ev.eq("double").astype(int)
    out["is_3b"] = ev.eq("triple").astype(int)
    out["is_hr"] = ev.eq("home_run").astype(int)

    return out


def split_batting_home_away(pa_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (away_batting_df, home_batting_df)
    Each row is a PA, with a `batting_team` column set.
    """
    away_batting = pa_df.loc[pa_df["inning_topbot"].eq("Top")].copy()
    away_batting["batting_team"] = away_batting["away_team"]

    home_batting = pa_df.loc[pa_df["inning_topbot"].eq("Bot")].copy()
    home_batting["batting_team"] = home_batting["home_team"]

    return away_batting, home_batting


def aggregate_team_game_batting(pa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate PA-level rows to team-game totals needed for OBP/ISO later.
    Returns one row per (game_id, batting_team).
    """
    df = pa_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Derived totals
    df["H"]  = (df["is_1b"] + df["is_2b"] + df["is_3b"] + df["is_hr"]).astype(int)
    df["TB"] = (1*df["is_1b"] + 2*df["is_2b"] + 3*df["is_3b"] + 4*df["is_hr"]).astype(int)

    # AB excludes BB, HBP, SF, SH (sac bunt), CI
    df["AB"] = (1 - (df["is_bb"] + df["is_hbp"] + df["is_sf"] + df["is_sh"] + df["is_ci"])).clip(lower=0).astype(int)

    out = (
        df.groupby(["game_id", "game_date", "batting_team"], as_index=False)
          .agg(
              PA=("events", "size"),
              AB=("AB", "sum"),
              H=("H", "sum"),
              TB=("TB", "sum"),
              BB=("is_bb", "sum"),
              HBP=("is_hbp", "sum"),
              SF=("is_sf", "sum"),
              SH=("is_sh", "sum"),
              CI=("is_ci", "sum"),
              HR=("is_hr", "sum"),
              _1B=("is_1b", "sum"),
              _2B=("is_2b", "sum"),
              _3B=("is_3b", "sum"),
          )
    )
    return out


def add_time_rolling_batting_sums(
    team_game_df: pd.DataFrame,
    team_col: str = "batting_team",
    date_col: str = "game_date",
    game_id_col: str = "game_id",
    sum_cols: list[str] | None = None,
    windows: tuple[str, ...] = ("3D", "7D"),
    min_periods: int = 1,
    prefix: str = "roll_",
    output_order: str = "team",  # "team" or "original"
) -> pd.DataFrame:

    df = team_game_df.copy()
    df["_orig_row"] = np.arange(len(df))
    df[date_col] = pd.to_datetime(df[date_col])

    if sum_cols is None:
        default_cols = ["AB", "H", "BB", "HBP", "SF", "HR", "_2B", "_3B", "TB"]
        sum_cols = [c for c in default_cols if c in df.columns]
    else:
        missing = [c for c in sum_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Requested sum_cols not found: {missing}")

    sort_keys = [team_col, date_col] + ([game_id_col] if game_id_col in df.columns else [])
    df = df.sort_values(sort_keys, kind="mergesort")

    def _apply(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(
            [date_col] + ([game_id_col] if game_id_col in g.columns else []),
            kind="mergesort",
        ).copy()

        shifted = g[sum_cols].shift(1)
        base = pd.concat([g[[date_col]].reset_index(drop=True), shifted.reset_index(drop=True)], axis=1)

        for w in windows:
            rolled = (
                base.rolling(window=w, on=date_col, min_periods=min_periods)
                    .sum(numeric_only=True)
                    .reindex(columns=sum_cols)
            )
            rolled.columns = [f"{prefix}{w}_{c}" for c in sum_cols]
            g[rolled.columns] = rolled.to_numpy()

        return g

    out = df.groupby(team_col, group_keys=False, sort=False).apply(_apply)

    # ---- OUTPUT ORDER ----
    if output_order == "original":
        out = out.sort_values("_orig_row", kind="mergesort")
    elif output_order == "team":
        out = out.sort_values(sort_keys, kind="mergesort")
    else:
        raise ValueError('output_order must be "team" or "original"')

    out = out.drop(columns="_orig_row").reset_index(drop=True)
    return out


def add_rolling_obp_iso(
    df: pd.DataFrame,
    windows: tuple[str, ...] = ("3D", "7D"),
    prefix: str = "roll_",
    out_prefix: str = "roll_",
) -> pd.DataFrame:
    """
    Add rolling OBP and ISO columns derived from existing rolling sum columns.

    Assumes the dataframe already contains rolling sums like:
      - {prefix}{w}_AB, {prefix}{w}_H, {prefix}{w}_BB, {prefix}{w}_HBP,
        {prefix}{w}_SF, {prefix}{w}_HR, {prefix}{w}__2B, {prefix}{w}__3B

    Creates:
      - {out_prefix}{w}_OBP
      - {out_prefix}{w}_ISO

    OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
    ISO = (2B + 2*3B + 3*HR) / AB
    """
    out = df.copy()

    for w in windows:
        AB  = out[f"{prefix}{w}_AB"]
        H   = out[f"{prefix}{w}_H"]
        BB  = out[f"{prefix}{w}_BB"]
        HBP = out[f"{prefix}{w}_HBP"]
        SF  = out[f"{prefix}{w}_SF"]
        HR  = out[f"{prefix}{w}_HR"]
        _2B = out[f"{prefix}{w}__2B"]
        _3B = out[f"{prefix}{w}__3B"]

        denom_obp = AB + BB + HBP + SF
        obp = (H + BB + HBP) / denom_obp
        obp = obp.where(denom_obp > 0)  # avoid inf/0

        iso = (_2B + 2*_3B + 3*HR) / AB
        iso = iso.where(AB > 0)  # avoid inf/0

        out[f"{out_prefix}{w}_OBP"] = obp
        out[f"{out_prefix}{w}_ISO"] = iso

    return out


def add_rolling_obp_iso_batch(
    dfs: dict[str, pd.DataFrame] | list[pd.DataFrame],
    windows: tuple[str, ...] = ("3D", "7D"),
    prefix: str = "roll_",
    out_prefix: str = "roll_",
) -> dict[str, pd.DataFrame] | list[pd.DataFrame]:
    """
    Apply add_rolling_obp_iso to many dataframes.

    - If dfs is a dict: returns a dict with the same keys.
    - If dfs is a list: returns a list in the same order.
    """
    if isinstance(dfs, dict):
        return {
            name: add_rolling_obp_iso(df, windows=windows, prefix=prefix, out_prefix=out_prefix)
            for name, df in dfs.items()
        }
    elif isinstance(dfs, list):
        return [
            add_rolling_obp_iso(df, windows=windows, prefix=prefix, out_prefix=out_prefix)
            for df in dfs
        ]
    else:
        raise TypeError("dfs must be a dict[str, DataFrame] or a list[DataFrame].")


def combine_home_away_batting_rolls(
    home_batting: pd.DataFrame,
    away_batting: pd.DataFrame,
    game_id_col: str = "game_id",
    date_col: str = "game_date",
    metrics: tuple[str, ...] = ("roll_3D_OBP", "roll_3D_ISO", "roll_7D_OBP", "roll_7D_ISO"),
) -> pd.DataFrame:
    """
    Combine home + away batting rolling metrics into one row per game_id.

    Output columns:
      game_id, game_date,
      roll_3D_OBP_home, roll_3D_ISO_home, roll_7D_OBP_home, roll_7D_ISO_home,
      roll_3D_OBP_away, roll_3D_ISO_away, roll_7D_OBP_away, roll_7D_ISO_away
    """
    # checks
    for df, label in [(home_batting, "home_batting"), (away_batting, "away_batting")]:
        missing = [c for c in [game_id_col, date_col, *metrics] if c not in df.columns]
        if missing:
            raise ValueError(f"{label} missing columns: {missing}")

        dups = df.duplicated([game_id_col]).sum()
        if dups:
            raise ValueError(f"{label} has {dups} duplicate {game_id_col} rows (expected 1 per game_id).")

    home = home_batting[[game_id_col, date_col, *metrics]].copy()
    away = away_batting[[game_id_col, date_col, *metrics]].copy()

    home = home.rename(columns={m: f"{m}_home" for m in metrics})
    away = away.rename(columns={m: f"{m}_away" for m in metrics})

    # keep only one game_date column (prefer home)
    away = away.drop(columns=[date_col])

    out = (
        home.merge(away, on=game_id_col, how="inner")
            .sort_values([date_col, game_id_col])
            .reset_index(drop=True)
    )
    return out


def make_batting_delta_df(
    game_batting_rolls: pd.DataFrame,
    game_id_col: str = "game_id",
    date_col: str = "game_date",
    metrics: tuple[str, ...] = ("roll_3D_OBP", "roll_3D_ISO", "roll_7D_OBP", "roll_7D_ISO"),
) -> pd.DataFrame:
    """
    Create a NEW dataframe with batting deltas (home - away) for selected rolling metrics.

    Expects columns like:
      <metric>_home, <metric>_away

    Returns columns:
      game_id, game_date, Δ<metric>
    """
    df = game_batting_rolls.copy()

    required = [game_id_col, date_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required base columns: {missing}")

    out = df[[game_id_col, date_col]].copy()

    for m in metrics:
        home_col = f"{m}_home"
        away_col = f"{m}_away"
        if home_col not in df.columns or away_col not in df.columns:
            raise ValueError(f"Missing columns for metric '{m}': {home_col}, {away_col}")

        out[f"Δ{m}"] = df[home_col] - df[away_col]

    return out


