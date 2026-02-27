import pandas as pd


def add_game_id(df):
    df = df.copy()

    # YYYYMMDD
    df["game_date_temp"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y%m%d")

    # clean pitcher name:
    # - remove commas
    # - collapse whitespace
    # - replace spaces with underscores
    df["pitcher_id"] = (
        df["pitcher_name"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )

    df["game_id"] = (
        df["game_date_temp"]
        + "_"
        + df["away_team"].astype(str)
        + "@"
        + df["home_team"].astype(str)
        + "_"
        + df["pitcher_id"]
    )

    return df.drop(columns=["game_date_temp", "pitcher_id"])


def merge_game_number_and_pitcher(at_bats_df: pd.DataFrame, double_headers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-merge game_number + pitcher from double_headers_df into at_bats_df by game_id.
    Keeps only columns: game_id, game_number, pitcher from double_headers_df (deduped).
    """
    keep = double_headers_df[["game_id", "game_number", "pitcher"]].drop_duplicates()

    return at_bats_df.merge(keep, on="game_id", how="left")


def trim_game_id_inplace(df: pd.DataFrame, col: str = "game_id") -> pd.DataFrame:
    """
    Overwrite game_id from:
    YYYYMMDD_AWAY@HOME_Pitcher_Name
    to:
    YYYYMMDD_AWAY@HOME
    """
    df = df.copy()
    df[col] = df[col].astype(str).str.split("_", n=2).str[:2].str.join("_")
    return df


def append_game_number_to_game_id(
    df: pd.DataFrame,
    game_id_col: str = "game_id",
    game_number_col: str = "game_number",
    drop_cols: tuple[str, ...] = ("game_number", "pitcher"),
) -> pd.DataFrame:
    """
    If game_number is not NaN, append it to game_id:
      YYYYMMDD_AWAY@HOME  ->  YYYYMMDD_AWAY@HOME_1

    Leaves game_id unchanged when game_number is NaN.

    After appending, drops columns in `drop_cols` (if present).
    """
    df = df.copy()

    mask = df[game_number_col].notna()

    df.loc[mask, game_id_col] = (
        df.loc[mask, game_id_col].astype(str)
        + "_"
        + df.loc[mask, game_number_col].astype(int).astype(str)
    )

    # Drop the columns if they exist
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df


PA_ENDING_EVENTS = {
    "single", "double", "triple", "home_run",
    "walk", "intent_walk", "hit_by_pitch",
    "strikeout", "strikeout_double_play",
    "field_out", "force_out",
    "fielders_choice", "fielders_choice_out",
    "field_error",
    "sac_fly", "sac_bunt", "sac_fly_double_play",
    "catcher_interf",
    "double_play", "grounded_into_double_play", "triple_play",
}

def filter_plate_appearances(
    df: pd.DataFrame,
    events_col: str = "events",
    starter_col: str = "is_starter",
) -> pd.DataFrame:
    """
    Keep PA-ending rows (events in PA_ENDING_EVENTS), excluding 'truncated_pa'.

    Starter fallback rule:
      - If a starter has >=1 PA-ending row in that game/team, keep ONLY PA-ending rows.
      - If a starter has 0 PA-ending rows in that game/team (i.e., all events are null),
        keep the last pitch per PA-state for that starter:
          (game_id, pitching_team, inning, inning_topbot, outs_when_up)
    """
    out = df.copy()
    ev = out[events_col].astype("string")

    # Base PA-ending rows
    pa_end_mask = ev.notna() & ev.isin(PA_ENDING_EVENTS) & ~ev.eq("truncated_pa")

    # If we don't have starter info, just return PA-ending
    if starter_col not in out.columns:
        return out.loc[pa_end_mask].copy()

    # ---- Starter fallback: only for starters with ZERO PA-ending rows in their game/team ----
    starters = out[out[starter_col] == 1].copy()
    starters_ev = ev.loc[starters.index]

    # For each (game_id, pitching_team), does the starter ever have a PA-ending row?
    key = ["game_id", "pitching_team"]
    starters_has_pa_end = (
        starters.assign(_pa_end=pa_end_mask.loc[starters.index].values)
                .groupby(key)["_pa_end"].any()
    )
    # We want ONLY groups where starter_has_pa_end is False
    fallback_groups = starters_has_pa_end[~starters_has_pa_end].index  # MultiIndex of (game_id, pitching_team)

    if len(fallback_groups) == 0:
        return out.loc[pa_end_mask].copy()

    # Subset starter rows for those fallback groups
    fallback = starters.set_index(key).loc[list(fallback_groups)].reset_index()

    # Define PA-state proxy
    pa_state_cols = ["game_id", "pitching_team", "inning", "inning_topbot", "outs_when_up"]

    # Keep last pitch_number per PA-state (for these starters only)
    fallback_last = (
        fallback.sort_values(pa_state_cols + ["pitch_number"])
                .groupby(pa_state_cols, as_index=False)
                .tail(1)
    )

    # Combine: PA-ending rows for everyone + fallback rows for the rare starters with zero PA-ending events
    kept = pd.concat([out.loc[pa_end_mask], fallback_last], ignore_index=True)

    # De-dupe in case an overlap occurs
    kept = kept.drop_duplicates(subset=[
        "game_id","inning","inning_topbot","pitch_number","pitching_team","pitcher_id","batter_id"
    ])

    return kept


def combine_pitching_batting_deltas(
    pitching_deltas: pd.DataFrame,
    batting_deltas: pd.DataFrame,
    game_id_col: str = "game_id",
    date_col: str = "game_date",
    how: str = "inner",  # "inner" keeps only games present in both
) -> pd.DataFrame:
    """
    Merge pitching + batting deltas into one game-level feature table (one row per game_id).
    """
    # sanity checks
    for df, label in [(pitching_deltas, "pitching_deltas"), (batting_deltas, "batting_deltas")]:
        missing = [c for c in [game_id_col, date_col] if c not in df.columns]
        if missing:
            raise ValueError(f"{label} missing required columns: {missing}")
        dups = df.duplicated([game_id_col]).sum()
        if dups:
            raise ValueError(f"{label} has {dups} duplicate {game_id_col} rows (expected 1 per game_id).")

    # avoid duplicate date columns if both have game_date
    left = pitching_deltas.copy()
    right = batting_deltas.drop(columns=[date_col], errors="ignore").copy()

    out = (
        left.merge(right, on=game_id_col, how=how)
            .sort_values([date_col, game_id_col])
            .reset_index(drop=True)
    )
    return out
