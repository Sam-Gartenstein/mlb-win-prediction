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

def filter_plate_appearances(df: pd.DataFrame, events_col: str = "events") -> pd.DataFrame:
    """
    Keep only PA-ending rows (events in the allowed set), excluding 'truncated_pa'.
    """
    ev = df[events_col].astype("string")
    mask = ev.notna() & ev.isin(PA_ENDING_EVENTS) & ~ev.eq("truncated_pa")
    return df.loc[mask].copy()


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
