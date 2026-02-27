import pandas as pd


def inspect_game_timeline(
    season_df: pd.DataFrame,
    gid,
    game_id_col: str = "game_id",
    inning_col: str = "inning",
    topbot_col: str = "inning_topbot",
    at_bat_col: str = "at_bat_number",
    pitch_col: str = "pitch_number",
    batter_id_col: str = "batter_id",  # <-- updated default
    tail_n: int = 25,
    include_cols: tuple[str, ...] = (
        "game_date",
        "inning",
        "inning_topbot",
        "at_bat_number",
        "pitch_number",
        "batter_id",                    # <-- updated
        "events",
        "description",
        "post_home_score",
        "post_away_score",
    ),
) -> pd.DataFrame:
    """
    Return a chronologically ordered slice of a single game's pitch/PA timeline.

    - Adds topbot_flag: Top=0, Bot=1 for correct half-inning ordering.
    - Sorts by: inning, topbot_flag, at_bat_number (if present), pitch_number (if present), original row index.
    - Returns the last `tail_n` rows (use tail_n=None to return full game).
    - Includes batter id column (defaults to "batter_id") for cross-referencing.
    """
    df = season_df.copy()

    # Fallback for id col (Statcast often uses game_pk)
    if game_id_col not in df.columns:
        if "game_pk" in df.columns:
            game_id_col = "game_pk"
        else:
            raise ValueError("Need a game id column: game_id or game_pk not found.")

    g = df[df[game_id_col] == gid].copy()
    if g.empty:
        raise ValueError(f"No rows found for {game_id_col} == {gid}")

    # Top=0, Bot=1
    if topbot_col not in g.columns:
        raise ValueError(f"'{topbot_col}' not found in dataframe columns.")
    tb = g[topbot_col].astype(str).str.lower().str.strip()
    g["topbot_flag"] = tb.map({"top": 0, "t": 0, "bot": 1, "bottom": 1, "b": 1})

    # Stable tie-breaker
    g["_row"] = g.index

    # Build sort columns
    sort_cols = []
    if inning_col in g.columns:
        sort_cols.append(inning_col)
    sort_cols.append("topbot_flag")

    if at_bat_col in g.columns:
        sort_cols.append(at_bat_col)
    if pitch_col in g.columns:
        sort_cols.append(pitch_col)

    sort_cols.append("_row")

    g_sorted = g.sort_values(sort_cols)

    # If user passed a different batter_id_col, swap it into include list
    normalized_include = tuple(
        batter_id_col if c == "batter_id" else c for c in include_cols
    )

    cols = [c for c in normalized_include if c in g_sorted.columns]

    # Ensure these helper cols are visible
    if game_id_col not in cols:
        cols.insert(0, game_id_col)
    if "topbot_flag" not in cols:
        cols.insert(1, "topbot_flag")

    out = g_sorted[cols]

    if tail_n is None:
        return out.reset_index(drop=True)

    return out.tail(tail_n).reset_index(drop=True)



