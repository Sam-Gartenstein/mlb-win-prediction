import pandas as pd


def make_game_outcomes_from_statcast_maxscore(
    pa_df: pd.DataFrame,
    game_id_col: str = "game_id",
    home_score_col: str = "post_home_score",
    away_score_col: str = "post_away_score",
    add_topbot_flag: bool = False,
    inning_topbot_col: str = "inning_topbot",
) -> pd.DataFrame:
    """
    One row per game using max(post_home_score) and max(post_away_score) as final scores.
    Robust to extra innings AND walk-offs (post-play scores).
    Optionally adds a numeric top/bot flag useful for ordering within games.
    """
    df = pa_df.copy()

    # Handle id column (raw Statcast often uses game_pk)
    if game_id_col not in df.columns:
        if "game_pk" in df.columns:
            game_id_col = "game_pk"
        else:
            raise ValueError("Need a game id column: game_id or game_pk not found.")

    # Ensure numeric scores
    if home_score_col not in df.columns or away_score_col not in df.columns:
        raise ValueError(
            f"Missing score columns. Expected '{home_score_col}' and '{away_score_col}'. "
            f"Available score columns: {[c for c in df.columns if 'score' in c]}"
        )

    df[home_score_col] = pd.to_numeric(df[home_score_col], errors="coerce")
    df[away_score_col] = pd.to_numeric(df[away_score_col], errors="coerce")

    # Optional ordering helper: Top=0, Bot=1
    if add_topbot_flag:
        if inning_topbot_col not in df.columns:
            raise ValueError(f"add_topbot_flag=True but '{inning_topbot_col}' not found.")
        tb = df[inning_topbot_col].astype(str).str.lower()
        df["topbot_flag"] = tb.map({"top": 0, "bot": 1, "bottom": 1})
        # If you have other encodings (e.g., 'T', 'B'), include them:
        df["topbot_flag"] = df["topbot_flag"].fillna(
            tb.map({"t": 0, "b": 1})
        )

    agg_dict = dict(
        game_date=("game_date", "first"),
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
        final_home_score=(home_score_col, "max"),
        final_away_score=(away_score_col, "max"),
    )

    if add_topbot_flag:
        agg_dict["has_topbot_flag"] = ("topbot_flag", lambda s: int(s.notna().any()))

    outcomes = df.groupby(game_id_col, as_index=False).agg(**agg_dict)

    outcomes["home_win"] = (outcomes["final_home_score"] > outcomes["final_away_score"]).astype(int)
    outcomes["run_diff"] = outcomes["final_home_score"] - outcomes["final_away_score"]

    ordered = [
        game_id_col, "game_date", "home_team", "away_team",
        "final_home_score", "final_away_score",
        "home_win", "run_diff",
    ]
    if add_topbot_flag:
        ordered.append("has_topbot_flag")

    return outcomes[ordered]

