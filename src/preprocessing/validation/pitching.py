import pandas as pd


def starter_complete_game_flags(pa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per (game_id, pitching_team) with a flag indicating whether
    ONLY the starter pitched for that team in that game (i.e., no bullpen appearances).
    """
    df = pa_df.copy()

    # Defensive: keep only rows that represent plate appearances (optional, but usually desired)
    if "is_pa_countable" in df.columns:
        df = df[df["is_pa_countable"] == True].copy()

    # For each (game, team): did we ever see a non-starter pitcher?
    team_flags = (
        df.groupby(["game_id", "pitching_team"], as_index=False)
          .agg(
              used_bullpen=("is_starter", lambda s: (s == 0).any()),
              n_rows=("is_starter", "size"),
          )
    )

    team_flags["starter_complete_game"] = ~team_flags["used_bullpen"]
    return team_flags


def summarize_complete_games(pa_df: pd.DataFrame) -> dict:
    """
    Summarizes counts of:
      - team-level starter complete games
      - games where BOTH teams had starter complete games
      - games where AT LEAST ONE team had starter complete games
    """
    team_flags = starter_complete_game_flags(pa_df)

    # How many games in this season df?
    n_games = team_flags["game_id"].nunique()

    # Team-level count (each game has up to 2 pitching teams)
    n_team_complete = int(team_flags["starter_complete_game"].sum())

    # Game-level: both teams / any team
    game_flags = (
        team_flags.groupby("game_id", as_index=False)
                  .agg(
                      teams_in_game=("pitching_team", "nunique"),
                      both_starters_complete=("starter_complete_game", lambda x: x.all()),
                      any_starter_complete=("starter_complete_game", lambda x: x.any()),
                  )
    )

    n_both = int(game_flags["both_starters_complete"].sum())
    n_any = int(game_flags["any_starter_complete"].sum())

    # Optional: identify games missing a pitching team after filtering
    missing_team_games = game_flags.loc[game_flags["teams_in_game"] != 2, "game_id"].tolist()

    return {
        "n_games": n_games,
        "n_team_starter_complete_games": n_team_complete,
        "n_games_both_starters_complete": n_both,
        "n_games_any_starter_complete": n_any,
        "both_game_ids": game_flags.loc[game_flags["both_starters_complete"], "game_id"].tolist(),
        "any_game_ids": game_flags.loc[game_flags["any_starter_complete"], "game_id"].tolist(),
        "games_with_missing_pitching_team": missing_team_games,
        "n_games_with_missing_pitching_team": len(missing_team_games),
    }


def validate_starter_lines_by_year(
    starter_lines_by_year: dict[int, pd.DataFrame],
    strict: bool = True,
) -> None:
    """
    Data-quality check for starter_lines tables:
      - exactly 2 starter rows per game_id
      - pitcher_name coverage
      - row count matches 2 * unique games

    If strict=True, raises AssertionError on any violation.
    """
    for year, df in starter_lines_by_year.items():
        n_games = df["game_id"].nunique()
        n_rows = len(df)
        na_names = df["pitcher_name"].isna().sum()

        counts = df.groupby("game_id").size()
        bad = counts[counts != 2]

        print(f"\n=== {year} starter_lines ===")
        print(f"rows: {n_rows:,} | unique games: {n_games:,} | expected rows (2*games): {2*n_games:,}")
        print(f"pitcher_name NaNs: {na_names:,}")
        print(f"games with starter rowcount != 2: {len(bad)}")
        if len(bad):
            print("example bad game_ids:", list(bad.index[:10]))

        if strict:
            assert n_rows == 2 * n_games, f"{year}: row count mismatch (rows={n_rows}, expected={2*n_games})"
            assert len(bad) == 0, f"{year}: found games with starter rowcount != 2 (n_bad={len(bad)})"
            assert na_names == 0, f"{year}: pitcher_name has NaNs (n_na={na_names})"

