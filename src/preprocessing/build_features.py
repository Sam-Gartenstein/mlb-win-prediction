import pandas as pd


def build_feature_set(
    df: pd.DataFrame,
    *,
    windows: dict,
    include_pitch: bool = True,
    include_bat: bool = True,
    include_field: bool = True,
    include_win: bool = True,
) -> pd.DataFrame:
    cols = []

    if include_pitch:
        w = windows["pitch"]
        pitch_prefixes = ("Δstarter_", "Δbullpen_")
        cols += [
            c for c in df.columns
            if c.startswith(pitch_prefixes) and c.endswith(f"_{w}")
        ]

    if include_bat:
        w = windows["bat"]
        cols += [
            c for c in df.columns
            if c.startswith(f"Δroll_{w}_")
        ]

    if include_field:
        w = windows["field"]
        cols += [
            c for c in df.columns
            if c.startswith("ΔBIP_") and c.endswith(f"_{w}")
        ]

    if include_win:
        w = windows["win"]
        win_col = f"Δwin_pct_{w}"
        if win_col in df.columns:
            cols.append(win_col)

    cols = list(dict.fromkeys(cols))

    if not cols:
        raise ValueError(f"No features selected with windows={windows}")

    return df[cols].copy()
