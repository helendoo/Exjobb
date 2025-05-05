import pandas as pd

def compare_holes(hole1_id: str, hole2_id: str, df: pd.DataFrame) -> str:
    h1 = df[df["holeID"].astype(str) == str(hole1_id)]
    h2 = df[df["holeID"].astype(str) == str(hole2_id)]

    if h1.empty or h2.empty:
        return f"One or both holes not found: Hole {hole1_id}, Hole {hole2_id}."

    h1_mean = h1.mean(numeric_only=True)
    h2_mean = h2.mean(numeric_only=True)

    comparison = pd.DataFrame({
        "Metric": h1_mean.index,
        f"Hole {hole1_id}": h1_mean.values,
        f"Hole {hole2_id}": h2_mean.values,
        "Difference": (h1_mean - h2_mean).values
    })

    summary_lines = [f"Comparison between Hole {hole1_id} and Hole {hole2_id}:"]
    for _, row in comparison.iterrows():
        metric = row["Metric"]
        val1 = row[f"Hole {hole1_id}"]
        val2 = row[f"Hole {hole2_id}"]
        diff = row["Difference"]
        summary_lines.append(f" - {metric}: {val1:.2f} vs {val2:.2f} (Δ {diff:+.2f})")

    return "\n".join(summary_lines)
