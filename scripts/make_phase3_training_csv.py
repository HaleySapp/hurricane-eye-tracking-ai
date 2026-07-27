import pandas as pd

INPUT_CSV = "combined_correction_data2.csv"
OUTPUT_CSV = "phase3_training_data2.csv"

def main():
    df = pd.read_csv(INPUT_CSV)

    required_cols = ["x", "y", "true_x", "true_y"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Keep only rows where true centers exist
    df = df.dropna(subset=["true_x", "true_y"]).copy()

    # Compute correction targets
    df["error_x"] = df["true_x"] - df["x"]
    df["error_y"] = df["true_y"] - df["y"]

    # Optional: keep only rows that were found
    if "found" in df.columns:
        df = df[df["found"] == True].copy()

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved Phase 3 training CSV: {OUTPUT_CSV}")
    print(f"Rows kept: {len(df)}")
    print("\nPreview:")
    print(df[["filename", "x", "y", "true_x", "true_y", "error_x", "error_y"]].head())

if __name__ == "__main__":
    main()