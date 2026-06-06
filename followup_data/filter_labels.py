import pandas as pd
import sys

def filter_csv(input_path, output_path):
    df = pd.read_csv(input_path)

    # Keep rows where file_name does NOT start with a letter.
    # Cast to string and check the first character with isalpha().
    mask = ~df["file_name"].astype(str).str[0].str.isalpha().fillna(False)
    filtered = df[mask]

    filtered.to_csv(output_path, index=False)
    print(f"Kept {len(filtered)} of {len(df)} rows -> {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter.py input.csv output.csv")
        sys.exit(1)
    filter_csv(sys.argv[1], sys.argv[2])