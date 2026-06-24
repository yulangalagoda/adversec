""""
cleaning.py
===========
Load, audit, and de-duplicate the CICIoV2024 decimal dataset.
The notebook can call each function to follow step by step while displaying results along the way.
"""


# Library import
import pandas as pd # To process dataset tables
import config       # Adversec's own settings file


# Load raw data
def load_raw_data():
        """
        Load all six per-class CSVs into one labelled DataFrame.
        Returns a single DF with all rows from all classes, each row tagged with its true class in a new 'true_class' column.
        """

        # To collect each file's DF
        frames = []

        # Loop over the files
        for class_name, filename in config.RAW_FILES.items():
                # Buid the full path
                path = config.RAW_DIR / filename

                # Read CSV into a DF
                df = pd.read_csv(path)

                # Strip whitespace from column headers
                df.columns = [c.strip() for c in df.columns]

                # Record the ground-truth class for every row from this file
                df["true_class"] = class_name                   
                
                # Add this file's DF to the collection
                frames.append(df)   

                # Print a progress line
                print(f" loaded {filename:42s} rows={len(df):>9,}")     
        
        # Stack all 6 DF vertically into one
        combined = pd.concat(frames, ignore_index=True)
        return combined


# Auditing duplication
def audit_duplication(df, subset):
        """
        Measure how duplicated the data is over a given set of columns.

        A "signature" is one unique combination of the 'subset' columns.
        The duplication rate is the fraction of rows that are not the first time their signatures appear,
        i.e. the proportion of rows that are redundant copies.

        Args:
            df: the DataFrame to audit.
            subset: list of columns that together define a signature.
        
        Returns:
            A dictionary of counts and rates, ready to print or save to a report.
        """

        # Total number of rows in the data
        total = len(df)

        # Keep the first occurrence of each unique signature
        n_unique = len(df.drop_duplicates(subset=subset))

        # Everything that is not unique is a duplicate copy
        n_duplicate = total - n_unique

        # The duplication rate as a fraction: safenet against divide-by-zero on empty data
        rate = (n_duplicate / total) if total else 0.0

        # Bundle the findings into a dictionary
        return {
                "total_rows": total,
                "unique_signatures": n_unique,
                "duplicate_rows": n_duplicate,
                "duplication_rate": round(rate, 6),
                "duplication_rate_pct": round(rate*100, 4),
        }


# Strict de-duplication
def strict_dedup(df, feature_columns):
        """
        Keep one row per unique (features + class) signature.

        This is the headline experiment. It isolates the genuinely unique attack and benign signatures.
        It discards the redundant copies that inflate accuracy.

        We de-duplicate on the feature columns plus true_class.
        So a pattern that legitimately appears under two classes is kept once per class rather than collapsed into one.
        """

        # The signature: the 9 CAN features together with the authoritative class
        signature = feature_columns + ["true_class"]
        
        # Keep the first unique occurrence of each signature
        deduped = df.drop_duplicates(subset=signature, keep="first").reset_index(drop=True)
        return deduped


# Relaxed de-duplication: Fallback
def relaxed_dedup(df, strict_df, feature_columns, min_rows_per_class):
        """
        Stratified relaxed de-duplication: the data-sufficiency fallback.

        Start from the strict unique signatures.
        For any class whose strict count is below min_rows_per_class, add some of its duplicates until it reaches the target or runs out.
        Classes already above the target are left as their strict signatures only.

        The result is a deliberately harder-than-original dataset that still gives minority classes enough voume for the 1D-CNN to converge on.
        """

        # The signature: the 9 CAN features together with the authoritative class
        signature = feature_columns + ["true_class"]

        # A list to collect pieces when building the relaxed set class by class
        pieces = []

        # Go through each class present in the data, in sorted order for determinism
        for class_name in sorted(df["true_class"].unique()):
                # This class' unique rows
                strict_rows = strict_df[strict_df["true_class"] == class_name]
                n_strict = len(strict_rows)

                # If we have enough unique signatures, leave them as is
                if n_strict >= min_rows_per_class:
                        pieces.append(strict_rows)
                        continue
                
                # Otherwise top-up
                class_all = df[df["true_class"] == class_name]

                # Find duplicate rows that is not a first occurrece
                is_duplicate = class_all.duplicated(subset=signature, keep="first")
                dup_pool = class_all[is_duplicate]

                # How many more rows required
                need = min_rows_per_class - n_strict
                take = min(need, len(dup_pool))

                # Sample that many duplicates, with the fixed seed for reproductability
                topup = dup_pool.sample(n=take, random_state=config.RANDOM_SEED)

                # This class' relaxed set = its unique rows + the sampled duplicates
                pieces.append(pd.concat([strict_rows, topup], ignore_index=True))
        
        # Stack every class's piece into one DF and shuffle so classes are not grouped
        relaxed = pd.concat(pieces, ignore_index=True)
        relaxed = relaxed.sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)
        return relaxed