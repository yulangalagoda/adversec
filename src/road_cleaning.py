"""
road_cleaning.py
================
Load, parse, and label the ROAD (Oak Ridge) CAN bus dataset into the same dataframe shape the existing CICIoV2024 pipeline expects:
    columns ID, DATA_0..DATA_7 (integers 0-255), plus a 'true_class' column.

Once a ROAD dataframe is in that shape, every existing downstream module
(preprocessing, models, crossval, evaluation) runs on it unchanged.
"""

# Library imports
import json                         # To read the capture metadata JSON
import pandas as pd                 # DataFrame handling, same as cleaning.py
import config                       # Project settings file


# Parse one raw .log capture into a DataFrame
def parse_log_file(log_path):
        """
        Read one ROAD candump .log file into a DataFrame of CAN frames.

        Each raw line looks like:
            (1030000000.000000) can0 354#200A000000027480

        Split it into:
            timestamp : float seconds (absolute epoch)
            ID        : arbitration ID, hex string -> integer
            DATA_0..7 : the 8 payload bytes, each hex -> integer 0-255

        Args:
            log_path: full path to one .log file.

        Returns:
            A DataFrame with columns: timestamp, ID, DATA_0..DATA_7.
            No labels yet: labelling happens in a separate step.
        """

        # A list to collect one parsed dictionary per frame
        records = []

        # Open the file and read it line by line
        with open(log_path, "r") as f:
                for line in f:
                        # Remove trailing newline and surrounding whitespace
                        line = line.strip()

                        # Skip any empty lines defensively
                        if not line:
                                continue

                        # Split the line into its three space-separated parts:
                        # "(timestamp)", "can0", "ID#PAYLOAD"
                        parts = line.split()

                        # The timestamp sits inside parentheses: strip them, then float
                        timestamp = float(parts[0].strip("()"))

                        # The third part is ID#PAYLOAD: split on "#"
                        id_hex, payload_hex = parts[2].split("#")

                        # Convert the arbitration ID from hex text to an integer
                        can_id = int(id_hex, 16)

                        # Split the 16-char payload into 8 two-char byte strings,
                        # converting each from hex to an integer 0-255
                        data_bytes = [int(payload_hex[i:i+2], 16) for i in range(0, 16, 2)]

                        # Build one record: timestamp, ID, then the 8 bytes
                        record = {
                                "timestamp": timestamp,
                                "ID": can_id,
                        }
                        # Add DATA_0 through DATA_7 using the same names as config
                        for i in range(8):
                                record[f"DATA_{i}"] = data_bytes[i]

                        records.append(record)

        # Turn the list of dicts into a DataFrame in one go
        df = pd.DataFrame(records)
        return df

# Load the metadata JSON for a set of captures
def load_metadata(metadata_path):
        """
        Read the capture_metadata.json file into a dictionary.

        The JSON maps each capture name to its attack details:
        injection_id, injection_data_str, injection_interval, modified, etc.

        Args:
            metadata_path: full path to capture_metadata.json.

        Returns:
            A dict keyed by capture name.
        """
        with open(metadata_path, "r") as f:
                meta = json.load(f)
        return meta


# Turn a wildcard payload mask into fixed-byte constraints
def parse_injection_mask(injection_data_str):
        """
        Convert an injection payload string into a list of (byte_index, value)
        constraints, ignoring wildcard 'X' bytes.

        The string is 16 hex chars = 8 bytes. Some bytes are literal hex that
        must match (e.g. 'FF'), others are 'XX' wildcards that match anything.

        Example:
            "XXXXXXXXXXFFXXXX" -> only byte 5 is fixed, and must equal 0xFF
            -> returns [(5, 255)]

            "595945450000FFFF" -> every byte fixed
            -> returns [(0,89),(1,89),(2,69),(3,69),(4,0),(5,0),(6,255),(7,255)]

        Args:
            injection_data_str: the 16-char injection string, or None.

        Returns:
            A list of (byte_index, integer_value) pairs for the fixed bytes.
            Empty list if the string is None or all wildcards.
        """
        # Some captures (fuzzing, accelerator) have no fixed payload constraint
        if injection_data_str is None:
                return []

        constraints = []

        # Walk the string two characters at a time = one byte per step
        for byte_index in range(8):
                # Slice out this byte's two hex characters
                chunk = injection_data_str[byte_index*2 : byte_index*2 + 2]

                # 'XX' (any case) means wildcard: this byte can be anything, skip it
                if chunk.upper() == "XX":
                        continue

                # Otherwise it is a fixed byte: convert hex to int and record it
                constraints.append((byte_index, int(chunk, 16)))

        return constraints


# Label the frames of one attack capture
def label_attack_capture(df, capture_name, meta):
        """
        Add a 'true_class' column to a parsed attack capture.

        A frame is the attack (labelled with class_label) if ALL hold:
            - its ID equals the injection_id, AND
            - its elapsed time falls inside injection_interval, AND
            - every fixed byte in the injection mask matches.
        Every other frame in the capture is labelled 'benign'.

        Args:
            df: parsed capture DataFrame (from parse_log_file).
            capture_name: the key into meta, e.g. 'correlated_signal_attack_1'.
            meta: the metadata dict from load_metadata.

        Returns:
            The DataFrame with a new 'true_class' column.
        """
        # Pull this capture's metadata entry
        entry = meta[capture_name]

        # Convert absolute timestamps to elapsed seconds from the first frame,
        # because injection_interval is given in elapsed seconds.
        start_time = df["timestamp"].iloc[0]
        elapsed = df["timestamp"] - start_time

        # The injection target ID is given as a hex string like "0x6e0"
        injection_id = int(entry["injection_id"], 16)

        # The injection window start and end, in elapsed seconds
        interval_start, interval_end = entry["injection_interval"]

        # The fixed-byte constraints from the payload mask
        constraints = parse_injection_mask(entry["injection_data_str"])

        # Start by building a boolean mask that is True for attack frames.
        # Condition 1: ID matches the injection target
        is_attack = (df["ID"] == injection_id)

        # Condition 2: elapsed time inside the injection window
        is_attack = is_attack & (elapsed >= interval_start) & (elapsed <= interval_end)

        # Condition 3: every fixed byte must match
        for byte_index, value in constraints:
                is_attack = is_attack & (df[f"DATA_{byte_index}"] == value)

        # Default everything to benign, then overwrite the attack rows
        df["true_class"] = "benign"
        df.loc[is_attack, "true_class"] = capture_name

        return df


# Build the labelled attack set across all chosen captures
def load_attack_captures(attack_map, attacks_dir, metadata_path, feature_columns):
        """
        Parse and label every attack capture, pooling by clean class name.

        For each clean class (e.g. 'max-speedometer') we loop over its list of
        capture files, parse each, label the injected frames, keep ONLY those
        injected frames, and relabel them from the messy capture name to the
        clean class name. Benign frames from attack captures are handled later.

        Args:
            attack_map: dict of clean_class_name -> list of capture base names.
            attacks_dir: Path to the folder holding the .log files.
            metadata_path: Path to capture_metadata.json.
            feature_columns: the 9 CAN feature columns (for the signature report).

        Returns:
            A DataFrame of attack frames only, with 'true_class' set to the
            clean class name. Also prints pooled unique-signature counts.
        """
        # Load the metadata once
        meta = load_metadata(metadata_path)

        # Collect one DataFrame per clean class
        class_frames = []

        # Go through each clean class and its list of captures
        for clean_name, capture_list in attack_map.items():
                per_class = []

                # Decide which labeller this class needs
                is_fuzzing = clean_name in config.ROAD_FUZZING_CLASSES

                for capture_name in capture_list:
                        log_path = attacks_dir / f"{capture_name}.log"
                        d = parse_log_file(log_path)

                        # Route to the correct labeller
                        if is_fuzzing:
                                d = label_fuzzing_capture(d, capture_name, meta)
                        else:
                                d = label_attack_capture(d, capture_name, meta)

                        atk = d[d["true_class"] == capture_name].copy()
                        atk["true_class"] = clean_name
                        per_class.append(atk)

                pooled = pd.concat(per_class, ignore_index=True)
                n_raw = len(pooled)
                n_sig = len(pooled[feature_columns].drop_duplicates())
                print(f"{clean_name:20s} raw={n_raw:6d}  pooled_unique_signatures={n_sig}")
                class_frames.append(pooled)

        # Stack every class into one attack DataFrame
        attacks_df = pd.concat(class_frames, ignore_index=True)
        return attacks_df


# Build the benign set from ambient captures
def load_ambient_benign(ambient_captures, ambient_dir, sample_per_capture, random_seed):
        """
        Parse ambient captures and sample benign frames from each.

        Ambient captures contain only normal traffic (no injections), so every
        frame is benign. They are large, so we sample a fixed number from each
        with a fixed seed, to keep benign from swamping the attack classes and
        to stay reproducible.

        Args:
            ambient_captures: list of ambient capture base names.
            ambient_dir: Path to the ambient folder.
            sample_per_capture: max benign frames to draw from each capture.
            random_seed: seed for reproducible sampling.

        Returns:
            A DataFrame of benign frames with 'true_class' = 'benign'.
        """
        benign_pieces = []

        for capture_name in ambient_captures:
                # Build the path and parse the capture
                log_path = ambient_dir / f"{capture_name}.log"
                d = parse_log_file(log_path)

                # If the capture has more frames than our cap, sample down.
                # Otherwise take all of it.
                if len(d) > sample_per_capture:
                        d = d.sample(n=sample_per_capture, random_state=random_seed)

                # Every ambient frame is benign
                d["true_class"] = "benign"

                print(f"{capture_name:40s} sampled={len(d):6d}")
                benign_pieces.append(d)

        # Stack all ambient samples into one benign frame
        benign_df = pd.concat(benign_pieces, ignore_index=True)
        return benign_df



# Assemble the full ROAD dataset in CICIoV shape
def load_road_dataset(
        attack_map, ambient_captures,
        attacks_dir, ambient_dir, metadata_path,
        feature_columns, sample_per_capture, random_seed,
):
        """
        Build one labelled ROAD DataFrame in the same shape as the CICIoV2024
        pipeline expects: columns ID, DATA_0..DATA_7, true_class.

        Combines the attack frames (pooled by clean class name) with the benign
        frames sampled from ambient captures, then drops the timestamp column so
        the output matches CICIoV exactly and every downstream module runs
        unchanged.

        Args:
            attack_map: dict clean_class_name -> list of capture names.
            ambient_captures: list of ambient capture names for benign.
            attacks_dir, ambient_dir: Paths to the two folders.
            metadata_path: Path to capture_metadata.json.
            feature_columns: the 9 CAN feature columns.
            sample_per_capture: benign cap per ambient capture.
            random_seed: seed for reproducible benign sampling.

        Returns:
            A single DataFrame: ID, DATA_0..DATA_7, true_class.
        """
        # Build the attack side (pooled by clean class name)
        attacks_df = load_attack_captures(
                attack_map, attacks_dir, metadata_path, feature_columns,
        )

        # Build the benign side (sampled from ambient)
        benign_df = load_ambient_benign(
                ambient_captures, ambient_dir, sample_per_capture, random_seed,
        )

        # Stack attacks and benign into one frame
        combined = pd.concat([attacks_df, benign_df], ignore_index=True)

        # Drop the timestamp column: it was only needed for interval labelling,
        # and must not become a model feature. Keep exactly the CICIoV columns.
        keep_columns = feature_columns + ["true_class"]
        combined = combined[keep_columns]

        # Final report
        print()
        print("full ROAD dataset shape:", combined.shape)
        print(combined["true_class"].value_counts())

        return combined



# Label a fuzzing capture (wildcard ID, all-FF payload filter)
def label_fuzzing_capture(df, capture_name, meta):
        """
        Label a fuzzing capture cleanly, isolating the true injected frames.

        Fuzzing sprays random arbitration IDs, each carrying a maximum (all-FF)
        payload. The injection window also contains ordinary bus traffic, so an
        interval-only rule would mislabel that real traffic as attack. We
        therefore require BOTH:
            - the frame falls inside the injection window, AND
            - its payload is all-FF (every byte == 255),
        which uniquely identifies the injected frames.

        Args:
            df: parsed capture DataFrame.
            capture_name: metadata key, e.g. 'fuzzing_attack_1'.
            meta: metadata dict from load_metadata.

        Returns:
            The DataFrame with a 'true_class' column ('benign' or capture_name).
        """
        entry = meta[capture_name]

        start_time = df["timestamp"].iloc[0]
        elapsed = df["timestamp"] - start_time

        interval_start, interval_end = entry["injection_interval"]

        # Condition 1: inside the injection window
        in_window = (elapsed >= interval_start) & (elapsed <= interval_end)

        # Condition 2: payload is all-FF (255 in every data byte)
        data_cols = [f"DATA_{i}" for i in range(8)]
        all_ff = (df[data_cols] == 255).all(axis=1)

        # Attack = both conditions. This isolates the true injections.
        is_attack = in_window & all_ff

        df["true_class"] = "benign"
        df.loc[is_attack, "true_class"] = capture_name

        return df