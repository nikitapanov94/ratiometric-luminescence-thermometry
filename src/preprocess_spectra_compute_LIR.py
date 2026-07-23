from pathlib import Path
import numpy as np

from flame_data_loader import load_folder


# ============================================================
# USER SETTINGS
# ============================================================

# Number of spectra to process
# Set to None to process ALL files
MAX_FILES = 10

# Cosmic ray removal
COSMIC_WINDOW = 7          # must be odd
COSMIC_THRESHOLD = 6.0     # larger = more conservative

# Baseline region
BASELINE_START_ROW = 1
BASELINE_END_ROW = 448

# Integration regions
L2_START_ROW = 2106
L2_END_ROW = 2457

L1_START_ROW = 2457
L1_END_ROW = 2916

# Export corrected spectra
EXPORT_CORRECTED_SPECTRA = True


# ============================================================
# COSMIC RAY REMOVAL
# ============================================================

def remove_cosmic_rays(y,
                       window_size=COSMIC_WINDOW,
                       threshold=COSMIC_THRESHOLD):
    """
    Remove positive and negative cosmic rays using
    local median + MAD filtering.

    Returns
    -------
    y_clean : np.ndarray
    n_corrected : int
    """

    y = np.asarray(y, dtype=float).copy()

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    half = window_size // 2

    n_corrected = 0

    y_clean = y.copy()

    for i in range(half, len(y) - half):

        window = y[i - half:i + half + 1]

        # exclude center point
        neighbors = np.delete(window, half)

        local_median = np.median(neighbors)

        mad = np.median(np.abs(neighbors - local_median))

        # avoid division issues
        if mad <= 0:
            continue

        deviation = abs(y[i] - local_median)

        if deviation > threshold * mad:

            # replace cosmic ray by local median
            y_clean[i] = local_median

            n_corrected += 1

    return y_clean, n_corrected


# ============================================================
# BASELINE CORRECTION
# ============================================================

def baseline_correct(y):
    """
    Baseline correction using mean intensity
    between specified rows.
    """

    start = BASELINE_START_ROW - 1
    end = BASELINE_END_ROW

    baseline = np.mean(y[start:end])

    y_corr = y - baseline

    return y_corr, baseline


# ============================================================
# INTEGRATION
# ============================================================

def integrate_region(x, y, start_row, end_row):
    """
    Integrate intensity using trapezoidal integration.
    """

    start = start_row - 1
    end = end_row

    x_region = x[start:end]
    y_region = y[start:end]

    area = np.trapz(y_region, x_region)

    return float(area)


# ============================================================
# EXPORT CORRECTED SPECTRUM
# ============================================================

def export_corrected_spectrum(export_folder,
                              fname,
                              x,
                              y_corr):

    export_folder = Path(export_folder)

    base_name = Path(fname).stem

    out_path = export_folder / f"{base_name}_corrected.txt"

    data = np.column_stack([x, y_corr])

    np.savetxt(
        out_path,
        data,
        delimiter="\t",
        header="wavelength_nm\tcorrected_intensity",
        comments="",
        fmt="%.8f"
    )


# ============================================================
# SUMMARY EXPORT
# ============================================================

def export_summary(export_path,
                   measurement_condition,
                   L1_values,
                   L2_values,
                   LIR_values):

    L1_values = np.asarray(L1_values, dtype=float)
    L2_values = np.asarray(L2_values, dtype=float)
    LIR_values = np.asarray(LIR_values, dtype=float)

    headers = [
        "Measurement condition",
        "L1 mean",
        "L1 s.d.",
        "L2 mean",
        "L2 s.d.",
        "LIR mean",
        "LIR s.d.",
        "N spectra"
    ]

    row = [
        measurement_condition,
        f"{np.mean(L1_values):.8f}",
        f"{np.std(L1_values, ddof=1):.8f}",
        f"{np.mean(L2_values):.8f}",
        f"{np.std(L2_values, ddof=1):.8f}",
        f"{np.mean(LIR_values):.8f}",
        f"{np.std(LIR_values, ddof=1):.8f}",
        str(len(LIR_values))
    ]

    with open(export_path, "w", encoding="utf-8") as f:

        f.write("\t".join(headers) + "\n")
        f.write("\t".join(row) + "\n")


# ============================================================
# MAIN
# ============================================================

def main():

    # ----------------------------
    # INPUTS
    # ----------------------------

    folder_input = input(
        "Enter folder containing spectra: "
    ).strip().strip('"').strip("'")

    export_input = input(
        "Enter export folder: "
    ).strip().strip('"').strip("'")

    measurement_condition = input(
        "Measurement condition (example: 175 K): "
    ).strip()

    export_folder = Path(export_input)
    export_folder.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # LOAD DATA
    # ----------------------------

    all_data = load_folder(folder_input)

    # ----------------------------
    # STORAGE
    # ----------------------------

    L1_values = []
    L2_values = []
    LIR_values = []

    n_processed = 0

    # ----------------------------
    # PROCESS LOOP
    # ----------------------------

    for i, (fname, d) in enumerate(all_data.items()):

        if MAX_FILES is not None and i >= MAX_FILES:
            break

        print("\n--------------------------------")
        print(f"Processing: {fname}")

        x = np.asarray(d["x"], dtype=float)
        y = np.asarray(d["y"], dtype=float)

        # ----------------------------
        # COSMIC RAY REMOVAL
        # ----------------------------

        y_clean, n_cosmics = remove_cosmic_rays(y)

        print(f"Cosmic rays corrected: {n_cosmics}")

        # ----------------------------
        # BASELINE CORRECTION
        # ----------------------------

        y_corr, baseline = baseline_correct(y_clean)

        print(f"Baseline: {baseline:.6f}")

        # ----------------------------
        # INTEGRATION
        # ----------------------------

        L2 = integrate_region(
            x,
            y_corr,
            L2_START_ROW,
            L2_END_ROW
        )

        L1 = integrate_region(
            x,
            y_corr,
            L1_START_ROW,
            L1_END_ROW
        )

        # ----------------------------
        # LIR
        # ----------------------------

        if L1 != 0:
            LIR = L2 / L1
        else:
            LIR = np.nan

        print(f"L2 = {L2:.6f}")
        print(f"L1 = {L1:.6f}")
        print(f"LIR = {LIR:.6f}")

        # ----------------------------
        # STORE RESULTS
        # ----------------------------

        L1_values.append(L1)
        L2_values.append(L2)
        LIR_values.append(LIR)

        # ----------------------------
        # EXPORT CORRECTED SPECTRUM
        # ----------------------------

        if EXPORT_CORRECTED_SPECTRA:

            export_corrected_spectrum(
                export_folder,
                fname,
                x,
                y_corr
            )

        n_processed += 1

    # ============================================================
    # SUMMARY EXPORT
    # ============================================================

    safe_condition = measurement_condition.replace(" ", "_")
    summary_path = export_folder / f"Summary_{safe_condition}.txt"

    export_summary(
        summary_path,
        measurement_condition,
        L1_values,
        L2_values,
        LIR_values
    )

    # ============================================================
    # FINAL PRINT
    # ============================================================

    print("\n================================")
    print("DONE")
    print("================================")

    print(f"Spectra processed: {n_processed}")

    print("\nFolder statistics:")

    print(f"L1 mean = {np.mean(L1_values):.8f}")
    print(f"L1 s.d. = {np.std(L1_values, ddof=1):.8f}")

    print(f"L2 mean = {np.mean(L2_values):.8f}")
    print(f"L2 s.d. = {np.std(L2_values, ddof=1):.8f}")

    print(f"LIR mean = {np.mean(LIR_values):.8f}")
    print(f"LIR s.d. = {np.std(LIR_values, ddof=1):.8f}")

    print("\nSummary saved to:")
    print(summary_path)


if __name__ == "__main__":
    main()