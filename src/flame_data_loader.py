from pathlib import Path
import numpy as np


def load_folder(folder_path, max_rows=None):
    """
    Load all .txt spectra files in a folder.

    Returns
    -------
    dict
        {
            filename: {
                "x": np.ndarray,
                "y": np.ndarray
            }
        }
    """

    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    txt_files = sorted(folder.glob("*.txt"))

    all_data = {}

    for file in txt_files:
        all_data[file.name] = load_single_file(file, max_rows)

    return all_data


def load_single_file(txt_file, max_rows=None):
    """
    Load Ocean Optics spectral txt export with format:

    >>>>>Begin Spectral Data<<<<<
    wavelength    intensity
    """

    txt_file = Path(txt_file)

    lines = txt_file.read_text(errors="ignore").splitlines()

    # Find spectral data start
    data_start = None

    for i, line in enumerate(lines):
        if "Begin Spectral Data" in line:
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"Could not find spectral data in {txt_file.name}")

    x_vals = []
    y_vals = []

    for line in lines[data_start:]:

        s = line.strip()

        if not s:
            continue

        # split by whitespace or tab
        parts = s.split()

        if len(parts) < 2:
            continue

        try:
            x = float(parts[0])
            y = float(parts[1])

        except ValueError:
            continue

        x_vals.append(x)
        y_vals.append(y)

        if max_rows is not None and len(x_vals) >= max_rows:
            break

    return {
        "x": np.array(x_vals, dtype=float),
        "y": np.array(y_vals, dtype=float),
    }