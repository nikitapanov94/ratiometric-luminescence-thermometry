from flame_data_loader import load_folder


folder_input = input(
    "Enter path to folder containing spectral .txt files: "
).strip().strip('"')

all_data = load_folder(folder_input, max_rows=None)

print("\nDONE")
print("-" * 40)

print("Files loaded:", len(all_data))

if not all_data:
    print("No files found.")

else:

    first_name = next(iter(all_data))
    d = all_data[first_name]

    x = d["x"]
    y = d["y"]

    print("\nFirst file loaded:")
    print(first_name)

    print("\nArray info")
    print("x shape:", x.shape)
    print("y shape:", y.shape)

    print("\nFirst 3 rows:")
    for i in range(min(3, len(x))):
        print(f"{i}: wavelength = {x[i]:.3f}, intensity = {y[i]:.3f}")

    print("\nLast 3 rows:")
    for i in range(max(0, len(x) - 3), len(x)):
        print(f"{i}: wavelength = {x[i]:.3f}, intensity = {y[i]:.3f}")