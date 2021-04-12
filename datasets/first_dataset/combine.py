from pathlib import Path

import pandas as pd

data_folder = Path.cwd() / "data"
data_folder.mkdir(parents=True, exist_ok=True)

purity_folders = {
    float(folder.name.split("_")[1]): folder for folder in data_folder.glob("purity_*")
}

columns = ["pixel_index", "color", "purity", "log_p_z", "log_p_z_ref"]
all_rays = pd.DataFrame(columns=columns)

needs_header = True
for purity, folder in purity_folders.items():
    print((folder / "all_rays.csv").relative_to(Path.cwd()), end="...\n", flush=True)
    new_rays = pd.read_csv(folder / "all_rays.csv")
    new_rays["purity"] = purity
    new_rays.to_csv(
        data_folder / "all_rays.csv",
        columns=columns,
        mode="a",
        header=needs_header,
        index=False,
    )
    needs_header = False

# all_rays.to_csv(data_folder / "all_rays.csv", columns=columns)
