from pathlib import Path

from PIL import Image

data_folder = Path.cwd() / "data"
data_folder.mkdir(parents=True, exist_ok=True)

purity_folders = {
    float(folder.name.split("_")[1]): folder for folder in data_folder.glob("purity_*")
}
purities = sorted(purity_folders.keys())
images = [Image.open(purity_folders[p] / "result.png") for p in purities]

width = images[0].width
height = images[0].height

dst = Image.new("RGB", (len(purities) * width, height))
for i, purity in enumerate(purities):
    w = i * width
    dst.paste(images[i], (w, 0))

dst.save(data_folder / "combined_result.png")

# all_rays.to_csv(data_folder / "all_rays.csv", columns=columns)
