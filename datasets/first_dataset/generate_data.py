# save to `data_folder`
from pathlib import Path

data_folder = Path.cwd() / "data"
data_folder.mkdir(parents=True, exist_ok=True)

# make sure we can import sightpy
# (this has to be run from the /data folder)
import sys

sys.path.insert(1, "../../raytracer")
from sightpy import *


def get_all_rays(purity, data_folder):
    # Set Scene
    Sc = Scene(ambient_color=rgb(0.00, 0.00, 0.00))
    Sc.add_Camera(
        screen_width=128,
        screen_height=128,
        look_from=vec3(40, 400, 300),
        look_at=vec3(500, 0, -500),
        focal_distance=1.0,
        field_of_view=30,
    )

    # define materials to use
    gray_diffuse = Diffuse(
        diff_color=rgb(0.43, 0.43, 0.43), diff_color_ref=rgb(0.43, 0.43, 0.43)
    )
    white_diffuse = Diffuse(
        diff_color=rgb(0.73, 0.73, 0.73), diff_color_ref=rgb(0.73, 0.73, 0.73)
    )
    emissive_white = Emissive(color=rgb(0.9, 0.9, 0.9))
    glass = Refractive(
        n=vec3(1.5 + 0j, 1.5 + 0j, 1.5 + 0j), purity=purity, purity_ref=0.75
    )

    # this is the light
    Sc.add(
        Plane(
            material=emissive_white,
            center=vec3(213 + 130 / 2, 400, -227.0 - 105 / 2),
            width=130.0,
            height=105.0,
            u_axis=vec3(1.0, 0.0, 0),
            v_axis=vec3(0.0, 0, 1.0),
        ),
        importance_sampled=True,
    )
    Sc.add(
        Plane(
            material=emissive_white,
            center=vec3(5, 225, -227.0),
            width=130.0,
            height=105.0,
            u_axis=vec3(0.0, 1.0, 0),
            v_axis=vec3(0.0, 0, 1.0),
        ),
        importance_sampled=True,
    )

    Sc.add(
        Plane(
            material=white_diffuse,
            center=vec3(555 / 2, 555 / 2, -555.0),
            width=555.0,
            height=555.0,
            u_axis=vec3(0.0, 1.0, 0),
            v_axis=vec3(1.0, 0, 0.0),
        )
    )

    Sc.add(
        Plane(
            material=gray_diffuse,
            center=vec3(0, 555 / 2, -555 / 2),
            width=555.0,
            height=555.0,
            u_axis=vec3(0.0, 1.0, 0),
            v_axis=vec3(0.0, 0, -1.0),
        )
    )

    Sc.add(
        Plane(
            material=gray_diffuse,
            center=vec3(555.0, 555 / 2, -555 / 2),
            width=555.0,
            height=555.0,
            u_axis=vec3(0.0, 1.0, 0),
            v_axis=vec3(0.0, 0, -1.0),
        )
    )

    Sc.add(
        Plane(
            material=white_diffuse,
            center=vec3(555 / 2, 555, -555 / 2),
            width=555.0,
            height=555.0,
            u_axis=vec3(1.0, 0.0, 0),
            v_axis=vec3(0.0, 0, -1.0),
        )
    )

    Sc.add(
        Plane(
            material=white_diffuse,
            center=vec3(555 / 2, 0.0, -555 / 2),
            width=555.0,
            height=555.0,
            u_axis=vec3(1.0, 0.0, 0),
            v_axis=vec3(0.0, 0, -1.0),
        )
    )

    # Sc.add(
    #     Sphere(
    #         material=glass,
    #         center=vec3(400.5, 100, -70),
    #         radius=100,
    #         shadow=False,
    #         max_ray_depth=3,
    #         # mc=True,
    #     ),
    #     importance_sampled=True,
    # )

    Sc.add(
        Sphere(
            material=glass,
            center=vec3(320.5, 100, -240),
            radius=100,
            shadow=False,
            max_ray_depth=2,
            # mc=True,
        ),
        importance_sampled=True,
    )

    # Sc.add(
    #     Sphere(
    #         material=glass,
    #         center=vec3(350.5, 160, -300),
    #         radius=140,
    #         shadow=False,
    #         max_ray_depth=3,
    #         # mc=True,
    #     ),
    #     importance_sampled=True,
    # )

    # Render
    img, gold_bars = Sc.render(
        samples_per_pixel=32, progress_bar=True, save_csv=data_folder / "all_rays.csv"
    )
    with open(data_folder / "scores.csv", "w", newline="") as csvfile:
        fieldnames = ["joint score", "reference joint score", "joint likelihood ratio"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(
            {
                "joint score": gold_bars[0],
                "reference joint score": gold_bars[1],
                "joint likelihood ratio": gold_bars[2],
            }
        )

    img.save(data_folder / "result.png")
    img.show()


if __name__ == "__main__":
    for purity in np.arange(0.5, 0.99, 0.05):
        current_folder = data_folder / f"purity_{purity:.2f}"
        current_folder.mkdir(parents=True, exist_ok=True)
        get_all_rays(purity=purity, data_folder=data_folder / f"purity_{purity:.2f}")
