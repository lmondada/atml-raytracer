from datetime import datetime
import progressbar

# save to `img_folder`
run_id = "testing"
from pathlib import Path
img_folder = Path.cwd() / "refractive_spheres" /run_id
img_folder.mkdir(parents=True, exist_ok=True)

# make sure we can import sightpy
# (this has to be run from the /data folder)
import sys
sys.path.insert(1, '../sightpy')
from sightpy import *


def render_image(refrac: float = 1.5):
    # Set Scene

    Sc = Scene(ambient_color=rgb(0.00, 0.00, 0.00))

    angle = -0

    # Sc.add_Camera(
    #     screen_width=100,
    #     screen_height=100,
    #     look_from=vec3(422, 78, 200),
    #     look_at=vec3(480, 50, 0),
    #     focal_distance=1.0,
    #     field_of_view=40,
    # )
    # Sc.add_Camera(screen_width = 100 ,screen_height = 100,
    # 			  look_from = vec3(278, 278, 800), look_at = vec3(278,278,0),
    # 			  focal_distance= 1., field_of_view= 40)
    Sc.add_Camera(
        screen_width=128,
        screen_height=128,
        look_from=vec3(40, 400, 300),
        look_at=vec3(500, 40, -500),
        focal_distance=1.0,
        field_of_view=40,
    )

    # define materials to use
    green_diffuse = Diffuse(diff_color=rgb(0.12, 0.45, 0.15))
    red_diffuse = Diffuse(diff_color=rgb(0.65, 0.05, 0.05))
    white_diffuse = Diffuse(diff_color=rgb(0.73, 0.73, 0.73))
    emissive_white = Emissive(color=rgb(15.0, 15.0, 15.0))
    emissive_blue = Emissive(color=rgb(2.0, 2.0, 3.5))
    blue_glass = Refractive(n=vec3(refrac + 0.05e-8j, refrac + 0.02e-8j, refrac + 0.0j))
    green_glass = Refractive(n=vec3(1.5 + 0.05e-8j, 1.5 + 0.0j, 1.5 + 0.02e-8j))

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
            material=green_diffuse,
            center=vec3(0, 555 / 2, -555 / 2),
            width=555.0,
            height=555.0,
            u_axis=vec3(0.0, 1.0, 0),
            v_axis=vec3(0.0, 0, -1.0),
        )
    )

    Sc.add(
        Plane(
            material=red_diffuse,
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

    Sc.add(
        Sphere(
            material=green_glass,
            center=vec3(400.5, 100, -70),
            radius=100,
            shadow=False,
            max_ray_depth=3,
            # mc=True,
        ),
        importance_sampled=True,
    )

    Sc.add(
        Sphere(
            material=blue_glass,
            center=vec3(180.5, 100, -140),
            radius=100,
            shadow=False,
            max_ray_depth=3,
            # mc=True,
        ),
        importance_sampled=True,
    )
    Sc.add(
        Sphere(
            material=blue_glass,
            center=vec3(350.5, 160, -300),
            radius=140,
            shadow=False,
            max_ray_depth=3,
            # mc=True,
        ),
        importance_sampled=True,
    )

    # Render
    img, gold_bars = Sc.render(samples_per_pixel=1, progress_bar=False)#, batch_size=4)
    return img, gold_bars


if __name__ == "__main__":
    refrac_array = np.arange(1., 1.6, 0.05)
    bar = progressbar.ProgressBar(maxval=len(refrac_array))
    bar.start()
    start = datetime.now()
    for i, refrac in enumerate(refrac_array):
        bar.update(i)
        img, gold_bars = render_image(refrac)
        print(gold_bars)
        print(gold_bars.shape)
        img.save(img_folder / f"refrac_{refrac:.2f}.png")
    end = datetime.now()
    print(f"Elapsed: {str(end - start).split('.')[0]}")
    bar.finish()