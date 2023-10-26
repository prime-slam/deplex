from deplex.utils import DepthImage
import deplex

import timeit
import numpy as np

from pathlib import Path

data_dir = Path(__file__).parent.parent.parent.resolve() / "benchmark/data"
image_path = data_dir / Path("depth") / Path("000004415622.png")
config_path = data_dir / Path("config") / Path("TUM_fr3_long_val.ini")
intrinsics_path = data_dir / Path("config") / Path("intrinsics.K")


def benchmarkProcessCloud():
    NUMBER_OF_RUNS = 10

    execution_time = []

    config = deplex.Config(str(config_path))
    camera_intrinsic = np.genfromtxt(intrinsics_path)
    image = DepthImage(str(image_path))

    print("Image Height:", image.height, "Image Width:", image.width, '\n')

    coarse_algorithm = deplex.PlaneExtractor(image_height=image.height, image_width=image.width, config=config)

    for i in range(NUMBER_OF_RUNS):
        start_time = timeit.default_timer()

        pcd_points = image.transform_to_pcd(camera_intrinsic)

        labels = coarse_algorithm.process(pcd_points)

        end_time = timeit.default_timer()
        execution_time.append((end_time - start_time) * 1000)
        print(f'Iteration #{i + 1} Planes found: {max(labels)}')

    elapsed_time_min = min(execution_time)
    elapsed_time_max = max(execution_time)
    elapsed_time_mean = np.average(execution_time)

    dispersion = np.var(execution_time)
    standard_deviation = np.std(execution_time)
    standard_error = standard_deviation / np.sqrt(NUMBER_OF_RUNS)

    # 95% confidence interval
    t_value = 1.96
    lower_bound = elapsed_time_mean - t_value * standard_error
    upper_bound = elapsed_time_mean + t_value * standard_error

    print('\nDispersion:', f"{dispersion:.5f}")
    print('Standard deviation:', f"{standard_deviation:.5f}")
    print('Standard error:', f"{standard_error:.5f}")
    print('Confidence interval (95%):', f"[{lower_bound:.5f};", f"{upper_bound:.5f}]\n")

    print('Elapsed time (ms.) (min):', f"{elapsed_time_min:.5f}")
    print('Elapsed time (ms.) (max):', f"{elapsed_time_max:.5f}")
    print('Elapsed time (ms.) (mean):', f"{elapsed_time_mean:.5f}")
    print('FPS (max):', f"{1000 / elapsed_time_min:.5f}")
    print('FPS (min):', f"{1000 / elapsed_time_max:.5f}")
    print('FPS (mean):', f"{1000 / elapsed_time_mean:.5f}")


if __name__ == '__main__':
    benchmarkProcessCloud()
