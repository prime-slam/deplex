## Depth Plane Extraction
Or simply deplex, is an open-source cross-platform C++ library specifically designed for highly optimized plane extraction from RGB-D images. The library also comes with prebuilt Python package.

Plane segmentation algorithm is based on the article:
*P. Proenca and Y. Gao, Fast Cylinder and Plane Extraction from Depth Cameras for Visual Odometry, IROS, 2018*
## External dependencies
Technically you don't need any! Library depends only on Eigen3 for math operations and fetches it automatically with CMake scripts.
## Python package
We also provide Python bindings for building cross-platfrom Python wheels. As well as main library, package depends only on numpy.
## Usage
```python
from deplex import PlaneExtractor, Config
from deplex.utils import DepthImage

camera_intrinsics = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
image = DepthImage("sample_depth.png")
config = Config("custom_config.ini")
runner = PlaneExtractor(image.height, image.width, config=config)  
labels = runner.process(image.transform_to_pcd(camera_intrinsics))
```
## Installation
### Building from source
To build a library from source you need C++14 compatible compiler and CMake 3.18+
```bash
cmake -B build -<options>
cmake --build build
```
See CMakeLists.txt to find available options.
### Python
Our library is available as lightweight Python-wheel package.
Simply install it with pip.
```bash
python3 -m pip install deplex
```
Supported Python versions: 3.6-3.10
Supported platfrom: Linux, macOS and Windows

## License
Apache License, Version 2.0