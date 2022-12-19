## Depth Plane Extraction
Or simply deplex, is a Python package with highly-optimized C++ plane extraction algorithm.

The algorithm is based on the article:
*P. Proenca and Y. Gao, Fast Cylinder and Plane Extraction from Depth Cameras for Visual Odometry, IROS, 2018*

## Usage
```python
from deplex import PlaneExtractor

runner = PlaneExtractor(image_height, image_width)  
labels = runner.process(image_points)
```
## Installation
### Python
Our library is available as lightweight Python-wheel package.
Simply install it with pip.
```bash
python3 -m pip install deplex
```
Supported Python versions: 3.6-3.10
Supported platfrom: Linux (Mac and Windows coming soon...)

## License
Apache License, Version 2.0