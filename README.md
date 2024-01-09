# Timber Log Analysis

Automate wood log analysis using YOLOv8 and YOLOv5.

## Purpose
The purpose of this project is to automate the wood log analysis and minimise the human intervention.

## Output
The output that we'll be getting in a CSV file. 

## Dependencies
Make sure you have the following dependencies installed before running the code:
- [Ultralytics](https://github.com/ultralytics/yolov5)
- [NumPy](https://numpy.org/)
- [OpenCV](https://opencv.org/)
- [Pillow](https://pillow.readthedocs.io/)
- [scikit-image](https://scikit-image.org/)
- [PyTorch](https://pytorch.org/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [YoloV5 and YoloV8](https://github.com/ultralytics/yolov5) (Python code)
- [skimage](https://scikit-image.org/)
- [shutil](https://docs.python.org/3/library/shutil.html) (included in Python's standard library)

All required files are included in the repository.

## Usage

### Code Modifications
If not running in Google Colab, make the following modifications in the code:
- Adjust file paths as needed.

### Features
1. **Total Number of Logs**
2. **Log Diameter Analysis**
   - Set prices based on log diameter.
   - Maintain counts for different diameter ranges:
     - Logs with circumference < 12 inches
     - Logs with 12 to 38 inches circumference
     - Logs with circumference > 38 inches
3. **Crack Detection**
   - Identify logs with cracks.
4. **Wood Log Types**
   - Categorize logs based on their types.

## Contribution
Contributions are welcome! Check out the [Contribution Guidelines](https://github.com/Hridaydeep).

## Future Plans
We plan to add the following feature:
- **Identify Log Shape**
  - Distinguish between rounded and square logs.
  - Note: Square logs may require additional reshaping, leading to potential wastage.

## Technology Used
- YOLOv8
- YOLOv5
- Python
- scikit-learn
- scikit-image

## Author
- [Hridaydeep](https://github.com/Hridaydeep)
