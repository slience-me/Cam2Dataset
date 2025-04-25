# Image Collection Tool

This project is a Python-based image collection tool with data augmentation and graphical user interface (GUI) for easily collecting and saving images for machine learning tasks.

## Features

- **Real-time Image Capture:** Captures images from the camera and supports various data augmentations (flip, rotation, crop, brightness adjustment).
- **GUI Interface:** Easy-to-use interface for controlling image capture, selecting the save directory, and displaying real-time statistics.
- **Data Augmentation:** Automatically applies several augmentations (flip, rotate, crop, etc.) to each captured image.
- **Training Data Preparation:** Saves the augmented images into a training folder with file name mapping.
- **Real-time Image Count Update:** Displays the real-time count of images being captured and augmented.
- **Customizable Directories:** Allows the user to choose where to save images.

## 📹 Video Demo

See [video:demo.mp4](./docs/demo.mp4) for a demonstration of how the image collector works.

## Requirements

- Python 3.x
- OpenCV
- Tkinter
- colorlog
- numpy

You can install the necessary libraries using the following:

```bash
pip install opencv-python tkinter colorlog numpy
```

## Usage

1. **Run the GUI:**

   The GUI can be launched by running the script below. It will allow you to configure the directory and specify the number of images you want to collect.

   ```bash
   python image_collection_tool.py
   ```

2. **Choose the Save Directory:**

   By default, images are saved in the `./collected_images` directory. You can select a different directory through the "选择保存路径" (Choose Save Path) button.

3. **Configure Image Collection:**

   - Input the number of images you want to collect (this number refers to the number of original images; each original image will result in 6 augmented images).
   - Click "启动采集" (Start Collection) to start the process.

4. **View Results:**

   - The original images are saved in the `original` directory.
   - The augmented images are saved in the `augmented` directory.
   - A mapping of original and augmented images is saved in `train_file_mapping.txt`.

5. **Stop the Collection:**

   You can stop the image collection process at any time by clicking the "停止采集" (Stop Collection) button.

## File Structure

```
collected_images/
│
├── original/       # Folder for storing original images
├── augmented/      # Folder for storing augmented images
└── images/
    └── train/      # Folder for storing training images (augmented)
```

## System Information Display

When the tool starts, it will display system information such as:

- Platform information (OS version, architecture)
- Python version
- Output directory paths

## Logging

The tool uses color-coded logging to provide clear feedback in the console:

- **DEBUG**: Blue
- **INFO**: Green
- **WARNING**: Yellow
- **ERROR**: Red
- **CRITICAL**: Bold Red

Logs provide information on system settings, the image collection process, and any errors encountered.

## Example Output

Upon starting the image collection process, you will see logs similar to the following:

```
2025-04-25 12:34:56 - INFO - System Information
2025-04-25 12:34:56 - INFO - Platform: Windows 10 10.0.18363
2025-04-25 12:34:56 - INFO - Python Version: 3.9.5
2025-04-25 12:34:56 - INFO - Output Directory: collected_images
2025-04-25 12:34:56 - INFO -   - Original Images: collected_images/original
2025-04-25 12:34:56 - INFO -   - Augmented Images: collected_images/augmented
...
```

## Contributing

Feel free to fork the repository and submit pull requests for improvements or bug fixes. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This tool is designed for ease of use in collecting and augmenting images for training machine learning models, simplifying the image collection and preprocessing pipeline.
