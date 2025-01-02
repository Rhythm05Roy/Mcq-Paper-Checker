# OMR Grading System

This project is a web-based Optical Mark Recognition (OMR) grading system built with Streamlit. The application processes OMR sheets uploaded as images and evaluates the results based on predefined answers.

## Features

- **Image Upload:** Upload OMR sheet images in JPG or PNG format.
- **Image Processing:** Automatically detects and processes the OMR sheet using OpenCV.
- **Grading Logic:** Compares detected answers with a predefined answer key and calculates the score.
- **Interactive Interface:** Displays the uploaded image and grading results.

## Technologies Used

- **Python:** Core programming language.
- **Streamlit:** Framework for building the web application.
- **OpenCV:** For image processing and contour detection.
- **NumPy:** For efficient numerical operations.

## Setup and Installation

1. Clone the repository:
   ```bash
   https://github.com/Rhythm05Roy/Mcq-Paper-Checler.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Mcq-Paper-Checler
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## How It Works

1. Upload an image of the OMR sheet via the file uploader in the web app.
2. The system processes the uploaded image:
   - Resizes the image for uniformity.
   - Converts the image to grayscale, applies Gaussian blur, and detects edges.
   - Identifies the largest contour (assumed to be the OMR sheet).
   - Warps the perspective to align the sheet correctly.
   - Splits the sheet into individual answer boxes.
3. Each box is analyzed for marked answers.
4. The detected answers are compared with the predefined answer key, and a score is calculated.
5. The grading result is displayed.

## Answer Key

The predefined answers for the OMR sheet are as follows:

| Question | Correct Option |
|----------|----------------|
| 1        | 1              |
| 2        | 2              |
| 3        | 0              |
| 4        | 1              |
| 5        | 4              |

## Utility Functions

The `utils.py` file contains the following helper functions:

- `rectContour(contours)`: Identifies rectangular contours.
- `getCornerPoints(contour)`: Finds the corner points of a contour.
- `splitBoxes(image)`: Splits the OMR sheet into individual answer boxes.

Ensure that the `utils.py` file is in the same directory as the main script.

## Dependencies

The required libraries are listed in `requirements.txt`. Key dependencies include:

- `streamlit`
- `opencv-python`
- `numpy`

## Example Usage

1. Launch the app and upload an image of the OMR sheet.
2. View the uploaded image and the calculated grading score.

## Error Handling

- If the system encounters any issues during image processing, an error message is displayed.

## Future Enhancements

- Add support for more flexible answer key formats.
- Enhance error detection and handling for better reliability.
- Extend support for OMR sheets with varying layouts.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- OpenCV for robust image processing tools.
- Streamlit for the user-friendly web interface.

