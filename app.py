import streamlit as st
import cv2
import numpy as np
import utils  # Ensure utils.py is accessible

st.title("OMR Grading System")

# File uploader
uploaded_file = st.file_uploader("Upload an image for OMR processing", type=["jpg", "png"])

if uploaded_file:
    # Read the uploaded file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display the uploaded image
    st.image(img, channels="BGR", caption="Uploaded Image")

    # Processing
    try:
        height, width = 500, 500
        questions, choices = 5, 5
        ans = [1, 2, 0, 1, 4]

        img_resized = cv2.resize(img, (width, height))
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        img_canny = cv2.Canny(img_blur, 50, 50)

        contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rect_con = utils.rectContour(contours)

        if rect_con:
            biggest_contour = utils.getCornerPoints(rect_con[0])
            if biggest_contour.size != 0:
                # Perspective transform and grading logic
                pt1 = np.float32(biggest_contour)
                pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
                matrix = cv2.getPerspectiveTransform(pt1, pt2)
                img_warp_colored = cv2.warpPerspective(img, matrix, (width, height))

                img_thresh = cv2.threshold(cv2.cvtColor(img_warp_colored, cv2.COLOR_BGR2GRAY), 190, 255, cv2.THRESH_BINARY_INV)[1]
                boxes = utils.splitBoxes(img_thresh)

                # Process answers
                my_pixel_val = np.zeros((questions, choices))
                for count_row, row in enumerate(boxes):
                    for count_col, box in enumerate(row):
                        my_pixel_val[count_row][count_col] = cv2.countNonZero(box)

                my_index = [np.argmax(row) for row in my_pixel_val]
                grading = [1 if ans[i] == my_index[i] else 0 for i in range(questions)]
                score = sum(grading) / questions * 100

                # Display results
                st.write(f"Grading Score: {score:.2f}%")
    except Exception as e:
        st.error(f"Error processing the image: {e}")
