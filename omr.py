import cv2
import numpy as np
import utils

# Variable Declearation
path = '2-Photoroom.png'
height = 700
width = 700

# Pre-processing
img = cv2.imread(path)
img = cv2.resize(img, (width, width))
imgContours = img.copy()
imgbiggestContours = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
img_canny = cv2.Canny(img_blur, 50, 50)

# Contours
contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (255, 0, 0), 3)

# Rectangular Contours
rectCon = utils.rectContour(contours)
biggestContour = utils.getCornerPoints(rectCon[0])
gradepoints = utils.getCornerPoints(rectCon[1])

# print(biggestContour)
# print(gradepoints)

if biggestContour.size != 0 and gradepoints.size != 0:
    cv2.drawContours(imgbiggestContours, biggestContour, -1, (0, 255, 0), 20)
    cv2.drawContours(imgbiggestContours, gradepoints, -1, (151, 81, 68), 20)
    biggestContour = utils.reorder(biggestContour)
    gradepoints = utils.reorder(gradepoints)

    ## For answer selection
    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])    
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgwapColored  = cv2.warpPerspective(img, matrix, (width, height))

    ## For grading
    ptg1 = np.float32(gradepoints)
    ptg2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])    
    matrixg = cv2.getPerspectiveTransform(ptg1, ptg2)
    imageGradeDisplay  = cv2.warpPerspective(img, matrixg, (width, height))

img_blank = np.zeros_like(img)
img_array = ([img, img_gray, img_blur, img_canny],
             [imgContours, imgbiggestContours, imgwapColored, imageGradeDisplay])
imgStacked = utils.stackImages(img_array, 0.5)

cv2.imshow('Stacked Images', imgStacked)
cv2.waitKey(0)
cv2.destroyAllWindows