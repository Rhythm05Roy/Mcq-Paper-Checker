import cv2
import numpy as np
import utils

# Variable Declearation
path = '1.jpg'
height = 500
width = 500
questions = 5
choices = 5
ans = [1,2,0,1,4]
webCamFeed = True

url = 'http://192.168.0.1.103.8080'
cap = cv2.VideoCapture(1)
cap.set(10,150)

while True:
    if webCamFeed:
        success, img = cap.read()
        # img = cv2.flip(img, 1)
        
    else:
        img = cv2.imread(path)

    try:
    # Pre-processing
        img = cv2.imread(path)
        img = cv2.resize(img, (width, width))
        imgContours = img.copy()
        imgbiggestContours = img.copy()
        imgFinal = img.copy()
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

            ## Appy threshold
            imagewarpGray = cv2.cvtColor(imgwapColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imagewarpGray, 190, 255, cv2.THRESH_BINARY_INV)[1]

            boxes = utils.splitBoxes(imgThresh)
            # print(cv2.countNonZero(boxes[1]),cv2.countNonZero(boxes[2]))

            # Getting Non-zero pixel value of each box
            myPixelVal = np.zeros((questions, choices))
            countCol = 0
            countRow = 0

            for image in boxes:
                total = cv2.countNonZero(image)
                myPixelVal[countRow][countCol] = total
                countCol += 1
                if countCol == choices:
                    countCol = 0
                    countRow += 1

        ## Finding actual values of the markings
            myIndex = []
            for x in range (0,questions):
                arr = myPixelVal[x]
                # print('arr',arr)
                myIndexVal = np.where(arr == np.max(arr))
                # print(myIndexVal[0])
                myIndex.append(int(myIndexVal[0][0]))
        # print(myIndex)

        ## Grading logic
            graddings = []
            for x in range (0,questions):
                if ans[x] == myIndex[x]:
                    graddings.append(1)
                else:
                    graddings.append(0)
            #print(graddings)
            
            score = sum(graddings)/questions*100
            #print(score)


        ## Displaying Grading
            imgresult = imgwapColored.copy()
            imgresult = utils.showAnswers(imgresult, myIndex, graddings,ans,questions,choices)
            imgRawDrawing  = np.zeros_like(imgwapColored)
            imgRawDrawing = utils.showAnswers(imgRawDrawing, myIndex, graddings,ans,questions,choices)
            invmatrix = cv2.getPerspectiveTransform(pt2, pt1)
            imgInvWap = cv2.warpPerspective(imgRawDrawing, invmatrix, (width, height))
            
            imgRawGrade = np.zeros_like(imgwapColored)
            cv2.putText(imgRawGrade,str(int(score))+'%',(90, 260),cv2.FONT_HERSHEY_COMPLEX,6,(0,255,0),6)
            Invmatrixg = cv2.getPerspectiveTransform(ptg2, ptg1)
            imgInvGradeDisplay  = cv2.warpPerspective(imgRawGrade, Invmatrixg, (width, height))


            imgFinal = cv2.addWeighted(imgFinal, 0.5, imgInvWap, 0.5, 0)
            imgFinal = cv2.addWeighted(imgFinal, 0.5, imgInvGradeDisplay, 0.5, 0)
            
                




        img_blank = np.zeros_like(img)
        img_array = ([img, img_gray, img_blur, img_canny],
                    [imgContours, imgbiggestContours, imgwapColored, imgThresh],
                    [imgresult, imgRawDrawing, imgInvWap, imgFinal]
                )
    except:
        img_blank = np.zeros_like(img)
        img_array = ([img_blank, img_blank, img_blank, img_blank],
                    [img_blank, img_blank, img_blank, img_blank],
                    [img_blank, img_blank, img_blank, img_blank]
                )
    labels = [['Original', 'Gray', 'Blur', 'Canny'],
            ['Contours', 'Biggest Contour', 'Warp Perspective', 'Threshold'],
            ['Result', 'Raw Drawing', 'Inverse Warp', 'Final']]
    imgStacked = utils.stackImages(img_array, 0.5,labels)

    cv2.imshow('Final Image', imgFinal)
    cv2.imshow('Stacked Images', imgStacked)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('FinalImage.jpg',imgFinal)
        cv2.waitKey(0)
    elif cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()