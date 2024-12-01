import streamlit as st
import cv2
import numpy as np
import utils
import tempfile
import os

def process_omr_sheet(img, questions, choices, ans):
    """
    Process the OMR sheet and return results
    """
    try:
        # Preprocessing
        width = 500
        height = 500
        img = cv2.resize(img, (width, width))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        img_canny = cv2.Canny(img_blur, 50, 50)

        # Contours
        contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Rectangular Contours
        rectCon = utils.rectContour(contours)
        biggestContour = utils.getCornerPoints(rectCon[0])
        gradepoints = utils.getCornerPoints(rectCon[1])

        if biggestContour.size != 0 and gradepoints.size != 0:
            biggestContour = utils.reorder(biggestContour)
            gradepoints = utils.reorder(gradepoints)

            # Perspective Transform for Answer Sheet
            pt1 = np.float32(biggestContour)
            pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])    
            matrix = cv2.getPerspectiveTransform(pt1, pt2)
            imgwapColored = cv2.warpPerspective(img, matrix, (width, height))

            # Threshold
            imagewarpGray = cv2.cvtColor(imgwapColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imagewarpGray, 190, 255, cv2.THRESH_BINARY_INV)[1]

            # Split boxes and count pixels
            boxes = utils.splitBoxes(imgThresh)
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

            # Find marked answers
            myIndex = []
            for x in range(questions):
                arr = myPixelVal[x]
                myIndexVal = np.where(arr == np.max(arr))
                myIndex.append(int(myIndexVal[0][0]))

            # Grading
            graddings = []
            for x in range(questions):
                graddings.append(1 if ans[x] == myIndex[x] else 0)
            
            score = sum(graddings) / questions * 100

            return {
                'marked_answers': myIndex,
                'correct_answers': ans,
                'score': score,
                'graddings': graddings
            }
        
        return None

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def main():
    st.title('OMR Sheet Evaluator')

    # Sidebar for configuration
    st.sidebar.header('OMR Sheet Configuration')
    
    # Number of questions and choices
    questions = st.sidebar.number_input('Number of Questions', min_value=1, max_value=20, value=5)
    choices = st.sidebar.number_input('Number of Choices', min_value=2, max_value=10, value=5)

    # Answer key input
    ans = []
    st.sidebar.subheader('Answer Key')
    for i in range(questions):
        ans.append(st.sidebar.selectbox(f'Question {i+1}', list(range(choices)), index=0))

    # File uploader
    uploaded_file = st.file_uploader("Upload OMR Sheet", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        # Read the image
        img = cv2.imread(temp_file_path)

        # Process the image
        result = process_omr_sheet(img, questions, choices, ans)

        # Display results
        if result:
            st.subheader('Results')
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric('Score', f'{result["score"]:.2f}%')
            
            with col2:
                st.metric('Questions Correct', f'{sum(result["graddings"])} / {questions}')

            # Detailed results
            st.subheader('Detailed Answers')
            results_df = []
            for q in range(questions):
                results_df.append({
                    'Question': q+1, 
                    'Marked Answer': result['marked_answers'][q]+1, 
                    'Correct Answer': result['correct_answers'][q]+1,
                    'Status': '✅' if result['graddings'][q] == 1 else '❌'
                })
            
            st.dataframe(results_df)

        # Clean up temporary file
        os.unlink(temp_file_path)

if __name__ == '__main__':
    main()