import cv2
import numpy as np

def analyze_quality(img_array):
    # 1. Image Analytics: Convert to Grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # 2. Image Analytics: Blur and Threshold to isolate the kernel
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Find Shape/Contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        
        # Grading Logic (Industrial Standards)
        if area > 8000:
            grade = "W180 - Jumbo Premium"
        elif area > 5000:
            grade = "W240 - Standard Whole"
        else:
            grade = "Broken / LWP (Large White Pieces)"
            
        return grade, area
    return "Detection Failed", 0