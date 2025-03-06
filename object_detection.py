import cv2
import numpy as np

class RedObjectDetector:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)

    def detect_objects(self):
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            
            # Switch from Blue-green-red to Hue-saturation-value color space
            # Better for detecting a color under different lighting conditions
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Values that are red
            # 2 ranges because red wraps around the hue value of 0 degrees.
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])


            # Create masks that keep red pixels and disregard others
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 + mask2
            
            # Find contours in the mask
            # Only draw contours around the outermost countours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw bounding boxes for each detected contour
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Show a frame
            cv2.imshow('Object Detection', frame)
            
            # Close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('Object Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
        
        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = RedObjectDetector()
    detector.detect_objects()