import cv2
import time

def detect_and_time(video_path):
    cap = cv2.VideoCapture(video_path)

    # Load the HOG person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Initialize variables
    person_tracker = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect people in the image
        boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)

        # Simple tracking logic based on overlap
        for i, (x, y, w, h) in enumerate(boxes):
            matched = False
            for person_id, (prev_x, prev_y, prev_w, prev_h, start_time) in person_tracker.items():
                overlap_x = max(x, prev_x)
                overlap_y = max(y, prev_y)
                overlap_w = min(x + w, prev_x + prev_w) - overlap_x
                overlap_h = min(y + h, prev_y + prev_h) - overlap_y
                overlap_area = overlap_w * overlap_h
                area1 = w * h
                area2 = prev_w * prev_h
                iou = overlap_area / (area1 + area2 - overlap_area)
                if iou > 0.5:
                    person_tracker[person_id] = (x, y, w, h, start_time)
                    matched = True
                    break
            if not matched:
                person_tracker[i] = (x, y, w, h, time.time())

        # Draw bounding boxes and display time
        for person_id, (x, y, w, h, start_time) in person_tracker.items():
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            elapsed_time = time.time() - start_time
            time_text = f"Time: {elapsed_time:.2f}s"
            cv2.putText(frame, time_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        #cv2.imshow("Human Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    video_path = 'foot.mp4'

    detect_and_time(video_path)
