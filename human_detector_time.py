import cv2
import numpy as np
import time
import csv

def detect_and_time(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Initialize YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Change to DNN_TARGET_CUDA for GPU
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    person_tracker = {}
    table_tracker = {}
    next_person_id = 0
    next_table_id = 0
    frame_count = 0

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, int(fps / 3))  # Process every 10th frame or less

    # Open CSV file for writing
    with open('timedtable.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Table ID', 'Person ID', 'Time (seconds)', 'Bill Amount'])

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break

            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            height, width, _ = frame.shape

            # Detect objects
            blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            new_person_tracker = {}
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    
                    if class_id == 0:  # Person
                        matched = False
                        for person_id, (prev_x, prev_y, prev_w, prev_h, start_time, table_id) in person_tracker.items():
                            iou = calculate_iou((x, y, w, h), (prev_x, prev_y, prev_w, prev_h))
                            if iou > 0.3:
                                new_person_tracker[person_id] = (x, y, w, h, start_time, table_id)
                                matched = True
                                break
                        if not matched:
                            new_person_tracker[next_person_id] = (x, y, w, h, time.time(), None)
                            next_person_id += 1
                    
                    elif class_id == 60:  # Table (assuming class ID 60 is for tables, adjust if necessary)
                        matched = False
                        for table_id, (prev_x, prev_y, prev_w, prev_h) in table_tracker.items():
                            iou = calculate_iou((x, y, w, h), (prev_x, prev_y, prev_w, prev_h))
                            if iou > 0.5:
                                table_tracker[table_id] = (x, y, w, h)
                                matched = True
                                break
                        if not matched:
                            table_tracker[next_table_id] = (x, y, w, h)
                            next_table_id += 1

            # Check if people are sitting at tables
            for person_id, (px, py, pw, ph, start_time, current_table_id) in new_person_tracker.items():
                person_center = (px + pw // 2, py + ph)
                for table_id, (tx, ty, tw, th) in table_tracker.items():
                    if tx < person_center[0] < tx + tw and ty < person_center[1] < ty + th:
                        new_person_tracker[person_id] = (px, py, pw, ph, start_time, table_id)
                        break

            person_tracker = new_person_tracker

            # Draw bounding boxes and display information (only if needed for debugging)
            frame_with_boxes = frame.copy()
            for person_id, (x, y, w, h, start_time, table_id) in person_tracker.items():
                cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                elapsed_time = time.time() - start_time
                time_text = f"ID: {person_id}, Time: {elapsed_time:.2f}s"
                if table_id is not None:
                    time_text += f", Table: {table_id}"
                cv2.putText(frame_with_boxes, time_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Write to CSV if person has been at a table for more than 5 seconds
                if table_id is not None and elapsed_time > 5:
                    bill_amount = calculate_bill(elapsed_time)
                    csvwriter.writerow([table_id, person_id, f"{elapsed_time:.2f}", f"{bill_amount:.2f}"])

            for table_id, (x, y, w, h) in table_tracker.items():
                cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame_with_boxes, f"Table {table_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Display the frame (comment out for faster processing if not needed)
            cv2.imshow("Cafe Tracking", frame_with_boxes)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()



def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def calculate_bill(time_spent):
    # Example billing logic: $5 base fee + $0.10 per second
    return 5 + (time_spent * 0.10)

if __name__ == "__main__":
    video_path = 'Canteen.mp4'  # Path to your video file
    detect_and_time(video_path)