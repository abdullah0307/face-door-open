import cv2
from deepface import DeepFace

# Initialize the video capture
cap = cv2.VideoCapture(
    "Face Test Video.mp4")  # Use 0 for the default camera, change to the appropriate index if you have multiple cameras
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output 2.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:

    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame")
        break

    faces = DeepFace.extract_faces(frame, enforce_detection=False, align=True)
    for face in faces:
        x, y, w, h = (face['facial_area']['x'],
                      face['facial_area']['y'],
                      face['facial_area']['w'],
                      face['facial_area']['h'])
        if x != 0 and y != 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    dfs = DeepFace.find(frame,
                        db_path="KnownFace",
                        model_name="ArcFace",  # For Face Recognition
                        detector_backend="opencv",  # For Face Detection
                        enforce_detection=False,
                        distance_metric="cosine",
                        silent=True)

    for face in dfs:
        if face.shape[0] > 0:
            x, y, w, h = face.source_x[0], face.source_y[0], face.source_w[0], face.source_h[0]
            if x != 0 and y != 0:
                distance = face.distance[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
                most_similar_face = face.iloc[0].identity
                similar_face = cv2.imread(most_similar_face)
                name = most_similar_face.split("\\")[-1][:-4]
                cv2.putText(frame, name, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Face Recognized Door Opened", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown face Door Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Display the frame
    cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
    cv2.imshow('Face Detection', frame)

    # Write the frame into the output video
    out.write(frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
out.release()
cap.release()
cv2.destroyAllWindows()
