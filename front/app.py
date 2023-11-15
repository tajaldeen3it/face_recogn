import face_recognition
import cv2

# Load the image of the student
student_image = cv2.imread('student.jpg')

# Encode the image of the student
student_encoding = face_recognition.face_encodings(student_image)[0]

# Capture the video from the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find all faces in the frame
    faces = face_recognition.face_locations(gray_frame)
    face_encodings = face_recognition.face_encodings(gray_frame, faces)

    # Iterate over each face
    for (top, right, bottom, left), face_encoding in zip(faces, face_encodings):
        # Compare the face encoding to the student's encoding
        match = face_recognition.compare_faces([student_encoding], face_encoding)

        # If the face matches the student's face, mark the student as present
        if match[0]:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "Present", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Attendance', frame)

    # Check if the user wants to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()

# Close all windows
cv2.destroyAllWindows()
