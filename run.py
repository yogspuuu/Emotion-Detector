import cv2
import numpy as np
from keras import models
from tensorflow.keras.preprocessing.image import img_to_array

# Load haraarcascade to make box around face
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the saved model
model = models.load_model('model_weights.h5')

# Capture video using cv2
video = cv2.VideoCapture(0)

# Prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# Dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Smile", 1: "Non-smile"}  # {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Keep looping
while True:
    # Retrive video frame to draw bounding box around face
    retrive, frame = video.read()

    if not retrive:
        break

	# Convert frame to BGR2GRAY
    gray = cv2.cvtColor(
		frame, 
		cv2.COLOR_BGR2GRAY
	)

	# Detect face
    faces = detector.detectMultiScale(
		gray, 
		scaleFactor=1.3, 
		minNeighbors=5
	)

	# Loop each attribute in faces
    for (x, y, w, h) in faces:
		# Draw rectangle for face detection overlay
        cv2.rectangle(
			frame, 
			(x, y - 50), 
			(x + w, y + h + 10), 
			(255, 0, 0),
			2
		)

		# Conver color to gray
        roi_gray = gray[
			y:y + h, 
			x:x + w
		]

		# crop image(chage size) to 64x64
        cropped_img = np.expand_dims(
			np.expand_dims(
				cv2.resize(
					roi_gray, 
					(64, 64)
				), -1
			), 0
		)

		# Prediction cropped images
        prediction = model.predict(cropped_img)

		# Get index value(prediction array value from model_weight.h5)
        maxindex = int(np.argmax(prediction))

		# Add text to rectangle overlay
        cv2.putText(
			frame, 
			emotion_dict[maxindex], 
			(x+20, y-60),
			cv2.FONT_HERSHEY_SIMPLEX, 1, 
			(255, 255, 255), 2, 
			cv2.LINE_AA
		)

	# Show image output
    cv2.imshow(
		'Video', 
		cv2.resize(
			frame, 
			(1600, 960), 
			interpolation = cv2.INTER_CUBIC
		)
	)

	# Make q hotkey for close camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Run video
video.release()

# Cloase all related window(video)
cv2.destroyAllWindows()
