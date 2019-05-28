import cv2

cap = cv2.VideoCapture(0)
modelFile = "opencv_face_detector_uint8.pb"
configFile = "opencv_face_detector.pbtxt"
tensorflowNet = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
# Input image
# img = cv2.imread('1.png')
# rows, cols, channels = img.shape

# Show the image with a rectagle surrounding the detected objects
# cv2.imshow('Image', img)
# cv2.waitKey()
# cv2.destroyAllWindows()
cv2.namedWindow('frame',0)
cv2.resizeWindow('frame',640,480)
while (cap.isOpened()):
    ret, frame = cap.read()
    rows, cols, channels = frame.shape

    # Use the given image as input, which needs to be blob(s).
    tensorflowNet.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))

    # Runs a forward pass to compute the net output
    networkOutput = tensorflowNet.forward()

    # Loop on the outputs
    for detection in networkOutput[0, 0]:

        score = float(detection[2])
        if score > 0.9:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows

            # draw a red rectangle around detected objects
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), thickness=2)
    print('发现{}张人脸'.format(len(networkOutput)))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
