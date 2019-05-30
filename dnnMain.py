import cv2

isClose = False
# cap = cv2.VideoCapture(0)
modelFile = "opencv_face_detector_uint8.pb"
configFile = "opencv_face_detector.pbtxt"
tensorflowNet = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
# Input image
img = cv2.imread('1.jpg')
rows, cols, channels = img.shape

# Show the image with a rectagle surrounding the detected objects
cv2.namedWindow('frame', 0)
cv2.resizeWindow('frame', 640, 480)
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     rows, cols, channels = frame.shape
#
#     # Use the given image as input, which needs to be blob(s).
#     tensorflowNet.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))
#
#     # Runs a forward pass to compute the net output
#     networkOutput = tensorflowNet.forward()
#
#     # Loop on the outputs
#     for detection in networkOutput[0,0]:
#
#         score = float(detection[2])
#         if score > 0.9:
#             left = detection[3] * cols
#             top = detection[4] * rows
#             right = detection[5] * cols
#             bottom = detection[6] * rows
#
#             # draw a red rectangle around detected objects
#             cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), thickness=1)
#     print('{0}'.format(detection))
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cap.release()
#         cv2.destroyAllWindows()
#         break
# ret, frame = img.read()
rows, cols, channels = img.shape

# Use the given image as input, which needs to be blob(s).
tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))

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
        cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), thickness=3)
# print('{0}'.format(detection))
cv2.imshow('frame', img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
