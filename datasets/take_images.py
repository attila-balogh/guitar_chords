import cv2


# laptop webcam
# capture = cv2.VideoCapture(0)

# phone camera
capture = cv2.VideoCapture(1)


# camera props
fps = capture.get(cv2.CAP_PROP_FPS)
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"fps: {fps}\nresolution: {width:.0f} × {height:.0f}")


# with what frequency we take photos for the dataset
freq = 30

actual_frame = 0
while True:
    actual_frame += 1

    success, frame = capture.read()

    # so we can see, what we're doing
    cv2.imshow("Frame", frame)

    # take photo at every 'freqth' frame
    if actual_frame % freq == 0:
        number = int(actual_frame/freq)
        # save image with given name and id number
        cv2.imwrite(f'Chord_name_{1100+number}.jpg', frame)
        print(f"Took photo number {number:3.0f}")

    # quit by hitting 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# kill everything
capture.release()
cv2.destroyAllWindows()
