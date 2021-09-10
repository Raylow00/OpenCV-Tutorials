import cv2

# Load the cascade data file
face_cascade = cv2.CascadeClassifier("../../HaarCascade_data/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml")
nose_cascade = cv2.CascadeClassifier("../../HaarCascade_data/opencv-master/data/haarcascades/haarcascade_mcs_nose.xml")
eye_cascade = cv2.CascadeClassifier("../../HaarCascade_data/opencv-master/data/haarcascades/haarcascade_mcs_eyepair_big.xml")

# Load the moustache image
moustache_image = cv2.imread("../../Images/moustache.png", -1)
# Load the glasses image
glasses_image = cv2.imread("../../Images/glasses.png", -1)

# Create a mask for the moustache
moustache_image_mask = moustache_image[:, :, 3]
print("Moustache image mask: ", moustache_image_mask)
moustache_image = moustache_image[:, :, 0:3]

# Create a mask for the moustache
glasses_image_mask = glasses_image[:, :, 3]
print("Moustache image mask: ", glasses_image_mask)
glasses_image = glasses_image[:, :, 0:3]

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Create the face ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        noses = nose_cascade.detectMultiScale(roi_gray)
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # Calculate the coords for placing the glasses image
            x1 = int(ex - ew/10) 
            x2 = int((ex + ew) + ew/10)
            y1 = int(ey)
            y2 = int(ey + eh + eh + eh/2)

            if x1 < 0 or x2 < 0 or x2 > w or y2 > h:
                continue

            glasses_image_width = int(x2-x1)
            glasses_image_height = int(y2-y1)

            mask = cv2.resize(glasses_image_mask, (glasses_image_width, glasses_image_height))
            mask_inv = cv2.bitwise_not(mask)

            img = cv2.resize(glasses_image, (glasses_image_width, glasses_image_height))

            roi = roi_color[y1:y2, x1:x2]
            # Create ROI foreground and background
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            roi_fg = cv2.bitwise_and(img, img, mask=mask)

            res = cv2.add(roi_bg, roi_fg)

            roi_color[y1:y2, x1:x2] = res

            break

        for (nx, ny, nw, nh) in noses:
            # Calculate the coords for placing the moustache image
            x1 = int(nx - nw/2) 
            x2 = int(nx + nw/2 + nw)
            y1 = int(ny + nh/2 + nh/8)
            y2 = int(ny + nh + nh/4 + nh/6)

            if x1 < 0 or x2 < 0 or x2 > w or y2 > h:
                continue

            moustache_image_width = int(x2-x1)
            moustache_image_height = int(y2-y1)

            mask = cv2.resize(moustache_image_mask, (moustache_image_width, moustache_image_height))
            mask_inv = cv2.bitwise_not(mask)

            img = cv2.resize(moustache_image, (moustache_image_width, moustache_image_height))

            roi = roi_color[y1:y2, x1:x2]
            # Create ROI foreground and background
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            roi_fg = cv2.bitwise_and(img, img, mask=mask)

            res = cv2.add(roi_bg, roi_fg)

            roi_color[y1:y2, x1:x2] = res

            break

    cv2.imshow("Snapchat-based moustache and glasses overlay", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()