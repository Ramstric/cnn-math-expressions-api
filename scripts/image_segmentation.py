import os

import cv2

def segment_upload():
    # Path to images
    image_dir = "../data/uploads"

    # Create a list of the images in the directory
    images = []
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        images.append(image_path)

    # Choose the only available image (there should be only one because the directory is cleared when the API is called)
    img = cv2.imread(images[0])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Digital processing of the image
    ee_blackhat = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, ee_blackhat)
    _, thresh = cv2.threshold(blackhat, 120, 255, cv2.THRESH_BINARY)

    ee = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    img_processed = cv2.dilate(thresh, ee)

    cv2.imwrite(os.path.join("../data/equation_processed", "processed.png"), img_processed)

    # Find contours in the image
    contours, _ = cv2.findContours(img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    srt_crt = sorted(contours, key=lambda _x: cv2.boundingRect(_x)[0])

    img_with_contours = img.copy()
    for i in range(len(srt_crt)):
        x, y, w, h = cv2.boundingRect(srt_crt[i])
        cv2.rectangle(img_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # TODO: Check for alignment of symbols on the x-axis (which means checking for equal signs)
        """
        if i < len(srt_crt) - 1:
            _x, _y, _w, _h = cv2.boundingRect(srt_crt[i+1])
    
            center = x + (w // 2)
            # Only equal symbols are aligned on the x-axis (one above the other)
            if center > _x:
                cv2.line(img_with_contours, (center, y), (center, y + h), (0, 255, 0), 10)
        """


    img_processed = cv2.bitwise_not(img_processed)
    img_with_contours = cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB)

    cv2.imwrite(os.path.join("../data/equation_processed", "contours.png"), img_with_contours)

    # Segmentation of the image
    path_to_save = "../data/equation_segmented"

    # Clear the directory
    for file in os.listdir(path_to_save):
        os.remove(os.path.join(path_to_save, file))

    # Save the segmented images, starting from left most box to right most box
    for i, contour in enumerate(sorted(contours, key=lambda _x: cv2.boundingRect(_x)[0])):
        x, y, w, h = cv2.boundingRect(contour)
        segment = img_processed[y:y+h, x:x+w]

        # Resize the segment to 50x50 pixels
        if w > 50 or h > 50:
            if w > h:
                _w = 50
                _aspect_ratio = h / w
                _h = int(_w * _aspect_ratio)
            else:
                _h = 50
                _aspect_ratio = w / h
                _w = int(_h * _aspect_ratio)

            segment = cv2.resize(segment, (_w, _h))
            w, h = _w, _h

        # Add padding to the segment
        pad_w = int((75 - w) / 2) if w < 75 else 0
        pad_h = int((75 - h) / 2) if h < 75 else 0
        segment = cv2.copyMakeBorder(segment, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        cv2.imwrite(os.path.join(path_to_save, f"segmented_{i}.png"), segment)