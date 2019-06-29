import cv2
import os

while True:
    cv2.waitKey(int(1000 / 60))
    img_rgb = cv2.imread('img.jpg')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('redbird-downflap.png', 0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img_rgb, top_left, bottom_right, 255, 2)

    cv2.imshow('image_test', img_rgb)

cv2.destroyAllWindows()
