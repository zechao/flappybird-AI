import cv2
import pygame

import game.wrapped_flappy_bird as flappy

# img = cv2.imread('image/im5.png')
#
# cv2.imshow('bird', img)
# rect =cv2.selectROI("bird",img,True,False)
#
# print("Selected area")
# (x, y, w, h) = rect
# # Crop image
# imCrop = img[y : y+h, x:x+w]
#
# # Display cropped image
# cv2.imshow("image_roi", imCrop)
#
#
# while (1):
#     cv2.imshow('bird', img)
#     k = cv2.waitKey(0) & 0xFF  # when using 64bit machine
#     if k == 27:  # wait for ESC key to exit
#         break
# cv2.destroyAllWindows()



game = flappy.GameState()
frame = game.next_frame(False)
cv2.imshow('bird', frame)
run = False
flap = False
while (1):
    flap = False
    # frame = game.next_frame(False)
    cv2.imshow('bird', frame)
    k = cv2.waitKey(1) & 0xFF  # when using 64bit machine
    if k == 27:  # wait for ESC key to exit
        break
    if k == ord('1'):
        run = True
    if k == ord('w'):
        run = False
        frame = game.next_frame(True)
    if k == ord('s'):
        run = False
        frame = game.next_frame(False)
    if k == 32:
        flap = True
    if k == ord('p'):
        run = False
    if run:
        frame = game.next_frame(flap)

game.quit()
cv2.destroyAllWindows()

