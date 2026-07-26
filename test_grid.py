import cv2
import numpy as np
from PIL import Image

cap = cv2.VideoCapture('videos/large_model_track_test/LM_test_0066/Depth/Depth.mp4')
frames = []
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for i in [1, 2, 3, 4]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * i / 5))
    ret, f = cap.read()
    if ret: 
        frames.append(cv2.resize(f, (224, 224)))
cap.release()

if len(frames) == 4:
    top = np.hstack((frames[0], frames[1]))
    bot = np.hstack((frames[2], frames[3]))
    grid = np.vstack((top, bot))
    cv2.imwrite('grid_test.jpg', grid)
    print('Grid saved')
else:
    print('Failed to get 4 frames')
