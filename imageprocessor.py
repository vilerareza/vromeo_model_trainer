from cv2 import imread, resize
import numpy as np


class ImageProcessor():
    def __init__(self) -> None:
        from mtcnn.mtcnn import MTCNN
        self.detector = MTCNN()

    def extract_face(self, image_path, target_size = (224,224)):
        img, box = self.detect_face(image_path)
        if box:
            x1, y1, width, height = box
            x2, y2 = x1 + width, y1 + height
            # face data array
            face = img[y1:y2, x1:x2]
            # Resizing
            factor_y = target_size[0] / face.shape[0]
            factor_x = target_size[1] / face.shape[1]
            factor = min (factor_x, factor_y)
            face_resized = resize(face, (int(face.shape[0]* factor), int(face.shape[1]*factor)))
            diff_y = target_size[0] - face_resized.shape[0]
            diff_x = target_size[1] - face_resized.shape[1]
            # Padding
            face_resized = np.pad(face_resized,((diff_y//2, diff_y - diff_y//2), (diff_x//2, diff_x-diff_x//2), (0,0)), 'constant')
            # Progress
            #self.progress_value += self.progress_step
            return face_resized
        return None

    def detect_face(self, image_path):
        img = imread(image_path)
        detection = self.detector.detect_faces(img)
        if (len(detection)>0):
            box = detection[0]['box']
            return img, box
        else:
            return img, None

