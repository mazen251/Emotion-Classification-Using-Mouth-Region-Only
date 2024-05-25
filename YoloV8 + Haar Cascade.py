import cv2
import numpy as np
import math
import argparse
import os
from datetime import datetime
from tensorflow.keras.models import load_model

def extract_mouth_haar(frame, face_box, drawing_frame, save_path, frame_count, target_resolution=(100, 50)):
    # Extract face region from the frame
    x, y, w, h = [int(v) for v in face_box]  # Convert to integers
    face_region = frame[y:y + h, x:x + w]

    # Convert face region to grayscale for Haar cascade
    gray_face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

    # Load Haar cascade for mouth detection
    mouth_cascade = cv2.CascadeClassifier('Weights & Models/haarcascade_mcs_mouth.xml')

    # Detect mouths
    mouths = mouth_cascade.detectMultiScale(gray_face_region, scaleFactor=1.8, minNeighbors=15)

    if len(mouths) > 0:
        # Assuming the first detected mouth is the one we want
        mx, my, mw, mh = mouths[0]
        mx += x  # Adjust X coordinate relative to the whole frame
        my += y  # Adjust Y coordinate relative to the whole frame

        # Crop mouth region from the original frame (not the grayscale one)
        mouth_region = frame[my:my + mh, mx:mx + mw]

        # Resize the mouth region
        resized_mouth = cv2.resize(mouth_region, target_resolution)

        # Draw bounding box around the mouth on the drawing frame
        cv2.rectangle(drawing_frame, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)

        # Save the cropped mouth image
        filename = os.path.join(save_path, f"mouth_frame_{frame_count}.jpg")
        cv2.imwrite(filename, resized_mouth)

        # Preprocess for the model
        preprocessed_mouth = resized_mouth / 255.0  # Normalize pixel values
        preprocessed_mouth = np.reshape(preprocessed_mouth, (1, target_resolution[1], target_resolution[0], 3))

        return preprocessed_mouth, (mx, my, mw, mh)  # Add mouth bounding box coordinates to the return statement

    else:
        return None, None  # Return None for both values if no mouth is detected

class YOLOv8_face:
    def __init__(self, path, conf_thres=0.2, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.class_names = ['face']
        self.num_classes = len(self.class_names)
        self.net = cv2.dnn.readNet(path)
        self.input_height = 640
        self.input_width = 640
        self.reg_max = 16
        self.project = np.arange(self.reg_max)
        self.strides = (8, 16, 32)
        self.feats_hw = [(math.ceil(self.input_height / self.strides[i]), math.ceil(self.input_width / self.strides[i]))
                         for i in range(len(self.strides))]
        self.anchors = self.make_anchors(self.feats_hw)

    def make_anchors(self, feats_hw, grid_cell_offset=0.5):
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h, w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset
            y = np.arange(0, h) + grid_cell_offset
            sx, sy = np.meshgrid(x, y)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points

    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_width, self.input_height
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(self.input_width / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_width - neww - left, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
            else:
                newh, neww = int(self.input_height * hw_scale), self.input_width
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_height - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_height - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
        else:
            img = cv2.resize(srcimg, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def detect(self, srcimg):
        input_img, newh, neww, padh, padw = self.resize_image(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB))
        scale_h, scale_w = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        input_img = input_img.astype(np.float32) / 255.0

        blob = cv2.dnn.blobFromImage(input_img)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        det_bboxes, det_conf, det_classid, landmarks = self.post_process(outputs, scale_h, scale_w, padh, padw)
        return det_bboxes, det_conf, det_classid, landmarks

    def post_process(self, preds, scale_h, scale_w, padh, padw):
        bboxes, scores, landmarks = [], [], []
        for i, pred in enumerate(preds):
            stride = int(self.input_height / pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))
            box = pred[..., :self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1, 1))
            kpts = pred[..., -15:].reshape((-1, 15))
            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = self.softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1, 4))
            bbox = self.distance2bbox(self.anchors[stride], bbox_pred,
                                      max_shape=(self.input_height, self.input_width)) * stride
            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (self.anchors[stride][:, 0].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (self.anchors[stride][:, 1].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 2::3] = 1 / (1 + np.exp(-kpts[:, 2::3]))

            bbox -= np.array([[padw, padh, padw, padh]])
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            kpts -= np.tile(np.array([padw, padh, 0]), 5).reshape((1, 15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1, 15))

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)

        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
        classIds = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences > self.conf_threshold
        bboxes_wh = bboxes_wh[mask]
        confidences = confidences[mask]
        classIds = classIds[mask]
        landmarks = landmarks[mask]

        indices = np.array(cv2.dnn.NMSBoxes(np.array(bboxes_wh), np.array(confidences), self.conf_threshold,
                                            self.iou_threshold)).flatten()

        if len(indices) > 0:
            mlvl_bboxes = bboxes_wh[indices]
            confidences = confidences[indices]
            classIds = classIds[indices]
            landmarks = landmarks[indices]
            return mlvl_bboxes, confidences, classIds, landmarks
        else:
            print("Negative")
            return np.array([]), np.array([]), np.array([]), np.array([])

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def draw_detections(self, image, boxes, scores, kpts):
        for box, score, kp in zip(boxes, scores, kpts):
            x, y, w, h = box.astype(int)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
            cv2.putText(image, "face:" + str(round(score, 2)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        thickness=2)
            for i in range(5):
                cv2.circle(image, (int(kp[i * 3]), int(kp[i * 3 + 1])), 4, (0, 255, 0), thickness=-1)
        return image

def crop_lower_third_for_mouth(frame, face_box, save_path, frame_count, target_resolution=(100, 50)):
    x, y, w, h = [int(v) for v in face_box]  # Convert to integers
    start_y = y + (2 * h // 3)
    mouth_region = frame[start_y:y + h, x:x + w]
    resized_mouth = cv2.resize(mouth_region, target_resolution)
    filename = os.path.join(save_path, f"mouth_frame_{frame_count}_yolo.jpg")
    cv2.imwrite(filename, resized_mouth)
    preprocessed_mouth = resized_mouth / 255.0
    preprocessed_mouth = np.reshape(preprocessed_mouth, (1, target_resolution[1], target_resolution[0], 3))
    return preprocessed_mouth

TARGET_RESOLUTION = (100, 50)
model = load_model('Weights & Models/binary_emotion_model (100x50) (New Model).h5')
categories = ['happy', 'sad']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videopath', type=str, default='Testing Vids & Images/Untitled video - Made with Clipchamp.mp4', help="video path")
    parser.add_argument('--modelpath', type=str, default='Weights & Models/yolov8n-face.onnx', help="onnx filepath")
    parser.add_argument('--confThreshold', default=0.45, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    args = parser.parse_args()

    YOLOv8_face_detector = YOLOv8_face(args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)

    cap = cv2.VideoCapture(args.videopath)
    if not cap.isOpened():
        print("Error: Couldn't open video file.")
        exit()

    frame_count = 0
    save_path = 'Cropped Images from Run'
    os.makedirs(save_path, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_for_drawing = frame.copy()
        boxes, scores, classids, kpts = YOLOv8_face_detector.detect(frame)
        YOLOv8_face_detector.draw_detections(frame_for_drawing, boxes, scores, kpts)

        for i, box in enumerate(boxes):
            mouth_region, mouth_bbox = extract_mouth_haar(frame, box, frame_for_drawing, save_path, frame_count)
            if mouth_region is None:
                # Fallback to Yolo once the haar cascade fails to detect the mouth region (FALLBACKKKKK)
                mouth_region = crop_lower_third_for_mouth(frame, box, save_path, frame_count, TARGET_RESOLUTION)

            if mouth_region is not None:
                prediction = model.predict(mouth_region)
                predicted_class = np.argmax(prediction)
                confidence = prediction[0][predicted_class]
                emotion = categories[predicted_class]

                label = f"{emotion}: {confidence * 100:.2f}%"
                cv2.putText(frame_for_drawing, label, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            frame_count += 1

        cv2.imshow('Emotion Detection', frame_for_drawing)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
