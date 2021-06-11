import cv2
import numpy as np
import sys
from os.path import join
#from timeit import default_timer as timer


class meter_value:
    def __init__(self, digit_counter, digit):
        self.digit_counter = digit_counter
        self.digit = digit

    def meter_reading(self):
        return str(self.digit)

    def num_digit(self):
        return self.digit_counter

    def is_complete(self):
        if self.digit_counter == 5:
            return True
        else:
            return False


def detect(img_path):
    net = cv2.dnn.readNet(join(sys.path[0], "yolov3_training_last.weights"), join(sys.path[0], "yolov3_testing_meterdigit.cfg"))
    classes = []
    with open(join(sys.path[0], "digit.names"), "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Loading image
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.6, fy=0.6)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # variables for writing to text file
    image_value = meter_value(0, 0)

    for i in range(len(boxes)):
        if i in indexes:
            label = str(classes[class_ids[i]])
            image_value.digit_counter = image_value.digit_counter + 1
            image_value.digit = str(image_value.digit) + str(label)
            if image_value.is_complete():
                return int(image_value.meter_reading())

    return -1


if __name__ == '__main__':
    # Parse the command line arguments
    # Command Line: python  yolov3_meterdigit_args.py   {full_image_location: /home/red/workspace/image.jpg}
    meter_value = detect(sys.argv[1])
    if meter_value != (-1):
        print(meter_value)
    else:
        print("error")


