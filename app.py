from flask import Flask, render_template, Response
from camera import Camera
import cv2

# import darknet functions to perform object detections
from darknet import *
# load in our YOLOv4 architecture network
print(" * Loading network...")
network, class_names, class_colors = load_network("cfg/yolov4-csp.cfg", "cfg/coco.data", "cfg/yolov4-csp.weights")
width = network_width(network)
height = network_height(network)


# Helper Functions

def darknet_helper(img, width, height):
  darknet_image = make_image(width, height, 3)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_resized = cv2.resize(img_rgb, (width, height),
                              interpolation=cv2.INTER_LINEAR)

  # get image ratios to convert bounding boxes to proper size
  img_height, img_width, _ = img.shape
  width_ratio = img_width/width
  height_ratio = img_height/height

  # run model on darknet style image to get detections
  copy_image_from_bytes(darknet_image, img_resized.tobytes())
  detections = detect_image(network, class_names, darknet_image)
  free_image(darknet_image)
  return detections, width_ratio, height_ratio

# Draw detections on an image
def overlay_boxes(img, detections, width_ratio, height_ratio):
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
        cv2.rectangle(img, (left, top), (right, bottom), class_colors[label], 2)
        cv2.putText(img, "{} [{:.2f}]".format(label, float(confidence)), (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors[label], 2)
    return img

app = Flask(__name__)

@app.route('/')
def index():
    return "App is in /camera"

@app.route('/camera')
def camera():
    return render_template('index.html')

def gen(camera):
    while True:
        frame, original = camera.get_frame()
        detections, width_ratio, height_ratio = darknet_helper(original, width, height)
        labeled = overlay_boxes(original, detections, width_ratio, height_ratio)
        ret, labeled_jpeg = cv2.imencode('.jpg', labeled)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + labeled_jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/test')
def test():
    return render_template('test.html')

def genImg():
   img = cv2.imread("data/person.jpg")
   detections, width_ratio, height_ratio = darknet_helper(img, width, height)
   labeled = overlay_boxes(img, detections, width_ratio, height_ratio)
   ret, labeled_jpeg = cv2.imencode('.jpg', labeled)
   yield (b'--frame\r\n'
          b'Content-Type: image/jpeg\r\n\r\n' + labeled_jpeg.tobytes() + b'\r\n\r\n')

@app.route('/imgtest')
def test_img():
    return Response(genImg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
