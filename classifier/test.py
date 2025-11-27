import cv2
import sys
from ultralytics import YOLO
import yaml
from argparse import Namespace

if __name__ == "__main__":

    config_path = './config.yaml'
    
    print(f"Loading config from: {config_path}")
    with open(config_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    model_path = conf.model_path

    model = YOLO(model_path)

    test_vid_path = conf.video_path

    capture = cv2.VideoCapture(test_vid_path)

    if not capture.isOpened():
        print("error capturing video")
        sys.exit()

    while(True):
        success, frame = capture.read()

        if (success):
            results = model.predict(frame, imgsz=736, verbose=False)

            annotated_frame = results[0].plot()

            cv2.imshow("Vehicle Detection Project", annotated_frame)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    

    capture.release()
    cv2.destroyAllWindows()


