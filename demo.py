"""Demo for use yolo v3
"""
import os
import sys
import time
import cv2
import argparse
import numpy as np
from model.yolo_model import YOLO
from pantilthat import *


# Input Args/Switches
ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--verbose", action="store_true", default=0, help="Whether or not to display location messages in terminal")
ap.add_argument("-bbb", "--show_blue_border_box", action="store_true", default=0, help="Draws a (blue) border boundary box which can dictate camera movement. Default: Show box")
ap.add_argument("-p", "--show_person_box", action="store_true", default=0, help="Draws a (green) box around the person. Default: Show box")
ap.add_argument("-m", "--still_camera", action="store_true", default=0, help="Keep camera Still when enabled (1); otherwise the camera moves (0). Default: Camera moves")
ap.add_argument("-c", "--is_cascade", action="store_true", default=0, help="Whether to use Haar Cascade (1) or YOLO (0). Default: Yolo (0)")
ap.add_argument("-f", "--is_wear_fun_hat", action="store_true", default=0, help="Puts a fun hat on you. Works only with Haar Cascade ATM. Default: No fun hat")
args = ap.parse_args()


# GLOBALS
show_text = False
show_boundary_box = args.show_blue_border_box
show_person_box = args.show_person_box
is_camera_still = args.still_camera
is_cascade = args.is_cascade
is_wear_fun_hat = args.is_wear_fun_hat
hat_path = './data/Propeller_hat.svg.med.png'
hat_img = cv2.imread(hat_path, -1)
# Frame dimensions vars
FRAME_W = 0
FRAME_H = 0
# Boundary Box vars
w_min = 0
w_max = 0
h_min = 0
h_max = 0


def man_move_camera(key_press):
    """Take keystrokes to dictate camera movemement.

    # Argument:
        key_press: takes in one letter for movement 
                    (same as gaming controls, no inversion):
                        w: Up
                        a: Left
                        s: Down
                        d: Right

    # Returns
        None
    """    
    cam_pan = get_pan()
    cam_tilt = get_tilt()
    move_x = 0
    move_y = 0
    
    if(key_press.lower() == 'a'):
        move_x = -2
    elif(key_press.lower() == 'd'):
        move_x = 2
    elif(key_press.lower() == 's'):
        move_y = -1
    elif(key_press.lower() == 'w'):
        move_y = 1

    if((cam_pan + move_x < 90) & (cam_pan - move_x > -90)):
        cam_pan += move_x
        pan(int(cam_pan))
        time.sleep(0.005)
    else:
        print(f'MAX PAN - cannot move:  {cam_pan + move_x}')

    if((cam_tilt + move_y < 90) & (cam_tilt - move_y > -90)):
        cam_tilt -= move_y
        tilt(int(cam_tilt))
        time.sleep(0.005)
    else:
        print(f'MAX TILT - cannot move:  {cam_tilt + move_y}')
    
    return

def move_camera(x, y, w, h):
    """Takes in object tracking coordinates and moves camera to try to "center" the subject.

    # Argument:
        x: coordinate on the x axis where subject is detected
        y: coordinate on the y axis where subject is detected
        w: width of object detected on screen
        h: height of object detected on screen

    # Returns
        None
    """
    if(is_camera_still):
        return
    
    cam_pan = get_pan()
    cam_tilt = get_tilt()
    move_x = 2
    move_y = 1
    yolo_offset = 0 if is_cascade else (h_min * -0.75)
    
    if(((x + w)*0.95 > w_max) & (x*0.95 < w_min)):
        # If both subject borders take up 95% or
        # more of the boundary box, do nothing
        pass
    elif(w > (w_max - w_min)*0.95):
        # If subject border-length take up 95% (not centered)
        # or more of the boundary box, correct movement by aligning centers
        if(x + w/2 > (FRAME_W + w_min)/2):
            cam_pan += move_x
            pan(int(cam_pan))
        elif(x - w/2 < (FRAME_W - w_min)/2):
            cam_pan -= move_x
            pan(int(cam_pan))
    elif((cam_pan + move_x < 90) & (cam_pan - move_x > -90)):
        if(x + w > w_max):
            cam_pan += move_x
            pan(int(cam_pan))
        elif(x < w_min):
            cam_pan -= move_x
            pan(int(cam_pan))
    else:
        print(f'MAX PAN - cannot move:  {cam_pan + move_x}')

    if(((y + h)*0.95 > h_max) & (y*0.95 < h_min)):
        # If both subject borders take up 95% or
        # more of the boundary box, do nothing
        pass
    elif(h > (h_max - h_min)*0.95):
        # If subject border-length take up 95% (not centered)
        # or more of the boundary box, correct movement by aligning centers
        if(y + h/2 > (FRAME_H + h_min)/2):
            cam_tilt += move_y
            tilt(int(cam_tilt))
        elif(y - h/2 < (FRAME_H - h_min)/2):
            cam_tilt -= move_y
            tilt(int(cam_tilt))
    elif((cam_tilt + move_y < 90) & (cam_tilt - move_y > -90)):
        if(y + h > h_max):
            cam_tilt += move_y
            tilt(int(cam_tilt))
        elif(y < h_min + yolo_offset):
            cam_tilt -= move_y
            tilt(int(cam_tilt))
    else:
        print(f'MAX TILT - cannot move:  {cam_tilt + move_y}')
    
    return


def process_image(img):
    """Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image


def get_classes(file):
    """Get classes name.

    # Argument:
        file: classes name for database.

    # Returns
        class_names: List, classes name.
    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names


def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    # Returns
        None
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))
        
        if(not is_camera_still):
            move_camera(x, y, w, h)

        if(show_person_box):
            cv2.rectangle(image, (top, left), (right, bottom), (0, 0, 255), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                        (top, left - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 1,
                        cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()
    
    return


# def detect_image(image, yolo, all_classes):
def detect_image(image, yolo, all_classes, w=0, h=0):
    """Use yolo v4 tiny to detect images.

    # Argument:
        image: original image.
        yolo: YOLO, yolo model.
        all_classes: all classes name.
        w: width of subject detected on screen
        h: height of subject detected on screen

    # Returns:
        image: processed image.
    """
    # pimage = process_image(image)

    start = time.time()
    # boxes, classes, scores = yolo.predict(pimage, image.shape)
    
    ###
    if(w==0 or h == 0):
        w = image.shape[0]
        h = image.shape[1]
    boxes, classes, scores = yolo.predict(image, (w, h))
    ###
    
    end = time.time()

    print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)

    return image


# def detect_video(video, yolo, all_classes):
#     """Use yolo v4 tiny to detect video.

#     # Argument:
#         video: video file.
#         yolo: YOLO, yolo model.
#         all_classes: all classes name.
#     """
#     video_path = os.path.join("videos", "test", video)
#     camera = cv2.VideoCapture(video_path)
#     cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)

#     # Prepare for saving the detected video
#     sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
#         int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#     fourcc = cv2.VideoWriter_fourcc(*'mpeg')

#     vout = cv2.VideoWriter()
#     vout.open(os.path.join("videos", "res", video), fourcc, 20, sz, True)

#     while True:
#         res, frame = camera.read()

#         if not res:
#             break

#         image = detect_image(frame, yolo, all_classes)
#         cv2.imshow("detection", image)

#         # Save the video frame by frame
#         vout.write(image)

#         if cv2.waitKey(110) & 0xff == 27:
#                 break

#     vout.release()
#     camera.release()
    
def reset_camera_position():
    """Resets Camera position.

    # Argument:
        None

    # Returns:
        None
    """
    pan(0)
    tilt(-20)
    time.sleep(2)



if __name__ == '__main__':
    """Main Function.

    # Argument:
        None

    # Returns:
        None
    """
    cfg_file = 'data/custom-yolov4-tiny-detector.cfg'
    weights_file = 'data/custom-yolov4-tiny-detector_best.weights'
    file = 'data/custom-yolov4-tiny-detector.names'
    all_classes = get_classes(file)
    yolo = YOLO(cfg_file, weights_file)

    # Turn the camera to the default position
    reset_camera_position()


    cap = cv2.VideoCapture(0) # Primary, Laptop Camera or rpi Camera
    # cap = cv2.VideoCapture(1) # Secondary, Monitor Camera

    # SET GLOBALS
    FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #width
    FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #height
    ################ BOUNDARY BOX ################
    w_min = int((FRAME_W)/6)
    w_max = int((FRAME_W) - w_min)
    h_min = int((FRAME_H)/5)
    h_max = int((FRAME_H) - h_min)
    # h_min = int(5)
    ################ BOUNDARY BOX ################

    cascPath = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)

    prev_time = time.time()
    show_detect_ok = False
    
    count = 0

    while True:
        ret, frame = cap.read()
        if ret == False:
            print("Error getting image")
            continue
        
        # Vertical flip for camera orientation (Ribbon on top of camera)
        frame = cv2.flip(frame, 0)

        if(is_cascade):# | args.is_cascade):
            faces = faceCascade.detectMultiScale(frame, 1.1, 3)
            for (x, y, w, h) in faces:
                ################## HAT ##################
                # https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
                if(is_wear_fun_hat):
                    try:
                        resize_x = int(w*1.1)
                        resize_y = int(w*2/3)
                        overlay = cv2.resize(hat_img, (resize_x, resize_y), interpolation = cv2.INTER_AREA)
                        # overlay = cv2.resize(overlay, (170, 100),interpolation = cv2.INTER_AREA) # Fixed size
                        x_offset = x - 10
                        y_offset = y - (h//2)
                        y1, y2 = y_offset, y_offset + overlay.shape[0]
                        x1, x2 = x_offset, x_offset + overlay.shape[1]
                        alpha_s = overlay[:, :, 3] / 255.0
                        alpha_l = 1.0 - alpha_s
                        for c in range (0, 3):
                            frame[y1:y2, x1:x2, c] = (alpha_s * overlay[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])
                        # Draw hat without transparency
                        # x_offset = y_offset = 50
                        # frame[y_offset:y_offset+overlay.shape[0], x_offset:x_offset+overlay.shape[1]] = overlay
                    except:
                        print('Cannot draw hat; face moved out of canvas-drawing area.')
                elif(show_person_box):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, 'Haar Cascade',
                        (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 1,
                        cv2.LINE_AA)

                if(not is_camera_still):
                    move_camera(x, y, w, h)
                break

            frame = cv2.resize(frame, (FRAME_W,FRAME_H))
            # count += 1
            # if(count == 20):
            #     is_cascade = False
        else:
            # IS YOLO
            # if(count == 20):
            #     is_cascade = True
            #     count = 0
            #     tmp_box = show_person_box
            #     show_person_box = False
            #     frame = detect_image(frame, yolo, all_classes, FRAME_W,FRAME_H)
            #     show_person_box = tmp_box
            # else:
            frame = detect_image(frame, yolo, all_classes, FRAME_W,FRAME_H)
        
        # Show stats, e.g. camera tracking on, positions, model, etc.
        if(show_text):
            nerd_text1 = f'Nerd Stats: x: {x}, y: {y}, w: {w}, h: {h}'
            cv2.putText(frame, nerd_text1,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (192, 96, 0), 1,
                cv2.LINE_AA)
            casc_txt = 'Method: Haar Cascade, ' if is_cascade else 'Method: YOLO, '
            cam_mov_txt = 'Camera Move: Manual Only' if is_camera_still else 'Camera Move: Auto+'
            nerd_text2 = casc_txt + cam_mov_txt
            cv2.putText(frame, nerd_text2,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (96, 96, 96), 1,
                cv2.LINE_AA)

        ################ BOUNDARY BOX ################
        if(show_boundary_box):
            cv2.rectangle(frame, (w_min, h_min), (w_max, h_max), (255, 0, 0), 2)

        # Show Frame!
        cv2.imshow('Video', frame)
        
        # Take input for features
        key_stroke = cv2.waitKey(1)
        
        if key_stroke & 0xFF == ord('q'):
            break
        elif key_stroke & 0xFF == 27:
            break
        elif key_stroke & 0xFF == ord('b'):
            show_boundary_box = not show_boundary_box
        elif key_stroke & 0xFF == ord('p'):
            show_person_box = not show_person_box
        elif key_stroke & 0xFF == ord('m'):
            is_camera_still = not is_camera_still
        elif key_stroke & 0xFF == ord('c'):
            is_cascade = not is_cascade
        elif key_stroke & 0xFF == ord('y'):
            is_cascade = not is_cascade
        elif key_stroke & 0xFF == ord('h'):
            is_wear_fun_hat = not is_wear_fun_hat
        elif key_stroke & 0xFF == ord('f'):
            is_wear_fun_hat = not is_wear_fun_hat
        elif key_stroke & 0xFF == ord('r'):
            reset_camera_position()
        elif key_stroke & 0xFF == ord('w'):
            man_move_camera('w')
        elif key_stroke & 0xFF == ord('a'):
            man_move_camera('a')
        elif key_stroke & 0xFF == ord('s'):
            man_move_camera('s')
        elif key_stroke & 0xFF == ord('d'):
            man_move_camera('d')
        elif key_stroke & 0xFF == ord('t'):
            show_text = not show_text
        
        prev_time = time.time()
    
    # Release/Destroy resources when finished
    cap.release()
    cv2.destroyAllWindows()
