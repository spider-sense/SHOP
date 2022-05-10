# general-purpose libraries
import argparse
import os
import cv2
import youtube_dl
import numpy as np
import torch
from tqdm import tqdm
import json

# needed libraries for YOLOv5
from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.general import (check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.metrics import bbox_iou

# needed libraries for DeblurGANv2
from DeblurGANv2.predict import Predictor

# needed libraries for tf-pose-estimator
try:
    from tf_pose_estimation.estimator import TfPoseEstimator
    from tf_pose_estimation.networks import get_graph_path
except:
    print("Pipeline being run from non-Linux environment")

# needed libraries for top-down pose-estimator
from pose_estimation.infer import Pose
from pose_estimation.pose.utils.utils import draw_keypoints
from pose_estimation.pose.utils.boxes import scale_boxes
from pose_estimation.pose.utils.decode import get_final_preds, get_simdr_final_preds

# importing necessary custom libraries
from helperFunctions import distGet, bbox_overlap, INNER_RATIO, HANDHELD_MAP

# SHOP pipeline class
class SHOP:    
    """
    Initiation function creates the necessary AI models to run inference
    """
    def __init__(self, 
                 noDeblur,
                 poseNum,
                 weights,
                 data,
                 device,
                 half,
                 dnn,
                 model,
                 scales,
                 det_model,
                 pose_model,
                 weights_path,
                 person_iou_thres, 
                 person_conf_thres,
                 imgsz,
                 augment):
        # Generating yolov5 model
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data)
        if half:
            self.model.half()
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.augment = augment
        
        # Half precision model depending on settings
        self.half = False
        self.half &= (self.pt or self.jit or self.onnx or self.engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
        
        # Generating DeblurGANv2 model if noDeblur is false + saving no deblur
        self.noDeblur = noDeblur        
        if not noDeblur:
            self.predictor = Predictor(weights_path=weights_path)

        # Checking if top-down pose-estimator needs to be made
        self.poseNum = poseNum
        if poseNum != 0:
            self.top_down = Pose(
                det_model,
                pose_model,
                imgsz[0],
                person_conf_thres,
                person_iou_thres
            )
        
        # Checking if bottom-up pose-estimator needs to be made
        if poseNum != -1:
            dim = [int(i) for i in model.split("_")[-1].split("x")]
            w, h = dim[0], dim[1]
            self.bottom_up = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
        
        # Giving torch memory statistics if GPU device is used.
        if str(self.device) == "cuda:0":
            print("\nAI Initialization Memory Allocation (GPU)", torch.cuda.memory_allocated(0) / 1000000000, "GB")
    
    """
    Inference function to actually run an AI prediction on the image
    """
    def infer(self, 
              source,
              project,
              name,
              conf_thres,
              upper_conf_thres,
              iou_thres,
              noElbow,
              noPose,
              allDet,
              overlap,
              save_txt,
              nosave,
              classes,
              agnostic_nms,
              visualize,
              line_thickness,
              wsl,
              imgsz,
              max_det,
              handheld,
              noCheck,
              saveAOI):        
        # Checking and verifying source type
        isDir = os.path.isdir(source)
        isIm = os.path.isfile(source) and source.split(".")[-1] in IMG_FORMATS
        isVid = os.path.isfile(source) and source.split(".")[-1] in VID_FORMATS
        isYouTube = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        isWeb = source.isnumeric()
        if [isIm, isDir, isVid, isWeb, isYouTube] == [False, False, False, False, False]:
            raise ValueError("Invalid Source Given For Inference")
        
        # Creating project folder and save path
        source = source.rstrip("/")
        nameTrack = 0
        projectFolder = project + name
        while os.path.isdir(projectFolder + str(nameTrack)):
            nameTrack += 1
        projectFolder += str(nameTrack) + "/"
        os.mkdir(projectFolder)
        savePath = projectFolder + source.split("/")[-1]
        
        # Loading image size settings
        imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        
        # creating pose saving cache if needed
        poseSave = {}
        
        # image analysis perform
        if isIm:
            # loads the image
            img = cv2.imread(source)
            
            # analyzes the image
            img, checkBoxes = self.forward(img, imgsz, upper_conf_thres, conf_thres, iou_thres, classes, agnostic_nms, max_det, line_thickness, handheld,\
                               allDet, overlap, noPose, noElbow, os.path.splitext(savePath)[0] + ".txt", save_txt, noCheck, visualize)
               
            # saving area of interest if needed
            if saveAOI:
                checkBoxes = [box.tolist() for box in checkBoxes]
                poseSave[source] = checkBoxes
                
            # saves the image's analysis to the project folder as long as nosave not enabled
            if not nosave:
                print(f"Saving image to {savePath}")
                cv2.imwrite(savePath, img)
            
        # directory of images analysis perform
        if isDir:
            # makes directory then iterates over images and writes them
            os.mkdir(savePath)
            imDir = os.listdir(source)
            # cache dir making if needed
            cacheDir = savePath + "/detections/"
            if save_txt:
                os.mkdir(cacheDir)
            for i, image in tqdm(enumerate(imDir)):
                if image.split(".")[-1] in IMG_FORMATS:
                    # getting the image
                    imPath = os.path.join(source, image)
                    print(f"Reading from {imPath}")
                    img = cv2.imread(imPath)
                    
                    # making cache path
                    cachePath = os.path.join(cacheDir, os.path.splitext(image)[0]) + ".txt"
                    
                    # analyzing the image
                    img, checkBoxes = self.forward(img, imgsz, upper_conf_thres, conf_thres, iou_thres, classes, agnostic_nms, max_det, line_thickness, handheld,\
                                       allDet, overlap, noPose, noElbow, cachePath, save_txt, noCheck, visualize)
                    
                    # saving AOI if needed
                    if saveAOI:
                        checkBoxes = [box.tolist() for box in checkBoxes]
                        poseSave[image] = checkBoxes
                        
                    # saves the image's analysis to the project folder
                    if not nosave:
                        print(f"Saving image to {savePath}")
                        cv2.imwrite(savePath + "/" + image, img)
                
        # video analysis perform
        if isVid:
            cam = cv2.VideoCapture(source)
            imageWidth = int(cam.get(3))
            imageHeight = int(cam.get(4))
            fps = cam.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            if not nosave:
                out = cv2.VideoWriter(savePath, fourcc, fps, (imageWidth, imageHeight))
                print(f"Saving to {savePath} with fps {fps}")
            frameTrack = 0
            
            # collecting cache path
            cacheDir = os.path.splitext(savePath)[0]
            if save_txt:
                os.mkdir(cacheDir)
            
            while True:
                # tracking the frames
                frameTrack += 1
                print(f"Frame: {frameTrack}")
                
                # making cache path
                cachePath = os.path.join(cacheDir, str(frameTrack)) + ".txt"
                
                # reading from video
                ret, frame = cam.read()
                
                if ret:
                    # analyzing the image
                    img, checkBoxes = self.forward(frame, imgsz, upper_conf_thres, conf_thres, iou_thres, classes, agnostic_nms, max_det, line_thickness, handheld,\
                                       allDet, overlap, noPose, noElbow, cachePath, save_txt, noCheck, visualize)
                    
                    # saving AOI if needed
                    if saveAOI:
                        checkBoxes = [box.tolist() for box in checkBoxes]
                        poseSave[frameTrack] = checkBoxes
                    
                    if not nosave:
                        # saves image to video writer if enabled
                        out.write(img)
                else:
                    break
            if not nosave:
                out.release()
            cam.release()
        
        # webcam analysis perform
        if isWeb:
            # adding cache path
            cacheDir = str(savePath)
            if save_txt:
                os.mkdir(cacheDir)
            
            # continuing with the webcam analysis
            savePath += ".mp4"
            cam = cv2.VideoCapture(0)
            imageWidth = int(cam.get(3))
            imageHeight = int(cam.get(4))
            fps = cam.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            if not nosave:
                out = cv2.VideoWriter(savePath, fourcc, fps, (imageWidth, imageHeight))
            frameTrack = 0
            print(f"saving webcam to path {savePath} with {fps}")
            while True:
                # tracking the frames
                frameTrack += 1
                print(f"Frame: {frameTrack}")
                
                # making cache path
                cachePath = os.path.join(cacheDir, str(frameTrack)) + ".txt"
                
                # reading from video
                ret, frame = cam.read()
                
                # not feasible to run with webcam on wsl
                cv2.imshow("webcam frame", ret)
                if not (cv2.waitKey(1) & 0xFF == ord(' ')):
                    # analyzing the image
                    img, checkBoxes = self.forward(frame, imgsz, upper_conf_thres, conf_thres, iou_thres, classes, agnostic_nms, max_det, line_thickness, handheld,\
                                       allDet, overlap, noPose, noElbow, cachePath, save_txt, noCheck, visualize)
                    
                    # saving to pose cache if needed
                    if saveAOI:
                        checkBoxes = [box.tolist() for box in checkBoxes]
                        poseSave[frameTrack] = checkBoxes
                        
                    if not nosave:
                        # saves image to video writer
                        out.write(img)
                else:
                    break
            if not nosave:
                out.release()
            cam.release()
              
        # youtube video analysis perform
        if isYouTube:
            # saving the youtube video
            print("downloading the youtube video")
            savePath = projectFolder            
            ydl = youtube_dl.YoutubeDL({'outtmpl': savePath + "video.mp4"})
            with ydl:
                ydl.download([source])

            # creating cacheDir
            cacheDir = os.path.join(projectFolder, "detections")
            
            # now analyzing the youtube video
            print("analyzing the youtube video")
            cam = cv2.VideoCapture(savePath + "video.mp4")
            savePath = projectFolder + "analyzedVideo.mp4"
            imageWidth = int(cam.get(3))
            imageHeight = int(cam.get(4))
            fps = cam.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            if not nosave:
                out = cv2.VideoWriter(savePath, fourcc, fps, (imageWidth, imageHeight))
            frameTrack = 0
            print(f"saving video to {savePath} with {fps} frames")
            if save_txt:
                print(f"saving results to {cacheDir}")
                os.mkdir(cacheDir)
            while True:
                # tracking the frames
                frameTrack += 1
                print(f"Frame: {frameTrack}")
                
                # creating cache path
                cachePath = os.path.join(cacheDir, str(frameTrack)) + ".txt"
                
                # reading from video
                ret, frame = cam.read()
                
                # not feasible to run with webcam on wsl
                if ret:
                    # analyzing the image
                    img, checkBoxes = self.forward(frame, imgsz, upper_conf_thres, conf_thres, iou_thres, classes, agnostic_nms, max_det, line_thickness, handheld,\
                                       allDet, overlap, noPose, noElbow, cachePath, save_txt, noCheck, visualize)
                    
                    # saving to pose cache if needed
                    if saveAOI:
                        checkBoxes = [box.tolist() for box in checkBoxes]
                        poseSave[frameTrack] = checkBoxes
                        
                    if not nosave:
                        # saves image to video writer
                        out.write(img)
                else:
                    break
            if not nosave:
                out.release()
            cam.release()
            
        # saving pose cache if needed
        if saveAOI:
            with open(os.path.join(projectFolder, "poseCache" + str(self.poseNum) + ".json"), 'w') as file:
                file.write(json.dumps(poseSave))
        
        # printing out final save path
        if isYouTube:
            print(f"Saving to {savePath}")
    
            
    """
    Takes an image as input and outputs an analyzed image for saving
    """
    @torch.no_grad()
    def forward(self,
                image,
                imgsz,
                upper_conf_thres,
                conf_thres,
                iou_thres,
                classes,
                agnostic_nms,
                max_det, 
                line_thickness,
                handheld,
                allDet,
                overlap,
                noPose,
                noElbow,
                savePath,
                saveTxt,
                noCheck,
                visualize):
        # starting to time the function execution
        t0 = time_sync()
        full = time_sync()
        
        # running preprocessing
        tensorImg, img, bgrImg = self.preprocess(image, imgsz)
        deblurTime = time_sync() - t0
           
        # keypoints found first if upper confidence threshold > 1
        if upper_conf_thres > 1:
            # collecting pose keypoints, center keypoints + distances, and which pose-estimator was used
            t1 = time_sync()
            keypoints, humans, Openpose = self.getKeyPoints(img, bgrImg, tensorImg, noElbow)
            
            # early exits if no keypoints were found
            if len(keypoints) == 0:
                print("lack of keypoints causes early exit")
                return image, []
            
            # now collects the check boxes
            checkBoxes = [self.getCropBoxes(point[0], img, INNER_RATIO, self.device, point[1], Openpose) for point in keypoints]        
            aoiTime = time_sync() - t1
            
            # runs the detection itself
            t2 = time_sync()
            pred = self.model(tensorImg, augment=self.augment, visualize=visualize)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            detTimes = time_sync() - t2
            
            # starts timing the post-processing and annotating
            t3 = time_sync()

            # process and filter the detections
            newDet, checkBoxes = self.postprocess(pred, checkBoxes, img, image, tensorImg, handheld, allDet, upper_conf_thres, overlap, line_thickness, noCheck, keypoints)
            
            # Drawing the pose-estimators if needed
            if not noPose:
                if Openpose:
                    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
                else:
                    draw_keypoints(image, img, humans, self.top_down.coco_skeletons) 
            
            # returning annotated image and caching path if needed
            if saveTxt:
                gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                print(f"Saving results to {savePath}")
                for *xyxy, conf, cls in reversed(newDet):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf)  # label format
                    with open(savePath, 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
            print(f"Full Exec.: {time_sync()-full:.3f}s | Preprocessing: {deblurTime:.3f}s | AOI Gen.: {aoiTime:.3f}s | Detection: {detTimes:.3f}s | Postprocessing {time_sync()-t3:.3f}s")
            return image, checkBoxes
        else:
            # Collecting all detections
            t1 = time_sync()
            pred = self.model(tensorImg, augment=self.augment, visualize=visualize)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            detTimes = time_sync() - t1
            
            # checking if upper-confidence threshold is not exceeded for a detection
            upperConfNotExceeded = False
            for i, det in enumerate(pred):
                for detection in det:
                    if detection[-2] < upper_conf_thres:
                        upperConfNotExceeded = True
                        break
            
            # if upper-confidence thresholds are not exceeded for a detection, we need to run AOI generation
            t2 = time_sync()
            if upperConfNotExceeded:
                # collecting keypoints
                keypoints, humans, Openpose = self.getKeyPoints(img, bgrImg, tensorImg, noElbow)
                
                # now collecting the check boxes
                checkBoxes = [self.getCropBoxes(point[0], img, INNER_RATIO, self.device, point[1], Openpose) for point in keypoints]        
            else:
                # zero areas of interest need to be collected in this case
                checkBoxes = []
                keypoints = []
            aoiTime = time_sync() - t2
            
            # starts timing the post-processing and annotating
            t3 = time_sync()

            # finally processing and filtering the detections
            newDet, checkBoxes = self.postprocess(pred, checkBoxes, img, image, tensorImg, handheld, allDet, upper_conf_thres, overlap, line_thickness, noCheck, keypoints)
            
            # Drawing the pose-estimators if needed
            if not noPose and upperConfNotExceeded:
                if Openpose:
                    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
                else:
                    draw_keypoints(image, img, humans, self.top_down.coco_skeletons) 
            
            # Returning the annotated image and caching path if needed
            if saveTxt:
                gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                print(f"Saving results to {savePath}")
                for *xyxy, conf, cls in reversed(newDet):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf)  # label format
                    with open(savePath, 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
            print(f"Full Exec.: {time_sync()-full:.3f}s | Preprocessing: {deblurTime:.3f}s | Detection: {detTimes:.3f}s | AOI Gen.: {aoiTime:.3f}s | Postprocessing: {time_sync()-t3:.3f}")
            return image, checkBoxes
    
    
    """
    This function filters out low confidence detections using areas of interest
    """
    def filterDet(self, 
                  checkBoxes,
                  det,
                  im0,
                  img, 
                  tensorImg,
                  upper_conf_thres,
                  overlap):
        # if there are detections, then starts filtering them
        newDet = []
        if len(det):
            # Check if any overlap between keypoint and checkBoxes (means object is handheld)
            for detection in det:
                # if upper confidence threshold is satisfied, detection is accepted
                if detection[-2] >= upper_conf_thres:
                    newDet.append(detection)
                    continue
                # if low confidence detection, then checks if it properly overlaps with area of interest
                for check in checkBoxes:
                    # checking if detection sufficiently overlaps with area of interest
                    if bbox_iou(detection, check) > 0 and bbox_overlap(detection, check) > overlap:
                        # checking if detection is not much larger than said area of interest
                        maxWidth = max(detection[3]-detection[1], detection[2]-detection[0])
                        maxCropW = max(check[3]-check[1], check[2]-check[0])
                        if maxWidth/maxCropW <= 2.5:
                            newDet.append(detection)
                            break
            
        # returning the filtered detections
        return newDet
                
        
    """
    This function collects the crop boxes given a point and some other inputs
    """
    def getCropBoxes(self, 
                     point, 
                     img, 
                     factor, 
                     device, 
                     cropWidth, 
                     Openpose):
        if Openpose:
            cropWidth *= factor
            pointX = round(img.shape[1] * point.x)
            pointY = round(img.shape[0] * point.y)
            lowX = pointX - cropWidth
            upX = pointX + cropWidth
            lowY = pointY - cropWidth
            upY = pointY + cropWidth
            box = [lowX, lowY, upX, upY, 0, 0]
            return box
        else:
            cropWidth *= factor
            pointX = point[0]
            pointY = point[1]
            lowX = pointX - cropWidth
            upX = pointX + cropWidth
            lowY = pointY - cropWidth
            upY = pointY + cropWidth
            box = [lowX, lowY, upX, upY, 0, 0]
            return box
    
    
    """
    This function collects the keypoints of the individual which are referred to when making the detections
    """
    def getKeyPoints(self,
                     img,
                     bgrImg,
                     tensorImg,
                     noElbow):
        # running only bottom up pose-estimator in this situation
        if self.poseNum == 0:
            return self.bottomUp(bgrImg, noElbow)
        else:
            # otherwise determining the number of people present in the image
            pred = self.top_down.det_model(tensorImg)[0]
            pred = non_max_suppression(pred, self.top_down.conf_thres, self.top_down.iou_thres, classes=0)
            
            # determining pose-estimator ran based on number of people and poseNum threshold
            for det in pred:
                pplCount = len(det)
                
                # if no humans detected then early exits
                if pplCount == 0:
                    return [], [], False
                
                # using pplCount and poseNum to determine pose-estimator to use
                if self.poseNum == -1 or pplCount <= self.poseNum:
                    return self.topDown(img, tensorImg, det)
                else:
                    return self.bottomUp(bgrImg, noElbow)
    
    
    """
    Collecting the top-down pose-estimator keypoints
    """
    def topDown(self,
                img,
                tensorImg,
                det):
        # scaling and centering bounding boxes then predicting poses
        boxes = scale_boxes(det[:, :4], img.shape[:2], tensorImg.shape[-2:]).cpu()
        boxes = self.top_down.box_to_center_scale(boxes)
        outputs = self.top_down.predict_poses(boxes, img)
        
        # collecting final predictions for poses
        if 'simdr' in self.top_down.model_name:
            coords = get_simdr_final_preds(*outputs, boxes, self.top_down.patch_size)
        else:
            coords = get_final_preds(outputs, boxes)    
        humans = coords
        
        # if no predictions could be found, then returns no keypoints
        if humans is None:
            return [], [], False
        
        # collects all keypoints and returns
        KP = []
        cropWidth = -1
        imType = 'h'
        for human in humans:
            headWidth = distGet(human[3], human[4])
            if headWidth > max([distGet(human[0], human[3]), distGet(human[0], human[4])]):
                cropWidth = headWidth
                imType = 'h'
            else:
                arms = []
                arms.append(distGet(human[6], human[8]))
                arms.append(distGet(human[8], human[10]))
                arms.append(distGet(human[5], human[7]))
                arms.append(distGet(human[7], human[9]))
                arms.sort(reverse=True)
                cropWidth = 0.667 * arms[0]
                imType = 'a'
            KP.append([human[9], cropWidth, imType])
            KP.append([human[10], cropWidth, imType])
            
        return KP, humans, False
    
    
    """
    Collecting the bottom-up pose-estimator keypoints using bgr image
    """
    def bottomUp(self,
                 img,
                 noElbow):
        # Running inference and calculating max dimension
        w = img.shape[1]
        h = img.shape[0]
        if w >= h:
            m = w
        else:
            m = h
        humans = self.bottom_up.inference(img, scales=[None])

        # Getting keypoints
        KP = []
        cropWidth = -1
        imType = ""
        upperLim = 6
        for human in humans:
            parts = human.body_parts
            #if not (parts.get(4) or parts.get(7)):
            if not (parts.get(4) or parts.get(7) or (not noElbow and parts.get(3)) or (not noElbow and parts.get(6))):
                continue
            
            head = human.get_face_box(w, h)
            if type(head) == dict and head["w"] <= m/upperLim and parts.get(16) != None and parts.get(0) != None and parts.get(17) != None and max([distGet((parts[16].x * w, parts[16].y * h), (parts[0].x * w, parts[0].y * h)), \
                                                                distGet((parts[0].x * w, parts[0].y * h), (parts[17].x * w, parts[17].y * h))]) < distGet((parts[16].x * w, parts[16].y * h), (parts[17].x * w, parts[17].y * h)):
                cropWidth = head["w"]
                imType = "h"
            else:
                # getting all arm lengths possible
                arms = []
                if parts.get(7) and parts.get(6):
                    arms.append(distGet((parts[6].x * w, parts[6].y * h), (parts[7].x * w, parts[7].y * h)))
                if parts.get(3) and parts.get(4):
                    arms.append(distGet((parts[3].x * w, parts[3].y * h), (parts[4].x * w, parts[4].y * h)))
                if parts.get(2) and parts.get(3):
                    arms.append(distGet((parts[2].x * w, parts[2].y * h), (parts[3].x * w, parts[3].y * h)))
                if parts.get(5) and parts.get(7):
                    arms.append(distGet((parts[5].x * w, parts[5].y * h), (parts[7].x * w, parts[7].y * h)))
                arms.sort(reverse=True)
                
                # Returning max arm length
                for armLength in arms:
                    if armLength * 0.667 <= m/upperLim:
                        cropWidth = armLength
                        imType = "a"
                        break
                
                # If can't get anything else then just uses upper body bounding box
                if cropWidth > 0:
                    cropWidth *= 0.667
                else:
                    # moving onto upper bounding box
                    upperBody = human.get_upper_body_box(w, h)
                    if type(upperBody) == dict and (upperBody["w"] / 2) <= m/upperLim:
                        cropWidth = upperBody["w"]/2
                        imType = "u"
                    else:
                        cropWidth = m/upperLim
                        imType = "f"
                        
             # Searching for wrists and settling for elbows if needed and allowed
            if parts.get(7):
                KP.append([parts[7], cropWidth, imType])
            elif not noElbow and parts.get(6):
                KP.append([parts[6], cropWidth, imType])
            if parts.get(4):
                KP.append([parts[4], cropWidth, imType])
            elif not noElbow and parts.get(3):
                KP.append([parts[3], cropWidth, imType])
            
        return KP, humans, True


    """
    Function to scale the check boxes for filtering and drawing use
    """
    def readyCheckBoxes(self, 
                        checkBoxes, 
                        img, 
                        image, 
                        device):
        for i in range(0, len(checkBoxes)):
            checkBoxes[i][0] = round(checkBoxes[i][0]/img.shape[1] * image.shape[1])
            checkBoxes[i][2] = round(checkBoxes[i][2]/img.shape[1] * image.shape[1])
            checkBoxes[i][1] = round(checkBoxes[i][1]/img.shape[0] * image.shape[0])
            checkBoxes[i][3] = round(checkBoxes[i][3]/img.shape[0] * image.shape[0])
        return [torch.Tensor(box).to(self.device) for box in checkBoxes if box[3]-box[1] > 0 and box[2]-box[0] > 0]  


    """
    This function runs all of the logical post-processing of the detections and draws them onto the image if needed
    """
    def postprocess(self, pred, checkBoxes, img, image, tensorImg, handheld, allDet, upper_conf_thres, overlap, line_thickness, noCheck, keypoints):
        # process and filter the detections
        for i, det in enumerate(pred):  # detections per image
            # ensuring check boxes and detections are ready for comparison and rescaled
            checkBoxes = self.readyCheckBoxes(checkBoxes, img, image, self.device)  
            newDet = []
            if len(det):
                # Rescale boxes from img_size to image size and same thing done for check boxes
                det[:, :4] = scale_coords(tensorImg.shape[2:], det[:, :4], image.shape).round()
        
                # filtering out non-handhelds if needed
                det = det if not handheld else [detection for detection in det if int(detection[5]) in HANDHELD_MAP]
            
                # filtering low confidence detections with areas of interest if allDet is disabled
                if allDet:
                    newDet = det
                else:
                    newDet = self.filterDet(checkBoxes, det, image, img, tensorImg, upper_conf_thres, overlap)


            # creating the annotation tool
            annotator = Annotator(image, line_width=line_thickness, example=str(self.names))
            
            # drawing out the desired detections
            for *xyxy, conf, cls in reversed(newDet):
                c = int(cls)  # integer class
                label = f'{self.names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
        
            # drawing out the check boxes and pose-estimators if noCheck disabled
            if not noCheck:
                i = 0
                for *xyxy, conf, cls in reversed(checkBoxes):
                    c = int(cls)
                    if xyxy[3] - xyxy[1] > 0 and xyxy[2] - xyxy[0] > 0:
                        #save_one_box(xyxy, imc, file=save_dir/ 'wrist_crops' / names[c] / f'{p.stem}.jpg', BGR=True, pad=0)
                        annotator.box_label(xyxy, keypoints[i][-1], color=colors(c, True))
                    i += 1

        return newDet, checkBoxes

    """
    This function takes an image and returns a tensor version ready for YOLOv5 and top-down pose-estimation.
    Depending on the settings, it may also deblur the image.
    """
    def preprocess(self,
                   image,
                   imgsz):
        
        # letterboxing the image
        img = letterbox(image, imgsz[0], stride=self.stride)[0]
        
        # BGR to RGB transitioning and storing bgrImg
        bgrImg = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # running deblurring if noDeblur is off
        if not self.noDeblur:
            img = self.predictor(img, None)
            # recreating bgr image for bottom-up pose-estimator if deblurring was done
            bgrImg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # transposing image
        Timg = np.ascontiguousarray(img.transpose((2, 0, 1)))
            
        # creating the torch tensor
        Timg = torch.from_numpy(Timg).to(self.device)    
        Timg = Timg.half() if self.half else Timg.float()
        Timg = Timg / 255.0
        if len(Timg.shape) == 3:
            Timg = Timg[None]
        return Timg, img, bgrImg 
        

# runs inference with desired options
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # General purpose + pipeline control flow arguments
    parser.add_argument('--source', type=str, default='./yolov5/data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--project', default='./runs/', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')    
    # imgsz will be shared between the top-down detection model and the yolov5 object detector
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--upper-conf-thres', type=float, default=1.1, help='confidence threshold at which pipeline won\'t be applied')
    parser.add_argument('--wsl', default=False, action='store_true', help='if wsl is used then image not shown')
    parser.add_argument('--noDeblur', default=False, action='store_true', help='option for disabling deblur')
    parser.add_argument('--noElbow', default=False, action='store_true', help='option for disabling elbow check')
    parser.add_argument('--noPose', default=False, action='store_true', help='option for not showing pose')
    parser.add_argument('--noCheck', default=False, action='store_true', help='option for not showing check boxes')
    parser.add_argument('--allDet', default=False, action='store_true', help='option for showing all detections')
    parser.add_argument('--poseNum', default=3, type=int, help='number of humans to swtich pose detections') 
    parser.add_argument('--overlap', default=0.25, type=float, help='amount of check overlap needed for check boxes') 
    parser.add_argument('--handheld', default=False, action='store_true', help='if wsl is used then image not shown')
    parser.add_argument('--saveAOI', default=False, action='store_true', help='if wsl is used then image not shown')

    # Arguments for creating the yolov5 object detector
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5/yolov5s.pt', help='model path(s)')
    parser.add_argument('--data', type=str, default='./yolov5/data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    
    # Arguments for creating the bottom-up pose-estimator (if it needs to be created)
    parser.add_argument('--model', type=str, default='./tf-pose-estimation/models/mobilenet_thin_432x368', help='cmu_640x480 / cmu_640x360 / mobilenet_thin_432x368')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    
    # Arguments for creating the top-down pose-estimator (if it needs to be created)
    parser.add_argument('--det-model', type=str, default='yolov5/crowdhuman_yolov5m.pt')
    parser.add_argument('--pose-model', type=str, default='checkpoints/pretrained/simdr_hrnet_w32_256x192.pth')
    parser.add_argument('--person-conf-thres', type=float, default=0.4)
    parser.add_argument('--person-iou-thres', type=float, default=0.45)
    
    # Arguments for creating the deblurrer (if it needs to be made)
    parser.add_argument('--weights_path', type=str, default='best_fpn.h5')

    # Getting parser arguments and calling the function
    opt = parser.parse_args()

    shop = SHOP(opt.noDeblur, opt.poseNum, opt.weights, opt.data, opt.device, opt.half,\
                opt.dnn, opt.model, opt.scales, opt.det_model, opt.pose_model, opt.weights_path,\
                opt.person_iou_thres, opt.person_conf_thres, opt.imgsz, opt.augment)
    testOut = shop.infer(opt.source, opt.project, opt.name, opt.conf_thres, opt.upper_conf_thres,\
                         opt.iou_thres, opt.noElbow, opt.noPose, opt.allDet, opt.overlap, opt.save_txt,\
                         opt.nosave, opt.classes, opt.agnostic_nms, opt.visualize, opt.line_thickness,\
                         opt.wsl, opt.imgsz, opt.max_det, opt.handheld, opt.noCheck, opt.saveAOI)