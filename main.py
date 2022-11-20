import cv2
import numpy as np
import time
import os
import sys
import subprocess
import threading
import random
import math

# 設定熱區
# TODO: DEBUGGER 可否共用 
videoCaptureObject = cv2.VideoCapture(0)

prev_zone1_fg_mask = None
prev_zone2_fg_mask = None

# 習近平
zone1 = {
    'upper_left':(50, 350),
    'bottom_right':(350, 550)
}

# TODO: 根據使用者宜木大小顯示
# 拍照
zone2 = {
    'upper_left':(950, 350),
    'bottom_right':(1250, 550)
}

last_taken_img = "xjp.jpg"
backSub = cv2.createBackgroundSubtractorMOG2() 
while(True):
    # 讀取影像
    ret,image_frame = videoCaptureObject.read()
    # 轉換成灰階
    # gray_img = cv2.cvtColor(image_frame, cv2.COLOR_RGB2BGR)

    # 熱區長方形
    r1 = cv2.rectangle(image_frame, zone1['upper_left'], zone1['bottom_right'], (100, 50, 200), 2)
    r2 = cv2.rectangle(image_frame, zone2['upper_left'], zone2['bottom_right'], (100, 50, 200), 2)
    cv2.putText(image_frame, "注意：此程式有可能會拍下大量前鏡頭照片", (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    cv2.putText(image_frame, "show_last_take_picture", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    cv2.putText(image_frame, "take_picture", (950, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    imgae_frame_clone = image_frame.copy()
    # 取得前台遮罩
    fg_image_frame = backSub.apply(imgae_frame_clone)
    # fg_image_frame = cv2.cvtColor(fg_image_frame, cv2.COLOR_BGR2RGB)


    ## ZONE1
    # 取得熱區去背景後的mask
    temp_zone1_fg_mask = fg_image_frame[zone1['upper_left'][1] : zone1['bottom_right'][1], zone1['upper_left'][0] : zone1['bottom_right'][0]]
    zone1_fg_mask = temp_zone1_fg_mask

    if prev_zone1_fg_mask is None:
        prev_zone1_fg_mask = zone1_fg_mask
        continue

    # 比較單前mask跟前一個mask的差異
    # 計算差異的面積
    absdiff = cv2.absdiff(zone1_fg_mask, prev_zone1_fg_mask)
    diff_area = np.sum(absdiff) / 255
    print(diff_area)
    # thresholdDiff = cv2.threshold(absdiff, 25, 255, cv2.THRESH_BINARY)[1]
    # diff = cv2.dilate(thresholdDiff, None, iterations=2)
    # diff_area = cv2.countNonZero(diff)
    # 判斷差異是否超過一定值
    if diff_area > 1000:
        # print("偵測到大量差異")
        img = cv2.imread(last_taken_img)
        if img is None:
            print("沒有上一張照片")
            img = cv2.imread("xjp.jpg")
            continue
        img_shape = img.shape
        # print("img_shape",img_shape)
        # x_start = int(img_shape[1] / 2)
        # image_frame = img

        # 將img顯示在imagefreame上
        image_frame[0:img_shape[0],0:img_shape[1]] = img
        # cv2.imshow('image',img)

    prev_zone1_fg_mask=zone1_fg_mask

    # ## ZONE2
    # # 取得熱區去背景後的mask
    temp_zone2_fg_mask = fg_image_frame[zone2['upper_left'][1] : zone2['bottom_right'][1], zone2['upper_left'][0] : zone2['bottom_right'][0]]
    zone2_fg_mask = temp_zone2_fg_mask

    if prev_zone2_fg_mask is None:
        prev_zone2_fg_mask = zone2_fg_mask
        continue

    absdiff = cv2.absdiff(zone2_fg_mask, prev_zone2_fg_mask)
    diff_area = np.sum(absdiff) / 255
    print(diff_area)
    # # 判斷差異是否超過一定值
    if diff_area > 1000:
        # 取得現在時間 YYYYMMDDHHMMSS
        now = time.strftime("%Y%m%d%H%M%S")
        new_img_name = now + ".jpg"
        camera_frame_shape = image_frame.shape
        new_image_shape = (int(camera_frame_shape[1]/2),int(camera_frame_shape[0]/2))
        last_taken_img=new_img_name
        new_image_frame = image_frame
        new_image = cv2.resize(new_image_frame, new_image_shape, interpolation = cv2.INTER_AREA)
        cv2.imwrite(new_img_name,new_image)
        cv2.putText(image_frame, f"{new_img_name} saved!", (950, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))


    prev_zone2_fg_mask=zone2_fg_mask

    cv2.imshow("HW4", image_frame)
    cv2.imshow("fg_mask", fg_image_frame)
    k = cv2.waitKey(1)
    if k%256 == 27:
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        cv2.imwrite("input_image.jpg",image_frame)
        img = cv2.imread("input_image.jpg")

        # you can do those function inside the sketch_transform def

        #gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        #gray_img = cv2.resize(gray_image, (28, 28)).reshape(1,28,28,1)
        #ax[2,1].imshow(gray_img.reshape(28, 28) , cmap = "gray")
        #cv2.imshow("image", gray_img.reshape(28, 28))

        # y_pred =  model.predict_classes(img)
        # print("predicted alphabet  = ", y_pred) 
        #text_to_audio(myDict.get(y_pred[0]))   
videoCaptureObject.release()
cv2.destroyAllWindows()