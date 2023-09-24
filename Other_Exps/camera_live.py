import cv2
import numpy as np
import time
import pandas as pd

def iou_calc(true_bbox , selectivesarch_bbox):
    true_xmin, true_ymin, true_width, true_height  = true_bbox
    bb_xmin, bb_ymin,  bb_width, bb_height = selectivesarch_bbox

    true_xmax = true_xmin + true_width
    true_ymax = true_ymin + true_height
    bb_xmax = bb_xmin + bb_width
    bb_ymax = bb_ymin + bb_height

    #calculating area
    true_area = true_width * true_height
    bb_area   = bb_width * bb_height 

    #calculating itersection cordinates
    inter_xmin = max(true_xmin , bb_xmin) 
    inter_ymin = max(true_ymin , bb_ymin)
    inter_xmax = min(true_xmax , bb_xmax)
    inter_ymax = min(true_ymax , bb_ymax)

    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        iou = 0


    else:
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)


        iou = inter_area / (true_area + bb_area - inter_area)
        
    assert iou<=1
    assert iou>=0
    
    return iou


cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


def iou_filter(image_path,img_name,true_bb,thresh=0.5):
    print(image_path)
    img_bb = true_bb[true_bb['filename']==img_name].reset_index(drop=True)
    img = cv2.imread(image_path)
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    ss_bb = rects[:2000]
    filtered_selective_search = []
    negative_examples = []
    maybe_neagative = []
    
    # loop to compute iou for all label of perticular image
    for label in range(len(img_bb)):
        

        #unpacking cordinates
        true_xmin, true_ymin, true_width, true_height  = img_bb.loc[label,'xmin'], img_bb.loc[label,'ymin'], img_bb.loc[label,'xmax']-img_bb.loc[label,'xmin'], img_bb.loc[label,'ymax']-img_bb.loc[label,'ymin']
        class_of_label = img_bb.loc[label,'class']
        
        #loop to compute iou for all selective search of perticular label
        for j,rect in enumerate(ss_bb):
            calculating_iou_for_selectivesearch = iou_calc([true_xmin, true_ymin, true_width, true_height],rect)
            
            if calculating_iou_for_selectivesearch > thresh:
                filtered_selective_search.append([list(rect),class_of_label])
            
            elif calculating_iou_for_selectivesearch <0.5:
                maybe_neagative.append(list(rect))
    
    #removing duplicate entries
    
    def Remove(duplicate): 
        final_list = [] 
        for num in duplicate: 
            if num not in final_list: 
                final_list.append(num) 
        return final_list 

    maybe_neagative = Remove(maybe_neagative)
    filtered_selective_search = Remove(filtered_selective_search)
   

    #this is will use for background class for CNN which has iou less than 0.2, In paper it's 0.3 but in that also written that it's depends on dataset. 

    only_labels_of_filtered_selective_search = [x[0] for x in filtered_selective_search]

    for lab in maybe_neagative:
        condition = []    
        for true_lab in only_labels_of_filtered_selective_search:
            
            iou_for_negative_ex = iou_calc(true_lab,lab)
            
            condition.append(True) if iou_for_negative_ex <= 0.5  else condition.append(False)

        if False not in condition:
            negative_examples.append(lab)
    
    negative_examples = Remove(negative_examples)
    random_background_images_index = np.random.randint(low=0, high=len(negative_examples), size=2*len(only_labels_of_filtered_selective_search)) 
    random_background_images = [negative_examples[x] for x in random_background_images_index]

    
    return filtered_selective_search , Remove(random_background_images)





# Set the desired frames per second (fps)
desired_fps = 15
frame_delay = 1.0 / desired_fps  # Calculate the delay time in seconds

cap = cv2.VideoCapture(0)
hand_cascade = cv2.CascadeClassifier('hand.xml')

while True:
    start_time = time.time()  # Record the start time of processing
    
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hands = hand_cascade.detectMultiScale(gray, 1.1, 5)

    for (x,y,w,h) in hands:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    
    if not ret:
        break
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
    # Calculate the time elapsed since the start of processing
    elapsed_time = time.time() - start_time
    
    # Introduce a delay to achieve the desired frame rate
    if elapsed_time < frame_delay:
        time.sleep(frame_delay - elapsed_time)

cap.release()
cv2.destroyAllWindows()