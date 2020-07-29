#TODO
    - coco, ytb-vos mask dataset deal for siampolar
    - add multi-forward train style to avoid batchsize is too small
    - add siampolar_tracker to pysot/tracker refer to siamcar_tracker
    - IOU-aware for poalr head
    - Deep snake for more precise mask result
    - refer to Ocean, DIMP, ATOM , add Online Update Module
    - predict the search Region
    - template update machine 

#  Under Way
    siampolar tracker
    
# 2020-7-28
    - add OTB100, UAV123, LASOT for test_dataset
    - add multidepthwise correlation as Module to pysot/models/fusion
    - add dwconv to pysot/models/fusion

# 2020-7-27
    - complete Siampolar Tracker model, dataset, loss
    - add polar mask head
    - add siampolar dataset deal (bbox -> ecllipse for mask)
    - add siampolar
    - refine focal loss with copy from  

# 2020-7-20
    - add GOT10k, LaSOT  to training dataset
    - non-local module refer to Alpha refine
    - add pixelwise correlation to pysot/core/xcorr.py refer to Alpha refine
      
# 2020-7-15 
    - add feature combination Module to pysot/model/fusion
    - add multidepthwise correlation
    - add weight depthwise correlation
    
# 2020-6-16
    - add SiamFCOS to SiamTrackers/
    - add fcos head to pysot/models/head/fcos.py
    - add iou loss, giou loss, diou loss, ciou loss to pysot/loss
    - add SiamCAR Tracker to pysot/tracker, CAR head to pysot/models/fcos , 