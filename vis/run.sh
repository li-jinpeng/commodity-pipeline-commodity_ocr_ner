#python vis.py main /mnt/longvideo/zhonghuasong/retreival_workshop/service/commodity-pipeline/src/test/output.list /mnt/longvideo/zhonghuasong/retreival_workshop/service/commodity-pipeline/src/test/output.html

#python vis_debug.py main \
#  /mnt/longvideo/zhonghuasong/retreival_workshop/service/commodity-pipeline/src/test/test_output.list \
#  /mnt/longvideo/zhonghuasong/retreival_workshop/service/commodity-pipeline/src/test/test_output_debug.html

#python vis_shandiangou.py main \
#  /mnt/longvideo/zhonghuasong/retreival_workshop/projects/shangyehua/shandiangou_detection/intention_results_doudi/test_output.list \
#  /mnt/longvideo/zhonghuasong/retreival_workshop/projects/shangyehua/shandiangou_detection/intention_results_doudi/test_output_debug_det_thres0.2_match_thres0.34_main_badcase.html

python vis_not_shandiangou.py main \
  /mnt/longvideo/zhonghuasong/retreival_workshop/projects/shangyehua/not_shandiangou_detection/intention_results_doudi/test_output.list \
  /mnt/longvideo/zhonghuasong/retreival_workshop/projects/shangyehua/not_shandiangou_detection/intention_results_doudi/test_output_debug_det_thres0.2_match_thres0.34_main_badcase.html
