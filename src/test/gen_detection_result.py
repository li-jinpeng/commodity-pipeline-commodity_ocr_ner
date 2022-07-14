import os
from tqdm import tqdm
import json


pid2cate = {}
with open("/mnt/longvideo/zhonghuasong/retreival_workshop/projects/shangyehua/target_object_detection.csv") as fr:
    lines = fr.readlines()
    for line in tqdm(lines):
        values = line.strip().split(",")
        pid = values[0]
        cate = values[-1].replace("/", "-")
        pid2cate[pid] = cate

def gen(input_path, output_dir):
    with open(input_path) as fr:
        lines = fr.readlines()
        output_list = []
        for line in tqdm(lines):
            items = json.loads(line.strip())
            tmp = []
            for item_info in items:
                item_id = item_info['item_id']
                bbox = item_info['bbox']
                score = max(item_info['matched_scores'])
                tmp.append(" ".join([item_id] + [str(score)] + [str(x) for x in bbox]))
            output_list.append(tmp)

    for out in output_list:
        if len(out) == 0:
            continue
        fw = open(os.path.join(output_dir, out[0].split(" ")[0] + ".txt"), 'w')
        for line in out:
            pid = line.split(" ")[0]
            cate = pid2cate[pid]
            fw.write(" ".join([cate] + line.split(" ")[1:]) + "\n")
        fw.close()

input_path = 'test_output.list'
output_dir = '/mnt/longvideo/zhonghuasong/retreival_workshop/projects/shangyehua/evaluate/mAP/our_input/detection-results'
gen(input_path, output_dir)
