import os
import sys
import re
import glob
import time
import numpy as np
from tqdm import tqdm
import base64
import json
import cv2


def vis_ann(vis_list, html_file, pid2cate, bad_set):
    html_file_fp = open(html_file, 'w')
    html_file_fp.write('<html>\n<body>\n')
    html_file_fp.write('<meta charset="utf-8">\n')

    for i, items in enumerate(vis_list):
        if i % 10 == 0:
            html_file_fp.write('<p>\n')
            html_file_fp.write('<table border="0" align="center">\n')
            html_file_fp.write('<tr><td height="10" colspan="100"><hr></td></tr>\n')
            html_file_fp.write('<tr>\n')

        bbox_list = []
        category_list = []
        matched_scores_list = []
        matched_entity_list = []
        for item in items:
            item_id = item['item_id']
            bbox = item['bbox']
            bbox_list.append(bbox)
            category = item['category']
            category_list.append(category)
            matched_scores = item['matched_scores']
            matched_entity = item['matched_entity']
            matched_scores_list.append(matched_scores)
            matched_entity_list.append(matched_entity)
            raw_text = item['raw_text']
            #candidate = item['candidate']
            candidate = ''

        #root = "/mnt/longvideo/zhonghuasong/retreival_workshop/projects/shangyehua/pair/images"
        #root = "/mnt/longvideo/zhonghuasong/retreival_workshop/projects/shangyehua/images"
        #img_path = os.path.join(root, str(int(item_id) % 100), item_id + ".jpg")
        #image = cv2.imread(img_path)
        #for bbox in bbox_list:
        #    xmin, ymin, xmax, ymax = bbox
        #    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
        #cv2.imwrite(img_path + "_".join([str(b) for b in bbox]) + ".jpg", image)
        #img_path = img_path + "_".join([str(b) for b in bbox]) + ".jpg"
        #print(img_path)

        if item_id not in pid2cate or pid2cate[item_id] not in bad_set:
            continue

        img_path = os.path.join("/mnt/longvideo/zhonghuasong/retreival_workshop/projects/shangyehua/evaluate/mAP/output/images", item_id + ".jpg")
        if not os.path.exists(img_path):
            print("not exist: ", img_path)
            continue

        color = 'white'
        html_file_fp.write(
            """
            <td bgcolor=%s align='center'>
                <img width="224" height="224" src="data:image/jpeg;base64, %s">
                <br> item id: %s
                <br> title: %s
                <br> category: %s
                <br> candidate: %s
                <br> 主体: %s
                <br> score: %s
                <br> 主体位置: %s
            </td>
            """ % (color, base64.b64encode(open(img_path, 'rb').read()).decode(), item_id, raw_text, category_list, candidate, matched_entity_list, matched_scores_list, bbox_list)
        )

        if (i + 1) % 10 == 0:
            html_file_fp.write('</tr>\n')
            html_file_fp.write('<tr><td height="10" colspan="100"><hr></td></tr>\n')
            html_file_fp.write('</table>\n')
            html_file_fp.write('</p>\n')
    html_file_fp.write('</body>\n</html>')


def main(input_path, output_path):
    pid2cate = {}
    with open("/mnt/longvideo/zhonghuasong/retreival_workshop/projects/shangyehua/small.csv") as fr:
        lines = fr.readlines()
        for line in tqdm(lines):
            try:
                path, text = line.strip().split("\t")
            except Exception as e:
                print(line)
                continue
            pid = path.split("/")[-1].split(".")[0]
            pid2cate[pid] = text.split("###")[-1].replace("/", "-")
            print(pid, pid2cate[pid])
    bad_set = set()
    with open("/mnt/longvideo/zhonghuasong/retreival_workshop/projects/shangyehua/evaluate/mAP/output/bad_cate.list") as fr:
        lines = fr.readlines()
        for line in tqdm(lines):
            bad_set.add(line.strip())

    vis_list = []
    with open(input_path) as fr:
        lines = fr.readlines()
        for line in tqdm(lines):
            info = json.loads(line.strip())
            info_dict = info
            vis_list.append(info_dict)
    vis_ann(vis_list, output_path, pid2cate, bad_set)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('usage: python tools.py func', file=sys.stderr)
