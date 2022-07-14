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


def vis_ann(vis_list, html_file):
    html_file_fp = open(html_file, 'w')
    html_file_fp.write('<html>\n<body>\n')
    html_file_fp.write('<meta charset="utf-8">\n')

    for i, items in enumerate(vis_list):
        try:
            if i % 5 == 0:
                html_file_fp.write('<p>\n')
                html_file_fp.write('<table border="0" align="center">\n')
                html_file_fp.write('<tr><td height="10" colspan="100"><hr></td></tr>\n')
                html_file_fp.write('<tr>\n')

            item_id = items[0]['item_id']
            raw_text = items[0]['raw_text']

            candidate_text = ['厨具/杯具', '古玩收藏', '宝石', '手工艺品/民俗', '木雕盘玩', '玉石', '金银饰', '饰品/流行首饰/时尚饰品']
            flag = True
            for candi in candidate_text:
                if candi in raw_text:
                    flag = False
                    break
            if flag:
                continue

            bbox_list = []
            category_list = []
            matched_scores_list = []
            matched_entity_list = []
            root = "/mnt/longvideo/zhonghuasong/retreival_workshop/projects/shangyehua/shandiangou_detection/images"
            img_path = os.path.join(root, str(int(item_id) % 100), item_id + ".jpg")
            image = cv2.imread(img_path)
            for item in items:
                bbox = item['bbox']
                bbox_list.append(bbox)
                category_list.append(item['category'])
                matched_scores_list.append(item['matched_scores'])
                matched_entity_list.append(item['matched_entity'])
                xmin, ymin, xmax, ymax = bbox
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
            cv2.imwrite(img_path + "_".join([str(b) for b in bbox]) + ".jpg", image)
            img_path = img_path + "_".join([str(b) for b in bbox]) + ".jpg"
            print(img_path)

            color = 'white'
            html_file_fp.write(
                """
                <td bgcolor=%s align='center'>
                    <img width="224" height="224" src="data:image/jpeg;base64, %s">
                    <br> item id: %s
                    <br> title: %s
                    <br> match entity word: %s
                    <br> match category: %s
                    <br> match score: %s
                    <br> match pos: %s
                </td>
                """ % (color, base64.b64encode(open(img_path, 'rb').read()).decode(), item_id, raw_text, matched_entity_list, category_list, matched_scores_list, bbox_list)
            )

            # gt
            root = "/mnt/longvideo/zhonghuasong/retreival_workshop/projects/shangyehua/shandiangou_detection/images"
            img_path = os.path.join(root, str(int(item_id) % 100), item_id + ".jpg")
            image = cv2.imread(img_path)

            root = "/mnt/longvideo/zhonghuasong/retreival_workshop/projects/shangyehua/shandiangou_detection/gt/gt_all_bbox/"
            gt_path = os.path.join(root, item_id + ".txt")
            with open(gt_path) as fr:
                for line in fr.readlines():
                    values = line.strip().split(" ")
                    if len(values) != 5:
                        continue
                    xmin, ymin, xmax, ymax = int(values[1]), int(values[2]), int(values[3]), int(values[4])
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
            cv2.imwrite(img_path + "_gt"+ ".jpg", image)
            img_path = img_path + "_gt" + ".jpg"
            print(img_path)
            html_file_fp.write(
                """
                <td bgcolor=%s align='center'>
                    <img width="224" height="224" src="data:image/jpeg;base64, %s">
                    <br> item id: %s
                    <br> ground truth
                </td>
                """ % (color, base64.b64encode(open(img_path, 'rb').read()).decode(), item_id)
            )


            if (i + 1) % 5 == 0:
                html_file_fp.write('</tr>\n')
                html_file_fp.write('<tr><td height="10" colspan="100"><hr></td></tr>\n')
                html_file_fp.write('</table>\n')
                html_file_fp.write('</p>\n')
        except Exception as e:
            print(e)
            continue
    html_file_fp.write('</body>\n</html>')


def main(input_path, output_path):
    vis_list = []
    with open(input_path) as fr:
        lines = fr.readlines()
        for line in tqdm(lines):
            info = json.loads(line.strip())
            info_dict = info
            vis_list.append(info_dict)
    vis_ann(vis_list, output_path)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('usage: python tools.py func', file=sys.stderr)
