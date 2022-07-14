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
        if i % 10 == 0:
            html_file_fp.write('<p>\n')
            html_file_fp.write('<table border="0" align="center">\n')
            html_file_fp.write('<tr><td height="10" colspan="100"><hr></td></tr>\n')
            html_file_fp.write('<tr>\n')

        item_id = items['item_id']
        bbox = items['bbox']
        category = items['category']
        matched_scores = items['matched_scores']
        matched_entity = items['matched_entity']
        raw_text = items['raw_text']
        #root = "/mnt/longvideo/zhonghuasong/retreival_workshop/projects/shangyehua/pair/images"
        root = "/mnt/longvideo/zhonghuasong/retreival_workshop/projects/shangyehua/images"
        img_path = os.path.join(root, str(int(item_id) % 100), item_id + ".jpg")

        image = cv2.imread(img_path)
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
                <br> category: %s
                <br> 主体: %s
                <br> score: %s
                <br> 主体位置: %s
            </td>
            """ % (color, base64.b64encode(open(img_path, 'rb').read()).decode(), item_id, raw_text, category, matched_entity, matched_scores, bbox)
        )

        if (i + 1) % 10 == 0:
            html_file_fp.write('</tr>\n')
            html_file_fp.write('<tr><td height="10" colspan="100"><hr></td></tr>\n')
            html_file_fp.write('</table>\n')
            html_file_fp.write('</p>\n')
    html_file_fp.write('</body>\n</html>')


def main(input_path, output_path):
    vis_list = []
    with open(input_path) as fr:
        lines = fr.readlines()
        for line in tqdm(lines):
            info = json.loads(line.strip())
            info_dict = info[0]
            vis_list.append(info_dict)
    vis_ann(vis_list, output_path)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('usage: python tools.py func', file=sys.stderr)
