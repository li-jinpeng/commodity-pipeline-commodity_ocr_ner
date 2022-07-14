import os
import sys
import re
import glob
import math
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
        if i % 5 == 0:
            html_file_fp.write('<p>\n')
            html_file_fp.write('<table border="0" align="center">\n')
            html_file_fp.write('<tr><td height="10" colspan="100"><hr></td></tr>\n')
            html_file_fp.write('<tr>\n')

        path, info = items
        color = 'white'
        html_file_fp.write(
            """
            <td bgcolor=%s align='center'>
                <img width="224" height="224" src="data:image/jpeg;base64, %s">
                <br> item_id: %s
                <br> frame_text: %s
                <br> attributes: %s
            </td>
            """ % (color, base64.b64encode(open(path, 'rb').read()).decode(), info['item_id'], info['frame_text'], info['attributes_dict'])
        )

        if (i + 1) % 5 == 0:
            html_file_fp.write('</tr>\n')
            html_file_fp.write('<tr><td height="10" colspan="100"><hr></td></tr>\n')
            html_file_fp.write('</table>\n')
            html_file_fp.write('</p>\n')
    html_file_fp.write('</body>\n</html>')


def main(input_path, output_path):
    vis_list = []
    with open(input_path) as fr:
        lines = fr.readlines()[:1000]
        for line in tqdm(lines):
            values = line.strip().split("\t")
            path, info = values
            info = json.loads(info)[0]
            if len(info['frame_text'][0][0]) < 3:
                continue
            #flag = True
            #for word in ['许可证']:
            #    if word in info['frame_res'][0][0]:
            #        flag = False
            #        break
            #if flag:
            #    continue
            #print(info)
            print(info['item_id'])
            vis_list.append([path, info])
    vis_ann(vis_list, output_path)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('usage: python tools.py func', file=sys.stderr)
