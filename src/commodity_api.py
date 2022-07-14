#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import math
import time
import json
import logging
import requests
import threading
from tqdm import tqdm
from queue import Empty, Queue

#import PIL.Image as Image
from io import BytesIO

import numpy as np

from kess.framework import (
  ClientOption,
  GrpcClient
)

from protos.model_serving_pb2 import PredictRequest
from protos.model_serving_pb2 import PredictResult
from protos.model_serving_pb2 import MetaInfo, Media, Feature, FloatArray
from protos.model_serving_pb2 import StringArray
from protos.model_serving_pb2_grpc import ModelServingStub
from protos.ann_pb2_grpc import CommonVisionSearchServiceStub
from protos.ann_pb2 import CommonVisionSearchRequest, CommonAttr, Image
from protos.mmu_ocr_detect_pb2 import OcrDetectRequest, OcrDetectResponse
from protos.mmu_ocr_detect_pb2_grpc import MmuOcrDetectServiceStub
from mmu.media_common_pb2 import ImgUnit

logger = logging.getLogger(__name__)
fmt_str = ('%(asctime)s.%(msecs)03d %(levelname)7s '
           '[%(thread)d][%(process)d] %(message)s')
fmt = logging.Formatter(fmt_str, datefmt='%H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


video_intention_client_option = ClientOption(
    biz_def='mmu',
    grpc_service_name='grpc_mmu_visionRetrievalVideoIntentionCls',
    grpc_stub_class=ModelServingStub)
video_intention_client = GrpcClient(video_intention_client_option)

def video_intention_cls(imagedata_list, text, timeout=100):
    try:
        req = PredictRequest(id='test')
        medias = []
        for imagedata in imagedata_list:
            media = Media()
            media.data = imagedata
            medias.append(media)
        req.medias.extend(medias)
        req.meta.str_str_entries['text'] = text
        resp = video_intention_client.Predict(req, timeout=timeout)
        result = resp.meta.str_array.str_elems[0]
        score = resp.meta.float_array.float_elems[0]
        if score > 0.8:
            return result
        return None
    except Exception as e:
        logger.warning(e)
        return None


video_cls_client_option = ClientOption(
    biz_def='mmu',
    grpc_service_name='grpc_mmu_visionRetrievalVideoCommodityCls',
    grpc_stub_class=ModelServingStub)
video_cls_client = GrpcClient(video_cls_client_option)

def video_commodity_cls(imagedata_list, text, timeout=100):
    try:
        req = PredictRequest(id='test')
        medias = []
        for imagedata in imagedata_list:
            media = Media()
            media.data = imagedata
            medias.append(media)
        req.medias.extend(medias)
        req.meta.str_str_entries['text'] = text
        resp = video_cls_client.Predict(req, timeout=timeout)
        meta = resp.meta
        res = []
        for key, value in zip(meta.str_array.str_elems, meta.float_array.float_elems):
            if value > 0.35:
                res.append(key)
        return res
    except Exception as e:
        logger.warning(e)
        return []


det_client_option = ClientOption(
    biz_def='mmu',
    #grpc_service_name='grpc_mmu_visionRetrievalCommodityDet',
    grpc_service_name='grpc_mmu_visionRetrievalCommodityEcommerceDet',
    #grpc_service_name='grpc_mmu_visionRetrievalCommodityDetIntention',
    grpc_stub_class=ModelServingStub)
det_client = GrpcClient(det_client_option)

#pid2detres = {}
#pid2shape = {}
##with open("/mnt/se/zhizhen/workspace/projects/commodity/detection/centernet/extractor/results/ecommerce/imgs_unsdg.txt") as fr:
##with open("/mnt/se/zhizhen/workspace/projects/commodity/detection/centernet/extractor/results/ecommerce/imgs_sdg.txt") as fr:
#with open("/mnt/se/zhizhen/workspace/projects/commodity/detection/centernet/extractor/results/2021/imgs_unsdg.txt") as fr:
##with open("/mnt/se/zhizhen/workspace/projects/commodity/detection/centernet/extractor/results/ecommerce/full_img_path_sdg.list") as fr:
#    lines = fr.readlines()
#    for line in tqdm(lines):
#        info = line.strip().split("\t")
#        if len(info) == 1:
#            continue
#        path = info[0]
#        pid = path.split("/")[-1].split(".")[0]
#        try:
#            height, width, channel = cv2.imread(path).shape
#        except Exception as e:
#            print(e, path)
#            continue
#        pid2shape[pid] = [width, height]
#        info = info[1:]
#        for j in range(0, int(len(info) / 6)):
#            det_res = info[j * 6 : (j + 1) * 6]
#            x1, y1, x2, y2 = int(det_res[0]), int(det_res[1]), int(det_res[2]), int(det_res[3])
#            label, score = int(det_res[4]), float(det_res[5])
#            if float(score) < 0.2:
#                continue
#            if pid not in pid2detres:
#                pid2detres[pid] = []
#            pid2detres[pid].append([x1, y1, x2, y2, label, '', score, 1])
#for pid, detres in pid2detres.items():
#    print(pid, detres)

def commodity_det(param_list, timeout=600):
    global pid2detres
    req = PredictRequest(id='commodity_det_ckb')
    pid, image_data = param_list
    req.media.data = image_data
    req.media.meta.float_val = 0.2

    count = 0
    while count < 5:
        try:
            resp = det_client.Predict(req, timeout=timeout)
            response = resp.feature.str_float_entries
            res = []
            for key, prob in response.items():
                x1, y1, x2, y2, lab, tag, main_object = key.split(':')
                res.append([int(x1), int(y1), int(x2), int(y2), int(lab), tag, float(prob), int(main_object)])
            height, width = resp.meta.int32_pair.first, resp.meta.int32_pair.second

            #if pid not in pid2detres:
            #    return None
            #res = pid2detres[pid]
            #width, height = pid2shape[pid]

            return res, [width, height]
        except Exception as e:
            logger.warning("det error: " + str(e))
            count += 1
    return None


cls_client_option = ClientOption(
    biz_def='mmu',
    grpc_service_name='grpc_mmu_visionRetrievalCommodityCls',
    grpc_stub_class=ModelServingStub)
cls_client = GrpcClient(cls_client_option)

def commodity_cls(param_list, timeout=200):
    image_data, pos, wh, det_tag, det_score = param_list
    req = PredictRequest(id='commodity_cls_ckb')
    req.media.data = image_data

    # media.meta
    width, height = wh
    if len(pos) == 4:
        x1, y1, x2, y2 = pos
        x1, y1, x2, y2 = x1/width, y1/height, x2/width, y2/height
        req.media.meta.float_array.float_elems.extend([x1, y1, x2, y2])
    try:
        resp = cls_client.Predict(req, timeout=timeout)
        meta = resp.meta
        key_value_list = []
        for key, id, value in zip(meta.str_array.str_elems, meta.int32_array.int32_elems, meta.float_array.float_elems):
            key_value_list.append(key + ":" + str(id) + ":" + str(value))
        #emb = [x for x in resp.feature.float_array.float_elems]
        return resp.feature.str_float_entries, key_value_list
    except Exception as e:
        logger.warning("cls: " + str(e))
        return None


#text_client_option = ClientOption(
#                    biz_def='mmu',
#                    grpc_service_name='mmu-vision-retrieval-text2image-product-text',
#                    grpc_stub_class=ModelServingStub)
#text_client = GrpcClient(text_client_option)
#def commodity_text(text_content, timeout=200):
#    req = PredictRequest(id=text_content)
#    req.media.data = text_content.encode('utf-8')
#    try:
#        resp = text_client.Predict(req, timeout=timeout)
#        return [x for x in resp.feature.float_array.float_elems]
#    except Exception as e:
#        logger.warning("text: " + str(e))
#        return None

text_client_option = ClientOption(
    biz_def='mmu',
    grpc_service_name='grpc_mmu_visionRetrievalCommodityTextEmbCopy',
    #grpc_service_name='grpc_mmu_visionRetrievalCommodityTextEmb',
    grpc_stub_class=ModelServingStub)
text_client = GrpcClient(text_client_option)
def commodity_text(text_content, timeout=200):
    req = PredictRequest(id='test')
    req.meta.str_str_entries['text_emb_zhs'] = text_content
    count = 0
    while count < 3:
        try:
            resp = text_client.Predict(req, timeout=timeout)
            feat_list = [float(x) for x in resp.meta.str_str_entries['text_emb_zhs'].split(",")]
            feat_arr = np.array(feat_list)
            normalized_arr = feat_arr / np.sqrt(np.sum(feat_arr ** 2))
            return normalized_arr.tolist()
        except Exception as e:
            logger.warning("text: " + str(e))
            logger.warning(text_content, resp.meta.str_str_entries['text_emb_zhs'].split(","))
            count += 1
    return None

image_client_option = ClientOption(
                        biz_def='mmu',
                        grpc_service_name='mmu-vision-retrieval-text2image-product-image',
                        grpc_stub_class=ModelServingStub)
image_client = GrpcClient(image_client_option)
def commodity_image(param_list, timeout=200):
    print("param_list: ", len(param_list))
    image_data, pos, wh, det_tag, det_score = param_list
    req = PredictRequest(id='commodity_image_ckb')
    req.media.data = image_data

    # media.meta
    width, height = wh
    if len(pos) == 4:
        x1, y1, x2, y2 = pos
        x1, y1, x2, y2 = x1/width, y1/height, x2/width, y2/height
        req.media.meta.float_array.float_elems.extend([x1, y1, x2, y2])
    try:
        resp = image_client.Predict(req, timeout=timeout)
        emb = [x for x in resp.feature.float_array.float_elems]
        return emb
    except Exception as e:
        logger.warning("image: " + str(e))
        return None

'''
emb_client_option = ClientOption(
    biz_def='mmu',
    grpc_service_name='grpc_mmu_visionRetrievalFlowDistEmbed',
    grpc_stub_class=ModelServingStub)
emb_client = GrpcClient(emb_client_option)

def commodity_emb(param_list, timeout=200):
    try:
        image_data, pos, wh, tag, clc_tag, det_score = param_list
        req = PredictRequest(id='commodity_emb_ckb')
        req.media.data = image_data

        # media.meta
        width, height = wh
        x1, y1, x2, y2 = pos
        x1, y1, x2, y2 = x1/width, y1/height, x2/width, y2/height
        req.media.meta.float_array.float_elems.extend([x1, y1, x2, y2])
        req.media.meta.str_str_entries['tag'] = tag
        req.media.meta.str_str_entries['clc_tag'] = clc_tag
        req.media.meta.str_str_entries['is_zhannei'] = 'True'

        resp = emb_client.Predict(req, timeout=timeout)
        embed = resp.feature.float_array.float_elems
        ann_flag = resp.meta.str_val
        return embed, ann_flag
    except Exception as e:
        logger.warning("emb: " + str(e))
        return None
'''


ann_client_option = ClientOption(
    biz_def='mmu',
    grpc_service_name='grpc_mmu_mmuCommodityVisionSearchService',
    grpc_stub_class=CommonVisionSearchServiceStub)
ann_client = GrpcClient(ann_client_option)

def commodity_ann(param_list, timeout=200):
    try:
        session_id = 'ann_zhs'
        service_type, count, embed, tag_name_list = param_list
        req = CommonVisionSearchRequest()
        req.session_id = session_id
        req.service_type = service_type
        req.count = int(count)

        image = Image()

        attr = CommonAttr()
        attr.name = 'feature'
        attr.float_list.value.extend(embed)
        image.attr.append(attr)

        attr = CommonAttr()
        attr.name = 'tag_name_list'
        attr.bytes_list.value.extend(tag_name_list)
        image.attr.append(attr)

        req.image.append(image)

        resp = ann_client.VisionSearch(req)
        id_list, score_list = [], []
        for search_item in resp.search_item:
            id, score = search_item.id, search_item.score
            id_list.append(id)
            score_list.append(score)

        return id_list, score_list
    except Exception as e:
        logger.warning("ann: " + str(e))
        return None


sim_client_option = ClientOption(
    biz_def='mmu',
    grpc_service_name='grpc_mmu_visionRetrievalCommodityEmbeddingSIM128d',
    grpc_stub_class=ModelServingStub)
sim_client = GrpcClient(sim_client_option)

def commodity_emb(param_list, timeout=100):
    try:
        frame_id, pos_list, image_data = param_list
        req = PredictRequest(id='test_vision')
        req.media.data = image_data
        req.meta.str_str_entries['emb_type'] = 'vision'
        if len(pos_list) == 4:
            req.media.meta.float_array.float_elems.extend(pos_list)
        resp = sim_client.Predict(req, timeout=timeout)
        return frame_id, pos_list, [round(x, 6) for x in resp.feature.float_array.float_elems]
    except Exception as e:
        logger.warning("request_vision: " + str(e))
        return None

attr_client_option = ClientOption(
    biz_def='mmu',
    grpc_service_name='grpc_mmu_visionRetrievalCommodityAttrs',
    grpc_stub_class=ModelServingStub)
attrs_client = GrpcClient(attr_client_option)

def commodity_attrs(text, timeout=100):
    try:
        req = PredictRequest(id='test')
        req.meta.str_val = text
        resp = attrs_client.Predict(req, timeout=timeout)
        return resp.meta.str_str_entries['attrs']
    except Exception as e:
        logger.warning("request_attrs: " + str(e))
        return None

ocr_client_option = ClientOption(
    biz_def='commodity_img_ocr',
    grpc_service_name='grpc_ztOcrDetectService',
    grpc_stub_class=MmuOcrDetectServiceStub
)
ocr_client = GrpcClient(ocr_client_option)
def commodity_ocr(imagedata_list, timeout=30):
    try:
        biz = 'commodity_img_ocr'
        name = 'test'
        req = OcrDetectRequest(req_id=name)
        for i, imagedata in enumerate(imagedata_list):
            input_img = ImgUnit()
            input_img.id = str(i + 1)
            input_img.image = imagedata
            req.img.extend([input_img])
        req.biz=biz
        resp = ocr_client.OcrDetect(req, timeout=timeout)
        return resp
    except Exception as e:
        logger.warning("request_ocr: " + str(e))
        return None

def worker(inqueue, outqueue, client):
    while inqueue.qsize() > 0:
        try:
            idx, data = inqueue.get(timeout=0.01)
            result = client(data)
            outqueue.put((idx, result))
        except Empty as err:
            logger.info('input queue is empty: {}'.format(err))
            return
        except Exception as err:
            logger.warning('err_msg: {}'.format(err))
            time.sleep(0.01)
            continue

class Predictor(object):
    def __init__(self, worker_num=4):
        self._worker_num = worker_num
        self._iqueue = Queue()
        self._oqueue = Queue()

    def run(self, data_list, client):
        for idx, data in enumerate(data_list):
            self._iqueue.put((idx, data))

        workers = []
        for i in range(self._worker_num):
            workers.append(threading.Thread(
                target=worker, args=(self._iqueue, self._oqueue, client)))

        for work in workers:
            work.start()

        for work in workers:
            work.join()

        outputs = []
        while self._oqueue.qsize() > 0:
            try:
                out = self._oqueue.get(timeout=0.01)
                outputs.append(out)
            except Empty as err:
                logger.info('output queue is empty: {}'.format(err))
                break
            except Exception as err:
                logger.warning('err_msg: {}'.format(err))
                continue

        outputs = sorted(outputs, key=lambda x : x[0])
        outputs = [x[1] for x in outputs]
        return outputs
