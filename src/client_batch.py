#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import json
import glob
import re
from PIL import Image
from tqdm import tqdm
from multiprocessing import Process
from google.protobuf.json_format import MessageToDict

from kess.framework import (
    ClientOption,
    GrpcClient
)

from protos.model_serving_pb2 import PredictRequest
from protos.model_serving_pb2 import PredictResult
from protos.model_serving_pb2 import BatchPredictRequest
from protos.model_serving_pb2 import BatchPredictResult
from protos.model_serving_pb2 import MetaInfo, Media, FloatArray, Feature
from protos.model_serving_pb2 import StringArray
from protos.model_serving_pb2_grpc import ModelServingStub


logger = logging.getLogger(__name__)
fmt_str = ('%(asctime)s.%(msecs)03d %(levelname)7s '
            '[%(thread)d][%(process)d] %(message)s')
fmt = logging.Formatter(fmt_str, datefmt='%H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


client_option = ClientOption(
    biz_def='mmu',
    grpc_service_name='grpc_mmu_visionRetrievalAttrs',
    #grpc_service_name='grpc_mmu_visionRetrievalCommodityOCRAttrsForShenhe',
    grpc_stub_class=ModelServingStub)
client = GrpcClient(client_option)

def predict(pid, timeout=600):
    try:
        req = PredictRequest(id=pid)
        req.meta.str_str_entries['item_id'] = pid
        #req.meta.str_str_entries['title'] = '皮皮虾酱（3瓶）'

        for i, path in enumerate(glob.glob("/home/lijinpeng/.jupyter/test/*.png")):
            if i == 0:
                continue
        #for path in glob.glob("/mnt/longvideo/zhonghuasong/retreival_workshop/tmp/属性库/1.生产厂商/*.jpeg"):
            media = Media()
            image = open(path, 'rb').read()
            media.data = image
            req.medias.append(media)
        print(len(req.medias))

        resp = client.Predict(req, timeout=timeout)
        res = resp.meta.str_str_entries['commodity_item_ocr_ner']
        return res
    except Exception as e:
        print("error: ", str(e))
        exit(0)

if __name__ == "__main__":
    count = 0
    #with open("/mnt/longvideo/zhonghuasong/retreival_workshop/projects/fengkong/evaluate_data/txt/test_0325.list", 'w') as fw:
    #    lines = open("/mnt/longvideo/zhonghuasong/retreival_workshop/projects/fengkong/evaluate_data/txt/good_path.list").readlines()[:10]
    #    for line in tqdm(lines):
    #        path = line.strip()
    #        print(path)
    #        #if 'bdeb6ff1-3114-4558-8ebe-a9a885ba0198.jpg' not in path:
    #        #    continue
    #        image = open(path, 'rb').read()
    #        res = predict(path.split("/")[-1], image)
    #        out = "\t".join([path, res]) + "\n"
    #        print(out)
    #        #fw.write(out)
    #        #count += 1
    #        #if count > 0 and count % 10 == 0:
    #        #    fw.flush()
    for i in range(1000):
        res = predict(pid = '123')
        print(res)
