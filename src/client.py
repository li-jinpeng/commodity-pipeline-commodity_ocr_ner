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
    grpc_service_name='grpc_mmu_visionRetrievalCommodityOCR_0609',
    grpc_stub_class=ModelServingStub)
client = GrpcClient(client_option)

def predict(pid, image, timeout=600):
    try:
        req = PredictRequest(id=pid)
        req.meta.str_str_entries['image_type'] = 'None'
        req.meta.str_str_entries['item_id'] = pid
        image = open(path, 'rb').read()
        req.media.data = image
        resp = client.Predict(req, timeout=timeout)
        res = resp.meta.str_str_entries['commodity_item_ocr_ner']
        return res
    except Exception as e:
        print("error: ", str(e))
        exit(0)

if __name__ == "__main__":
    for i in range(100):
        path = "/mnt/longvideo/zhonghuasong/retreival_workshop/projects/fengkong/evaluate_data/images/item-733352084-c13dbaf7becd48d89f1f06bd485787b6.jpg"
        image = open(path, 'rb').read()
        res = predict(path.split("/")[-1], image)
        print(res)
