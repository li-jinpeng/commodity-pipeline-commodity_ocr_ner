#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
import time
import json
import logging
import signal
import numpy as np
from commodity_api import video_intention_cls
from commodity_api import video_commodity_cls
from commodity_api import commodity_det
from commodity_api import commodity_cls
from commodity_api import commodity_text
from commodity_api import commodity_emb
from commodity_api import commodity_ann
from commodity_api import commodity_image
from commodity_api import commodity_attrs
from commodity_api import commodity_ocr
from commodity_api import Predictor
from io import BytesIO
import io
from PIL import Image

from protos.model_serving_pb2 import PredictRequest
from kess.framework import create_grpc_server, KessOption
from protos import model_serving_pb2_grpc
from protos import model_serving_pb2
from protos.model_serving_pb2 import MetaInfo, Media, Feature, FloatArray
from protos.model_serving_pb2_grpc import ModelServingStub
from optparse import OptionParser

logger = logging.getLogger(__name__)
fmt_str = ('%(asctime)s.%(msecs)03d %(levelname)7s '
           '[%(thread)d][%(process)d] %(message)s')
fmt = logging.Formatter(fmt_str, datefmt='%H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ModelServing(model_serving_pb2_grpc.ModelServingServicer):
    def __init__(self, optons):
        entity_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'commodity_entity.txt')
        self.product_entity = [line.strip() for line in open(entity_path, 'r')]
        logger.info("entity path: {}, length: {}".format(entity_path, len(self.product_entity)))

    def get_input(self, predictor, request):
        try:
            start = time.time()
            # image
            image_data = request.media.data
            item_id = request.meta.str_str_entries['item_id']
            image_type = request.meta.str_str_entries['image_type']
            end = time.time()
            time_use = end - start
            logger.info("step 1: (get input) time: {}, {}, {}".format(time_use, item_id, image_type))
            return item_id, image_data, image_type
        except Exception as e:
            logger.info("step 1 error.")
            logger.info(e)
            return None

    def apply_ocr(self, predictor, image_data, image_type):
        try:
            start = time.time()
            # 将一张图划分成多份
            image = Image.open(BytesIO(image_data))
            width, height = image.size
            image_data_list = []
            image_data_list.append(image_data)

            #part_num = 2
            #for i in range(part_num):
            #    image_data_part = image.crop((0, int(height / part_num) * i, width, int(height / part_num) * (i + 1)))
            #    buf = io.BytesIO()
            #    image_data_part.save(buf, format=image.format)
            #    image_data_part = buf.getvalue()
            #    image_data_list.append(image_data_part)

            ocr_results_list = predictor.run(image_data_list, commodity_ocr)
            word_res = []
            frame_res = []
            for ocr_results in ocr_results_list:
                for item in ocr_results.ocr_result[''].ocr_item:
                    word_res.append([item.words, [item.location.left, item.location.top, item.location.width, item.location.height]])
                frame_res.append([ocr_results.ocr_result[''].frame_words, ocr_results.ocr_result[''].frame_width, ocr_results.ocr_result[''].frame_height])
            end = time.time()
            time_use = end - start
            logger.info("step 2: (apply ocr) time: {}, {}, {}".format(time_use, len(word_res), len(frame_res)))
            return word_res, frame_res
        except Exception as e:
            logger.info("step 2 error.")
            logger.info(e)
            return None

    def apply_ner(self, predictor, frame_res_list):
        try:
            start = time.time()
            ner_dict = {}
            for idx, frame_res in enumerate(frame_res_list):
                text = frame_res[0]
                if len(text) != 0:
                    ner_results = predictor.run([text], commodity_attrs)[0]
                    if len(ner_results.split("\t")) <= 1:
                        continue
                    for ner in ner_results.split("\t")[1].split("|"):
                        try:
                            if len(ner) == 0:
                                continue
                            key, value, id = ner.split(":")
                            if key not in ner_dict:
                                ner_dict[key] = set()
                            ner_dict[key].add(value)
                        except Exception as e:
                            print(e)
                            logger.info(e.__traceback__.tb_lineno)
            for key, value in ner_dict.items():
                ner_dict[key] = list(value)
            end = time.time()
            time_use = end - start
            logger.info("step 3: (apply ner) time: {}".format(time_use))
            return ner_dict
        except Exception as e:
            logger.info("step 3 error.")
            logger.info(e)
            logger.info(e.__traceback__.tb_lineno)
            return None

    def get_output(self, item_id, word_res, frame_res, ner_dict):
        try:
            find_attrs_list = [
                               ['净含量'],
                               #['食品生产许可证编号', '食品生产许可证号', '生产许可证编号', '生产许可证号', '生产许可证'],
                               ['备案编号'],
                               ['书号'],
                              ]
            for attrs_list in find_attrs_list:
                for word in word_res:
                    word = word[0]
                    word = word.replace("：", "").replace(":", "").replace("】", "")
                    for attr in attrs_list:
                        if attr in word:
                            idx = word.find(attr) + len(attr)
                            text = word[idx : ].strip()
                            if len(text) == 0:
                                continue
                            if attr not in ner_dict:
                                ner_dict[attr] = set()
                            ner_dict[attr].add(text)
                            break

            prefix_list = [
                           'CQC', '粤G妆网备字', '国妆特字G', '国妆备进字J', '国妆特进字J'
                          ]
            for word in word_res:
                word = word[0]
                word = word.replace("：", "").replace(":", "")
                for prefix in prefix_list:
                    if prefix in word:
                        idx = word.find(prefix) + len(prefix)
                        if idx == len(word):
                            continue
                        for i in range(idx, len(word)):
                            if word[i] >= '0' and word[i] <= '9':
                                continue
                            break
                        text = word[word.find(prefix) : i + 1]
                        if len(text) == 0:
                            continue
                        if '备案编号' not in ner_dict:
                            ner_dict['备案编号'] = set()
                        ner_dict['备案编号'].add(text)
                        break

            # SC 编号, SC + 14位编号
            for word in word_res:
                word = word[0]
                word = word.replace(" ", "").replace("s", "S").replace("c", "C").replace("5C", "SC")
                prefix = "SC"
                if prefix in word:
                    idx_list = [m.start() for m in re.finditer(prefix, word)]
                    for idx in idx_list:
                        if len(word[idx + len(prefix):]) < 14:
                            continue
                        flag = True
                        start = idx + len(prefix)
                        for i in range(14):
                            if word[start + i] >= '0' and word[start + i] <= '9':
                                continue
                            else:
                                flag = False
                                break
                        if flag:
                            if '食品生产许可证编号' not in ner_dict:
                                ner_dict['食品生产许可证编号'] = set()
                            ner_dict['食品生产许可证编号'].add(word[idx : idx + len(prefix) + 14])
                            break

            for key, value in ner_dict.items():
                ner_dict[key] = list(value)

            start = time.time()
            results = []
            res_dict = {}
            res_dict = {"item_id" : item_id}
            res_dict['word_res'] = ""
            res_dict['frame_res'] = frame_res
            res_dict['ner_dict'] = ner_dict
            results.append(res_dict)
            meta = MetaInfo()
            if len(results) > 0:
                meta.str_str_entries['commodity_item_ocr_ner'] = json.dumps(results, ensure_ascii=False)
            end = time.time()
            time_use = end - start
            logger.info("step 4: (get output) time: {}, {}".format(time_use, len(results)))
            return meta
        except Exception as e:
            logger.info("step 4 error.")
            logger.info(e)
            return None

    # 多线程执行
    def Predict(self, request, timeout):
        # 假装在工作
        #time.sleep(1)
        logger.info('Predict: Hello, {}'.format(request.id))

        try:
            stime = time.time()
            feature = Feature()
            meta = MetaInfo()
            predictor = Predictor(worker_num=4)
            '''
            商品OCR和NER识别
            '''
            item_id, image_data, image_type = self.get_input(predictor, request)
            word_res, frame_res = self.apply_ocr(predictor, image_data, image_type)
            ner_dict = self.apply_ner(predictor, frame_res)
            meta = self.get_output(item_id, word_res, frame_res, ner_dict)

            etime = time.time()
            logger.info('time elasped\t{}s'.format(etime-stime))
        except Exception as e:
            print ("Error!!!!!", str(e))
            feature = Feature()
            meta = MetaInfo()

        return model_serving_pb2.PredictResult(feature=feature, meta=meta)

    # 多线程执行
    def BatchPredict(self, request, context):
        # 假装在工作
        time.sleep(1)
        logger.info('BatchPredict: Hello, {}'.format(request.id))
        result = model_serving_pb2.BatchPredictResult()
        meta_info = model_serving_pb2.MetaInfo(int32_val=99)
        predict_result = model_serving_pb2.PredictResult(meta=meta_info)
        result.result[request.id].CopyFrom(predict_result)
        return result

def parse_param():
    usage = "usage: %prog [port] [service_name]"
    parser = OptionParser(usage=usage, version="%prog 1.0")
    parser.add_option("-p",action="store",dest="port",default=None,type=int,help='port')
    parser.add_option("-s",action="store",dest="service_name",default="grpc_mmu_vision_retrieval_commodity_tags",type=str,help='service name')

    options, args = parser.parse_args()
    return options, args, usage

def sig_handler(signum, frame):
    logger.info('sig_handler: signum {}, wait 60 seconds before quit'.format(str(signum)))
    server.stop(10)
    logger.info('sig_handler: unregister from kess success')
    sys.exit(signum)

if __name__ == "__main__":

    # 解析参数
    options, args, usage = parse_param()

    # 描述服务
    kess_option = KessOption(
        name=options.service_name,
        owner='mmu',
        port=options.port,
        shard_name='s0',
        biz_def='mmu'
    )

    # 生成服务
    server = create_grpc_server(
        kess_option,
        model_serving_pb2_grpc.add_ModelServingServicer_to_server,
        ModelServing(options),
        None,
        8
    )

    # 启动服务
    server.start()
    start_msg = f'grpc服务 {kess_option} 成功启动'
    logger.info(start_msg)

    # 注册新号处理函数
    signal.signal(signal.SIGHUP, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGQUIT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # 进入服务状态
    signal.pause()
