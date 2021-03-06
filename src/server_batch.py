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
            image_data_list = []
            for media in request.medias:
                image_data = media.data
                image_data_list.append(image_data)
            item_id = request.meta.str_str_entries['item_id']
            if 'title' in request.meta.str_str_entries:
                title = request.meta.str_str_entries['title']
            else:
                title = ''

            end = time.time()
            time_use = end - start
            logger.info("step 1: (get input) time: {}, {}, {}".format(time_use, item_id, len(image_data_list)))
            return item_id, image_data_list, title
        except Exception as e:
            logger.info("step 1 error.")
            logger.info(e)
            return None

    def apply_ocr(self, predictor, image_data_list):
        try:
            start = time.time()
            image_data_input = []
            for image_data in image_data_list:
                image = Image.open(BytesIO(image_data))
                width, height = image.size
                image_data_input.append(image_data)

            ocr_results_list = predictor.run([image_data_input], commodity_ocr)
            ocr_results = ocr_results_list[0]

            word_res_list = []
            frame_res_list = []
            for i in range(len(image_data_input)):
                word_res = []
                frame_res = []
                for item in ocr_results.ocr_result['{}'.format(i + 1)].ocr_item:
                    word_res.append([item.words, [item.location.left, item.location.top, item.location.width, item.location.height]])
                frame_res.append([ocr_results.ocr_result['{}'.format(i + 1)].frame_words, \
                                  ocr_results.ocr_result['{}'.format(i + 1)].frame_width, \
                                  ocr_results.ocr_result['{}'.format(i + 1)].frame_height])
                word_res_list.append(word_res)
                frame_res_list.append(frame_res)
                print("i: ", i + 1)
                print("word_res: ", word_res)
                print("frame_res: ", frame_res)
            #print("word_res_list: ", word_res_list)
            #print("frame_res_list: ", frame_res_list)
            end = time.time()
            time_use = end - start
            logger.info("step 2: (apply ocr) time: {}, {}, {}".format(time_use, len(word_res_list), len(frame_res_list)))
            return word_res_list, frame_res_list
        except Exception as e:
            import traceback
            logger.info("step 2 error.")
            logger.info(e)
            logger.info(traceback.print_exc())
            return None

    def apply_ner(self, predictor, frame_res_list):
        try:
            start = time.time()
            ner_dict = {}
            print("apply ner: ", frame_res_list)
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
                               ['?????????'],
                               ['????????????'],
                              ]
            for attrs_list in find_attrs_list:
                for word in word_res:
                    word = word[0]
                    word = word.replace("???", "").replace(":", "").replace("???", "").replace(" ", "")
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

            # ????????????
            prefix_list = [
                           '????????????', '?????????'
                          ]
            for word in word_res:
                word = word[0]
                word = word.replace("???", "").replace(":", "").replace(" ", "")
                for prefix in prefix_list:
                    if prefix in word:
                        idx = word.find(prefix) + len(prefix)
                        if idx == len(word):
                            continue
                        text = word[word.find(prefix) : ]
                        if len(text) == 0:
                            continue
                        if '????????????' not in ner_dict:
                            ner_dict['????????????'] = set()
                        ner_dict['????????????'].add(text)
                        break

            # ?????????
            prefix_list = [
                           '?????????'
                          ]
            for word in word_res:
                word = word[0]
                word = word.replace("???", "").replace(":", "").replace(" ", "")
                for prefix in prefix_list:
                    if prefix in word:
                        idx = word.find(prefix) + len(prefix)
                        if idx == len(word):
                            continue
                        for i in range(idx, len(word)):
                            if word[i] not in ['???', 'l', 'L', '??????', 'ml', "mL", '???', 'g', '??????', 'kg']:
                                continue
                            break
                        text = word[word.find(prefix) + len(prefix) : i + 1]
                        if len(text) == 0:
                            continue
                        if '?????????' not in ner_dict:
                            ner_dict['?????????'] = set()
                        ner_dict['?????????'].add(text)
                        break

            # ?????????
            prefix_list = [
                           '?????????'
                          ]
            for word in word_res:
                word = word[0]
                word = word.replace("???", "").replace(":", "").replace(" ", "")
                for prefix in prefix_list:
                    if prefix in word:
                        idx = word.find(prefix) + len(prefix)
                        if idx == len(word):
                            continue
                        for i in range(idx, len(word)):
                            if word[i] not in ['???', '???', '???', '???', '??????']:
                                continue
                            break
                        text = word[word.find(prefix) + len(prefix) : i + 1]
                        if len(text) == 0:
                            continue
                        if '?????????' not in ner_dict:
                            ner_dict['?????????'] = set()
                        ner_dict['?????????'].add(text)
                        break

            # 3C??????
            prefix_list = [
                           '????????????'
                          ]
            for word in word_res:
                word = word[0]
                word = word.replace("???", "").replace(":", "").replace(" ", "")
                for prefix in prefix_list:
                    if prefix in word:
                        idx = word.find(prefix) + len(prefix)
                        if idx == len(word):
                            continue
                        for i in range(idx, len(word)):
                            if (word[i] >= '0' and word[i] <= '9'):
                                continue
                            break
                        text = word[word.find(prefix) + len(prefix) : i + 1]
                        if len(text) != 16:
                            continue
                        if '3C??????' not in ner_dict:
                            ner_dict['3C??????'] = set()
                        ner_dict['3C??????'].add(text)
                        break

            # ??????
            prefix_list = [
                           'isbn', 'ISBN', 'issn', 'ISSN'
                          ]
            for word in word_res:
                word = word[0]
                word = word.replace("???", "").replace(":", "").replace(" ", "")
                for prefix in prefix_list:
                    if prefix in word:
                        idx = word.find(prefix) + len(prefix)
                        if idx == len(word):
                            continue
                        for i in range(idx, len(word)):
                            if (word[i] >= '0' and word[i] <= '9') or word[i] == '-':
                                continue
                            break
                        text = word[word.find(prefix) : i + 1]
                        if len(text) == 0:
                            continue
                        if 'isbn/issn' not in ner_dict:
                            ner_dict['isbn/issn'] = set()
                        ner_dict['isbn/issn'].add(text)
                        break

            # ????????????
            prefix_list = [
                           'CQC', '???G????????????', '????????????G', '???????????????J', '???????????????J'
                          ]
            for word in word_res:
                word = word[0]
                word = word.replace("???", "").replace(":", "").replace(" ", "")
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
                        if '?????????????????????' not in ner_dict:
                            ner_dict['?????????????????????'] = set()
                        ner_dict['?????????????????????'].add(text)
                        break

            # SC ??????, SC + 14?????????
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
                            if 'SC' not in ner_dict:
                                ner_dict['SC'] = set()
                            ner_dict['SC'].add(word[idx : idx + len(prefix) + 14])
                            break

            for key, value in ner_dict.items():
                ner_dict[key] = list(value)

            start = time.time()
            res_dict = {}
            res_dict = {"item_id" : item_id}
            res_dict['ner_dict'] = ner_dict

            end = time.time()
            time_use = end - start
            logger.info("step 4: (get output) time: {}".format(time_use))
            return res_dict
        except Exception as e:
            logger.info("step 4 error.")
            logger.info(e)
            return None

    # ???????????????
    def Predict(self, request, timeout):
        # ???????????????
        #time.sleep(1)
        logger.info('Predict: Hello, {}'.format(request.id))

        try:
            stime = time.time()
            feature = Feature()
            meta = MetaInfo()
            predictor = Predictor(worker_num=4)
            '''
            ??????OCR???NER??????
            '''
            item_id, image_data_list, title = self.get_input(predictor, request)

            total_ner_dict = {
                "item_id" : item_id,
                'frames' : [],
                'attrs' : {
                    '????????????' : [],
                    '?????????' : [],
                    'SC' : [],
                    '?????????????????????' : [],
                    '3C??????' : [],
                    '??????' : [],
                    'isbn/issn' : [],
                    '?????????' : [],
                    '????????????' : [],
                    '?????????' : [],
                }
            }

            word_res_list, frame_res_list = self.apply_ocr(predictor, image_data_list)
            for word_res, frame_res in zip(word_res_list, frame_res_list):
                try:
                    ner_dict = self.apply_ner(predictor, frame_res)
                    res_dict = self.get_output(item_id, word_res, frame_res, ner_dict)
                    if res_dict is None:
                        continue
                    for key, value in res_dict['ner_dict'].items():
                        if key not in total_ner_dict['attrs']:
                            continue
                        total_ner_dict['attrs'][key].extend(value)
                except Exception as e:
                    print(e)
                    continue

            # title
            ner_dict = self.apply_ner(predictor, [[title]])
            if ner_dict is not None and '??????' in ner_dict:
                total_ner_dict['attrs']['??????'].extend(ner_dict['??????'])

            for key, value in total_ner_dict['attrs'].items():
                value = list(set(value))
                total_ner_dict['attrs'][key] = value

            # frames text
            total_ner_dict['frames'] = frame_res_list
            meta.str_str_entries['commodity_item_ocr_ner'] = json.dumps(total_ner_dict, ensure_ascii=False)

            etime = time.time()
            logger.info('time elasped\t{}s'.format(etime-stime))
        except Exception as e:
            print ("Error!!!!!", str(e))
            feature = Feature()
            meta = MetaInfo()

        return model_serving_pb2.PredictResult(feature=feature, meta=meta)

    # ???????????????
    def BatchPredict(self, request, context):
        # ???????????????
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

    # ????????????
    options, args, usage = parse_param()

    # ????????????
    kess_option = KessOption(
        name=options.service_name,
        owner='mmu',
        port=options.port,
        shard_name='s0',
        biz_def='mmu'
    )

    # ????????????
    server = create_grpc_server(
        kess_option,
        model_serving_pb2_grpc.add_ModelServingServicer_to_server,
        ModelServing(options),
        None,
        8
    )

    # ????????????
    server.start()
    start_msg = f'grpc?????? {kess_option} ????????????'
    logger.info(start_msg)

    # ????????????????????????
    signal.signal(signal.SIGHUP, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGQUIT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # ??????????????????
    signal.pause()
