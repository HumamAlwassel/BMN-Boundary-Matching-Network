# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:21:39 2017

@author: wzmsltw
"""
import json
import os

import numpy as np
import pandas as pd
from progressbar import ProgressBar

pbar = ProgressBar()
import multiprocessing as mp


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def getDatasetDict():
    df = pd.read_csv("data/activitynet_annotations/video_info_new.csv")
    json_data = load_json("data/activitynet_annotations/anet_anno_action.json")
    database = json_data
    train_dict = {}
    val_dict = {}
    test_dict = {}
    for i in range(len(df)):
        video_name = df.video.values[i]
        video_info = database[video_name]
        video_new_info = {}
        video_new_info['duration_frame'] = video_info['duration_frame']
        video_new_info['duration_second'] = video_info['duration_second']
        video_new_info["feature_frame"] = video_info['feature_frame']
        video_subset = df.subset.values[i]
        video_new_info['annotations'] = video_info['annotations']
        if video_subset == "training":
            train_dict[video_name] = video_new_info
        elif video_subset == "validation":
            val_dict[video_name] = video_new_info
        elif video_subset == "testing":
            test_dict[video_name] = video_new_info
    return train_dict, val_dict, test_dict


def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor


def Soft_NMS(df, nms_threshold):
    df = df.sort_values(by="score", ascending=False)

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])

    rstart = []
    rend = []
    rscore = []

    while len(tscore) > 1 and len(rscore) < 101:
        max_index = tscore.index(max(tscore))
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx], tend[idx])
                if tmp_iou > 0.:
                    tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) / nms_threshold)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf


def min_max(x):
    x = (x - min(x)) / (max(x) - min(x))
    return x


def _proposal2detection(opt, video_list, video_dict, cuhk_data_score, cuhk_data_action):
    nms_threshold = opt["soft_nms"]
    for video_name in video_list:
        if opt['eval']=='validation':
            cuhk_score = cuhk_data_score[video_name[2:]]
            cuhk_class_1 = cuhk_data_action[np.argmax(cuhk_score)]
            cuhk_score_1 = max(cuhk_score)
        else:
            result = cuhk_data_score[video_name[2:]]
            cuhk_score = [x['score'] for x in result]
            cuhk_cls = [x['label'] for x in result]
            cuhk_class_1 = cuhk_cls[np.argmax(cuhk_score)]
            cuhk_score_1 = max(cuhk_score) / 10

        df = pd.read_csv(os.path.join(opt['output'], 'BMN_results/', video_name + '.csv'))
        # df['score'] =  df.xmin_score.values[:] * df.xmax_score.values[:]  * df.iou_score.values[:]
        df['score'] = df.clr_score.values[:]*df.reg_socre.values[:]
        if len(df) > 1:
            # df=NMS(df,nms_threshold)
            df = Soft_NMS(df, nms_threshold)

        df = df.sort_values(by="score", ascending=False)
        video_info = video_dict[video_name]
        video_duration = float(video_info["duration_frame"] / 16 * 16) / video_info["duration_frame"] * video_info[
            "duration_second"]
        proposal_list = []

        for j in range(min(100, len(df))):
            tmp_proposal = {}
            tmp_proposal["label"] = cuhk_class_1
            tmp_proposal["score"] = df.score.values[j] * cuhk_score_1
            tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * video_duration,
                                       min(1, df.xmax.values[j]) * video_duration]
            proposal_list.append(tmp_proposal)

        result_dict[video_name[2:]] = proposal_list


def generate_detection(opt):

    train_dict, val_dict, test_dict = getDatasetDict()
    if opt['eval']=='validation':
        print('using val, not test to run detection code')
        cuhk_data = load_json("data/cuhk_val_simp_share.json")
        cuhk_data_score = cuhk_data["results"]
        cuhk_data_action = cuhk_data["class"]
        video_list = list(val_dict.keys())
        video_dict = val_dict
    else:
        print('using test, not val to run detection code')
        cuhk_data = load_json("data/cuhk_test_simp_share_org.json")
        cuhk_data_score = cuhk_data["results"]
        cuhk_data_action = load_json("data/anet_action.json")

        video_list = list(test_dict.keys())
        video_dict = test_dict

    global result_dict
    result_dict = mp.Manager().dict()

    num_videos = len(video_list)
    num_videos_per_thread = int(num_videos / opt["post_process_thread"])
    processes = []
    for tid in range(opt["post_process_thread"] - 1):
        tmp_video_list = video_list[tid * num_videos_per_thread:(tid + 1) * num_videos_per_thread]
        p = mp.Process(target=_proposal2detection,
                       args=(opt, tmp_video_list, video_dict, cuhk_data_score, cuhk_data_action))
        p.start()
        processes.append(p)
    tmp_video_list = video_list[(opt["post_process_thread"] - 1) * num_videos_per_thread:]
    p = mp.Process(target=_proposal2detection,
                   args=(opt, tmp_video_list, video_dict, cuhk_data_score, cuhk_data_action))
    p.start()
    processes.append(p)
    for p in processes:
        p.join()

    result_dict = dict(result_dict)

    output_dict = {"version": "VERSION 1.3", "results": result_dict, "external_data": {}}
    outfile = open(os.path.join(opt['output'], "result_detect_cuhk_100_t1.json"), "w")
    json.dump(output_dict, outfile)
    outfile.close()


if __name__ == "__main__":
    import os
    import numpy as np

    import opts  # _thumos as opts
    from detection_result_generate_cuhk_share import generate_detection
    from eval import evaluation_proposal, evaluation_detection

    opt = opts.parse_opt()
    opt = vars(opt)
    generate_detection(opt)
    evaluation_detection(opt)
