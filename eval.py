# -*- coding: utf-8 -*-
import sys, os
sys.path.append('./Evaluation')
from eval_proposal import ANETproposal
from eval_detection import ANETdetection
import matplotlib.pyplot as plt
import numpy as np

def plot_metric(opt,average_nr_proposals, average_recall, recall, tiou_thresholds=np.linspace(0.5, 0.95, 10)):

    fn_size = 14
    plt.figure(num=None, figsize=(12, 8))
    ax = plt.subplot(1,1,1)
    
    colors = ['k', 'r', 'yellow', 'b', 'c', 'm', 'b', 'pink', 'lawngreen', 'indigo']
    area_under_curve = np.zeros_like(tiou_thresholds)
    for i in range(recall.shape[0]):
        area_under_curve[i] = np.trapz(recall[i], average_nr_proposals)

    for idx, tiou in enumerate(tiou_thresholds[::2]):
        ax.plot(average_nr_proposals, recall[2*idx,:], color=colors[idx+1],
                label="tiou=[" + str(tiou) + "], area=" + str(int(area_under_curve[2*idx]*100)/100.), 
                linewidth=4, linestyle='--', marker=None)
    # Plots Average Recall vs Average number of proposals.
    ax.plot(average_nr_proposals, average_recall, color=colors[0],
            label="tiou = 0.5:0.05:0.95," + " area=" + str(int(np.trapz(average_recall, average_nr_proposals)*100)/100.), 
            linewidth=4, linestyle='-', marker=None)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1]] + handles[:-1], [labels[-1]] + labels[:-1], loc='best')
    
    plt.ylabel('Average Recall', fontsize=fn_size)
    plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
    plt.grid(b=True, which="both")
    plt.ylim([0, 1.0])
    plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
    plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
    #plt.show()    
    plt.savefig(opt['output']+'/'+opt["save_fig_path"])


def evaluation_proposal(opt):
    app = ANETproposal(ground_truth_filename="./Evaluation/data/activity_net_1_3_new.json",
                        proposal_filename=os.path.join(opt['output'], opt["result_file"]),
                        tiou_thresholds=np.linspace(0.5, 0.95, 10),
                        max_avg_nr_proposals=100,
                        subset='validation', verbose=True, check_status=False)
    app.evaluate()
    parent_path, run_id = os.path.split(os.path.normpath(opt['output']))
    results = (f'[{run_id}|Proposals]'
               f' AUC {app.auc*100:.3f}'
               f' AR@1 {np.mean(app.recall[:,0])*100:.3f}'
               f' AR@5 {np.mean(app.recall[:,4])*100:.3f}'
               f' AR@10 {np.mean(app.recall[:,9])*100:.3f}'
               f' AR@100 {np.mean(app.recall[:,-1])*100:.3f}')
    print(results)
    with open(os.path.join(parent_path, 'results.txt'), 'a') as fobj:
        fobj.write(f'{results}\n')

    #plot_metric(opt, app.proposals_per_video, app.avg_recall, app.recall)

def evaluation_detection(opt):
    app = ANETdetection(ground_truth_filename="./Evaluation/data/activity_net_1_3_new.json",
                        prediction_filename=os.path.join(opt['output'], "result_detect_cuhk_100_t1.json"),
                        subset='validation', verbose=True, check_status=False)
    app.evaluate()
    parent_path, run_id = os.path.split(os.path.normpath(opt['output']))
    mAP_at_tIoU = [f'mAP@{t:.2f} {mAP*100:.3f}' for t, mAP in zip(app.tiou_thresholds, app.mAP)]
    results = f'[{run_id}|Detection] average-mAP {app.average_mAP*100:.3f} {" ".join(mAP_at_tIoU)}'
    print(results)
    with open(os.path.join(parent_path, 'results.txt'), 'a') as fobj:
        fobj.write(f'{results}\n')

def evaluation_detection_testset():
    app = ANETdetection(ground_truth_filename="./activity_net_test.v1-3.min.json",
                        prediction_filename="./output/result_detect_cuhk_100_t1.json",
                        subset='test', verbose=True, check_status=False)

    app.evaluate()

if __name__ == "__main__":
    evaluation_detection_testset()
