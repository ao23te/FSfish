import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from pytorch_metric_learning.losses import NormalizedSoftmaxLoss
from pytorch_metric_learning.losses import ProxyAnchorLoss, SoftTripleLoss, ArcFaceLoss, CircleLoss, ProxyNCALoss
from torch.backends import cudnn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from scipy.spatial.distance import cdist

sys.path.append('..')
from model import Extractor_mixed
from utils import NormDataset_mixed

# for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False
torch.backends.cudnn.enabled = False


def map_sake(gt_labels_gallery, predicted_features_query, gt_labels_query, scores, k=None):
    mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    mean_mAP = []
    for fi in range(predicted_features_query.shape[0]):
        mapi = eval_ap(gt_labels_query[fi], scores[fi], gt_labels_gallery, top=k)
        mAP_ls[gt_labels_query[fi]].append(mapi)
        mean_mAP.append(mapi)
    return mean_mAP


def prec_sake(gt_labels_gallery, predicted_features_query, gt_labels_query, scores, k=None):
    # compute precision for two modalities
    prec_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    mean_prec = []
    for fi in range(predicted_features_query.shape[0]):
        prec = eval_precision(gt_labels_query[fi], scores[fi], gt_labels_gallery, top=k)
        prec_ls[gt_labels_query[fi]].append(prec)
        mean_prec.append(prec)
    return np.nanmean(mean_prec)


def eval_ap(inst_id, scores, gt_labels, top=None):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]
    tot_pos = np.sum(pos_flag)

    sort_idx = np.argsort(-scores)
    tp = pos_flag[sort_idx]
    fp = np.logical_not(tp)

    if top is not None:
        top = min(top, tot)
        tp = tp[:top]
        fp = fp[:top]
        tot_pos = min(top, tot_pos)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    try:
        rec = tp / tot_pos
        prec = tp / (tp + fp)
    except:
        return np.nan

    ap = voc_ap(rec, prec)
    return ap


def voc_ap(rec, prec):
    mrec = np.append(0, rec)
    mrec = np.append(mrec, 1)

    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)

    for ii in range(len(mpre) - 2, -1, -1):
        mpre[ii] = max(mpre[ii], mpre[ii + 1])

    msk = [i != j for i, j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk] - mrec[0:-1][msk]) * mpre[1:][msk])
    return ap


def eval_precision(inst_id, scores, gt_labels, top=5):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]
    top = min(top, tot)
    sort_idx = np.argsort(-scores)
    return np.sum(pos_flag[sort_idx][:top]) / top


def sake_metric(predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, k=None):
    if k is None:
        k = {'precision': 5, 'map': predicted_features_gallery.shape[0]}
    if k['precision'] is None:
        k['precision'] = 5
    if k['map'] is None:
        k['map'] = predicted_features_gallery.shape[0]

    scores = -cdist(predicted_features_query, predicted_features_gallery, metric='cosine')
    gt_labels_query = gt_labels_query.flatten()
    gt_labels_gallery = gt_labels_gallery.flatten()
    aps = map_sake(gt_labels_gallery, predicted_features_query, gt_labels_query, scores, k=k['map'])
    prec = prec_sake(gt_labels_gallery, predicted_features_query, gt_labels_query, scores, k=k['precision'])
    return sum(aps) / len(aps), prec


def compute_metric(sketch_vectors, sketch_labels, photo_vectors, photo_labels):
    acc = {}

    photo_vectors = photo_vectors.numpy()
    sketch_vectors = sketch_vectors.numpy()
    photo_labels = photo_labels.numpy()
    sketch_labels = sketch_labels.numpy()
    map_all, p_5 = sake_metric(photo_vectors, photo_labels, sketch_vectors, sketch_labels)
    map_10, p_10 = sake_metric(photo_vectors, photo_labels, sketch_vectors, sketch_labels,
                                 {'precision': 10, 'map': 10})

    acc['P@5'], acc['P@10'], acc['mAP@10'], acc['mAP@all'] = p_5, p_10, map_10, map_all
    # the mean value is chosen as the representative of precise
    acc['precise'] = (acc['P@5'] + acc['P@10'] + acc['mAP@10'] + acc['mAP@all']) / 4
    return acc


# train for one epoch
def train(backbone, data_loader):
    backbone.train()
    total_extractor_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    for c1,c2,c3, label in train_bar:
        c1,c2,c3, label = c1.cuda(),c2.cuda(),c3.cuda(), label.cuda()
        # extractor #
        optimizer_extractor.zero_grad()
        img_proj = backbone(c1,c2,c3)

        # extractor loss
        class_loss =class_criterion(img_proj,label)
        total_extractor_loss += class_loss.item() * c1.size(0)

        class_loss.backward()
        optimizer_extractor.step()

        total_num += c1.size(0)
        e_loss = total_extractor_loss / total_num
        train_bar.set_description('Train Epoch: [{}/{}] E-Loss: {:.4f}'.format(epoch, epochs, e_loss))
    return e_loss


# # val for one epoch
def val(backbone, query_loader, gallery_loader):
    backbone.eval()
    query_vectors, query_labels = [], []
    gallery_vectors, gallery_labels = [], []
    with torch.no_grad():
        for c1,c2,c3, label in tqdm(query_loader, desc='Query Feature extracting', dynamic_ncols=True):
            c1,c2,c3 = c1.cuda(),c2.cuda(),c3.cuda()
            img_emb = backbone(c1,c2,c3)
            query_vectors.append(img_emb.cpu())
            query_labels.append(label)
        query_vectors = torch.cat(query_vectors, dim=0)
        query_labels = torch.cat(query_labels, dim=0)

        for c1, c2, c3, label in tqdm(gallery_loader, desc='Gallery Feature extracting', dynamic_ncols=True):
            c1, c2, c3 = c1.cuda(), c2.cuda(), c3.cuda()
            img_emb = backbone(c1, c2, c3)
            gallery_vectors.append(img_emb.cpu())
            gallery_labels.append(label)
        gallery_vectors = torch.cat(gallery_vectors, dim=0)
        gallery_labels = torch.cat(gallery_labels, dim=0)

        acc = compute_metric(query_vectors, query_labels, gallery_vectors,gallery_labels)
        results['P@5'].append(acc['P@5'] * 100)
        results['P@10'].append(acc['P@10'] * 100)
        results['mAP@10'].append(acc['mAP@10'] * 100)
        results['mAP@all'].append(acc['mAP@all'] * 100)
        print('Val Epoch: [{}/{}] | P@5:{:.1f}% | P@10:{:.1f}% | mAP@10:{:.1f}% | mAP@all:{:.1f}%'
              .format(epoch, epochs, acc['P@5'] * 100, acc['P@10'] * 100, acc['mAP@10'] * 100,
                      acc['mAP@all'] * 100))
    return acc['precise']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    # common args
    parser.add_argument('--data_root', default='C:\\Users\\ouc\\Desktop\\fish_reid\\fish_reid\\reid_data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='fish_reid', type=str, choices=['fish_reid', 'C1','C2','C3'],
                        help='Dataset name')
    parser.add_argument('--backbone_type', default='resnet50', type=str, choices=['resnet50', 'vgg16'],
                        help='Backbone type')
    parser.add_argument('--loss_function', default='softtriple', type=str, choices=['normsoftmax', 'softtriple', 'proxyanchor','circleloss',
                                                                                    'arcfaceloss','proxynca'],
                        help='Backbone type')
    parser.add_argument('--emb_dim', default=512, type=int, help='Embedding dim')
    parser.add_argument('--batch_size', default=32, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=40, type=int, help='Number of epochs over the model to train')
    parser.add_argument('--warmup', default=1, type=int, help='Number of warmups over the extractor to train')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')

    # args parse
    args = parser.parse_args()
    data_root, data_name, backbone_type, emb_dim = args.data_root, args.data_name, args.backbone_type, args.emb_dim
    batch_size, epochs, save_root = args.batch_size, args.epochs, args.save_root

    # data prepare
    train_data = NormDataset_mixed(data_root, data_name, split='train')
    query_data = NormDataset_mixed(data_root, data_name, split='query')
    gallery_data = NormDataset_mixed(data_root, data_name, split='gallery')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    query_loader = DataLoader(query_data, batch_size=batch_size, shuffle=False, num_workers=8)
    gallery_loader = DataLoader(gallery_data, batch_size=batch_size, shuffle=False, num_workers=8)

    # model define
    extractor = Extractor_mixed(backbone_type, emb_dim).cuda()
    # loss setup
    if args.loss_function=="normsoftmax":
        class_criterion = NormalizedSoftmaxLoss(train_data.train_zzq_tmp_id, emb_dim).cuda()
    elif args.loss_function=="softtriple":
        class_criterion = SoftTripleLoss(train_data.train_zzq_tmp_id, emb_dim).cuda()
    elif args.loss_function == "proxyanchor":
        class_criterion = ProxyAnchorLoss(train_data.train_zzq_tmp_id, emb_dim).cuda()
    elif args.loss_function == "circleloss":
        class_criterion = CircleLoss(train_data.train_zzq_tmp_id,emb_dim).cuda()
    elif args.loss_function == "arcfaceloss":
        class_criterion = ArcFaceLoss(train_data.train_zzq_tmp_id,emb_dim).cuda()
    elif args.loss_function == "proxynca":
        class_criterion = ProxyNCALoss(train_data.train_zzq_tmp_id, emb_dim).cuda()
    else:
        print("unimplemented")
        # optimizer config
    optimizer_extractor = Adam(extractor.parameters(), lr=1e-4)
    # training loop
    results = {'extractor_loss': [], 'precise': [], 'P@5': [], 'P@10': [], 'mAP@10': [], 'mAP@all': []}
    save_name_pre = '{}_{}_{}_{}'.format(args.loss_function, data_name, backbone_type, emb_dim)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    best_precise = 0.0
    for epoch in range(1, epochs + 1):

        for param in list(extractor.backbone.parameters())[:-2]:
            param.requires_grad = False if epoch <= args.warmup else True
        extractor_loss = train(extractor, train_loader)
        results['extractor_loss'].append(extractor_loss)
        precise = val(extractor, query_loader, gallery_loader)
        results['precise'].append(precise * 100)

        # # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='epoch')
        #
        if precise > best_precise:
            best_precise = precise
            torch.save(extractor.state_dict(), '{}/{}_extractor.pth'.format(save_root, save_name_pre))