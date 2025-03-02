import itertools
import random
from collections import defaultdict

import numpy as np
import skimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from roofmapnet.config import M
from roofmapnet.models.regression import MultitaskLearner
FEATURE_DIM = 8


class LineVectorizer(nn.Module):
    def __init__(self, depth, head,num_stacks, num_blocks, num_classes):
        super().__init__()
        self.backbone = MultitaskLearner(depth =depth, head=head,num_stacks = num_stacks, num_blocks = num_blocks, num_classes = num_classes)

        lambda_1 = torch.linspace(0, 1, M.n_pts0)[:, None]
        self.register_buffer("lambda_1", lambda_1)
        self.do_static_sampling = M.n_stc_posl + M.n_stc_negl > 0
        self.fc1 = nn.Conv2d(256, M.dim_loi, 1)
        scale_factor = M.n_pts0 // M.n_pts1
        if M.use_conv:
            self.pooling = nn.Sequential(
                nn.MaxPool1d(scale_factor, scale_factor),
                Bottleneck1D(M.dim_loi, M.dim_loi),
            )
            self.fc2 = nn.Sequential(
                nn.ReLU(inplace=True), nn.Linear(M.dim_loi * M.n_pts1 + FEATURE_DIM, 1)
            )
        else:
            self.pooling = nn.MaxPool1d(scale_factor, scale_factor)
            self.fc2 = nn.Sequential(
                nn.Linear(M.dim_loi * M.n_pts1 + FEATURE_DIM, M.dim_fc),
                nn.ReLU(inplace=True),
                nn.Linear(M.dim_fc, M.dim_fc),
                nn.ReLU(inplace=True),
                nn.Linear(M.dim_fc, 1),
            )
    
        self.conv1_line = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        
        self.conv2_line = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        
        self.conv3_line = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.fc_line = nn.Linear(64 * 64, 2)

        self.sigmoid_line = nn.Sigmoid()  

    def to_int(self,x):
        return tuple(map(int, x))

    def forward(self, input_dict):
        result = self.backbone(input_dict)
        h = result["preds"]
        x = self.fc1(result["feature"])
        att =  h["lmap"].unsqueeze(1)
        x = x * att + x
        n_batch, n_channel, row, col = x.shape

        xs, ys, fs, ps, idx, jcs, xsline, xaline,segs = [], [], [], [], [0], [],[],[],[]
        for i, meta in enumerate(input_dict["meta"]):
            p, label, feat, jc = self.sample_lines(
                meta, h["jmap"][i], h["joff"][i], input_dict["mode"]
            )
            ys.append(label)
            fs.append(feat)
            ps.append(p)

            p = p[:, 0:1, :] * self.lambda_1 + p[:, 1:2, :] * (1 - self.lambda_1) - 0.5
            p = p.reshape(-1, 2) 
            px, py = p[:, 0].contiguous(), p[:, 1].contiguous()
            px0 = px.floor().clamp(min=0, max=127)
            py0 = py.floor().clamp(min=0, max=127)
            px1 = (px0 + 1).clamp(min=0, max=127)
            py1 = (py0 + 1).clamp(min=0, max=127)
            px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

            xp = (
                (
                    x[i, :, px0l, py0l] * (px1 - px) * (py1 - py)
                    + x[i, :, px1l, py0l] * (px - px0) * (py1 - py)
                    + x[i, :, px0l, py1l] * (px1 - px) * (py - py0)
                    + x[i, :, px1l, py1l] * (px - px0) * (py - py0)
                )
                .reshape(n_channel, -1, M.n_pts0)
                .permute(1, 0, 2)
                )

            xa = (
                (
                    att[i, :, px0l, py0l] * (px1 - px) * (py1 - py)
                    + att[i, :, px1l, py0l] * (px - px0) * (py1 - py)
                    + att[i, :, px0l, py1l] * (px1 - px) * (py - py0)
                    + att[i, :, px1l, py1l] * (px - px0) * (py - py0)
                )
                .reshape(1, -1, M.n_pts0)
                .permute(1, 0, 2)
                )

            xsline.append(xp)
            xaline.append(xa)
            idx.append(idx[-1] + xp.shape[0])

        xline = torch.cat(xaline) 
        xline = self.relu1(self.bn1(self.conv1_line(xline)))
        xline = self.relu2(self.bn2(self.conv2_line(xline)))
        xline = self.relu3(self.bn3(self.conv3_line(xline)))
        xline = xline.view(xline.size(0), -1)
        xline = self.fc_line(xline)
        xline = self.sigmoid_line(xline)
        
        for j, p in enumerate(ps):
            xline_p = xline[idx[j]:idx[j+1]]
            xline_p = torch.ceil(xline_p * 64)
            re_line = xsline[j]
            f_slice_indices = xline_p[:, 0].long()
            b_slice_indices = xline_p[:, 1].long()
            for i in range(len(xline_p)):
                re_line[i, :, f_slice_indices[i]:b_slice_indices[i]] = 0
            re_line = self.pooling(re_line)
            xs.append(re_line)

        x, y = torch.cat(xs), torch.cat(ys)
        f = torch.cat(fs)
        x = x.reshape(-1, M.n_pts1 * M.dim_loi)
        x = torch.cat([x, f], 1)
        x = self.fc2(x).flatten()

        if input_dict["mode"] != "training":
            p = torch.cat(ps)
            s = torch.sigmoid(x)
            b = s > 0.1
            lines = []
            score = []
            for i in range(n_batch):
                p0 = p[idx[i] : idx[i + 1]]
                s0 = s[idx[i] : idx[i + 1]]
                mask = b[idx[i] : idx[i + 1]]
                p0 = p0[mask]
                s0 = s0[mask]
                if len(p0) == 0:
                    lines.append(torch.zeros([1, M.n_out_line, 2, 2], device=p.device))
                    score.append(torch.zeros([1, M.n_out_line], device=p.device))
                else:
                    arg = torch.argsort(s0, descending=True)
                    p0, s0 = p0[arg], s0[arg]
                    lines.append(p0[None, torch.arange(M.n_out_line) % len(p0)])
                    score.append(s0[None, torch.arange(M.n_out_line) % len(s0)])
            result["preds"]["lines"] = torch.cat(lines)
            result["preds"]["score"] = torch.cat(score)

        return result

    def sample_lines(self, meta, jmap, joff, mode):
        with torch.no_grad():
            junc = meta["junc"]
            jtyp = meta["jtyp"]  
            Lpos = meta["Lpos"]
            Lneg = meta["Lneg"]

            n_type = jmap.shape[0]
            jmap = non_maximum_suppression(jmap).reshape(n_type, -1)
            joff = joff.reshape(n_type, 2, -1)
            max_K = M.n_dyn_junc // n_type
            N = len(junc)
            if mode != "training":
                K = min(int((jmap > M.eval_junc_thres).float().sum().item()), max_K)
            else:
                K = min(int(N * 2 + 2), max_K)
            if K < 2:
                K = 2
            device = jmap.device
            score, index = torch.topk(jmap, k=K)
            y = (index // 128).float() + torch.gather(joff[:, 0], 1, index) + 0.5
            x = (index % 128).float() + torch.gather(joff[:, 1], 1, index) + 0.5

            # xy: [N_TYPE, K, 2]
            xy = torch.cat([y[..., None], x[..., None]], dim=-1)
            xy_ = xy[..., None, :]
            del x, y, index

            # dist: [N_TYPE, K, N]
            dist = torch.sum((xy_ - junc) ** 2, -1)
            cost, match = torch.min(dist, -1)

            # xy: [N_TYPE * K, 2]
            # match: [N_TYPE, K]
            for t in range(n_type):
                match[t, jtyp[match[t]] != t] = N
            match[cost > 1.5 * 1.5] = N
            match = match.flatten()

            _ = torch.arange(n_type * K, device=device)
            u, v = torch.meshgrid(_, _)
            u, v = u.flatten(), v.flatten()
            up, vp = match[u], match[v]
            label = Lpos[up, vp]

            c = (u < v).flatten()

            # sample lines
            u = u[c]
            v = v[c]
            label = label[c]

            xy = xy.reshape(n_type * K, 2)
            xyu, xyv = xy[u], xy[v]

            u2v = xyu - xyv
            u2v /= torch.sqrt((u2v ** 2).sum(-1, keepdim=True)).clamp(min=1e-6)
            feat = torch.cat(
                [
                    xyu / 128 * M.use_cood,
                    xyv / 128 * M.use_cood,
                    u2v * M.use_slop,
                    (u[:, None] > K).float(),
                    (v[:, None] > K).float(),
                ],
                1,
            )
            line = torch.cat([xyu[:, None], xyv[:, None]], 1)

            xy = xy.reshape(n_type, K, 2)
            jcs = [xy[i, score[i] > 0.03] for i in range(n_type)]
            return line, label.float(), feat, jcs


def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask


class Bottleneck1D(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Bottleneck1D, self).__init__()

        planes = outplanes // 2
        self.op = nn.Sequential(
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv1d(inplanes, planes, kernel_size=1),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes, planes, kernel_size=3, padding=1),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes, outplanes, kernel_size=1),
        )

    def forward(self, x):
        return x + self.op(x)
