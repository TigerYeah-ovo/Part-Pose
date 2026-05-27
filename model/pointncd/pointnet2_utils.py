import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from torch.nn import init
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape, device="cuda")
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    g = sample_gumbel(logits.size())
    y = logits + g
    return F.softmax(y / temperature, dim=-1)


def log_sinkhorn(K, a=None, b=None, eps=1.0, max_iter=10):
    m, n = K.shape
    v = K.new_zeros((m,))
    if a is None:
        a = 0#-math.log(m)
    else:
        a = torch.log(a)
    if b is None:
        b = math.log(m / n)#-math.log(n)
    else:
        b = torch.log(b)

    K = K / eps

    for _ in range(max_iter):
        u = -torch.logsumexp(v.view(m, 1) + K, dim=0) + b
        v = -torch.logsumexp(u.view(1, n) + K, dim=1) + a

    return torch.exp(K + u.view(1, n) + v.view(m, 1))



class OTVectorQuantizer(nn.Module):
    def __init__(self):
        super(OTVectorQuantizer, self).__init__()

        self.eps = 0.5
        self.ot_iter = 10
        self.temperature = 1
        self.temp = nn.Parameter(torch.tensor(1.0))

    def set_temperature(self, value):
        self.temperature = value

    def forward(self, z_from_encoder, codebook, temp, flg_train=True):
        bs, n, dim_z = z_from_encoder.shape
        z_from_encoder = F.normalize(z_from_encoder, p=2, dim=-1)
        z_flat = z_from_encoder.view(-1, dim_z)
        codebook_norm = F.normalize(codebook, p=2, dim=-1)
        logit = z_flat.mm(codebook_norm.T) * self.temp.exp()

       # probabilities = torch.softmax(logit, dim=-1)
        log_probabilities = torch.log_softmax(logit, dim=-1)
        with torch.no_grad():
            q_ot = log_sinkhorn(logit, eps=self.eps, max_iter=self.ot_iter)

        # Quantization
        if flg_train:
            encodings = gumbel_softmax_sample(logit, self.temperature)
            z_quantized = torch.mm(encodings, codebook).view(bs, n, dim_z)
            #avg_probs = torch.mean(probabilities.detach(), dim=0)
            att = encodings.reshape(bs, n, -1)
        else:
            # if flg_quant_det:
            indices = torch.argmax(logit, dim=1).unsqueeze(1)
            encodings_hard = torch.zeros(indices.shape[0], codebook.shape[0], device="cuda")
            encodings_hard.scatter_(1, indices, 1)
            #avg_probs = torch.mean(encodings_hard, dim=0)
            # else:
            #     dist = Categorical(probabilities)
            #     indices = dist.sample().view(bs, width, height)
            #     encodings_hard = F.one_hot(indices, num_classes=self.size_dict).type_as(codebook)
            #     avg_probs = torch.mean(probabilities, dim=0)
            z_quantized = torch.matmul(encodings_hard, codebook).view(bs, n, dim_z)
            att = encodings_hard.reshape(bs, n, -1)

        z_to_decoder = z_quantized.contiguous()

        # loss = -torch.sum(q_ot * log_probabilities) / bs / self.size_dict
        loss = -torch.mean(q_ot * log_probabilities)
        perplexity = 0 #torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))

        return z_to_decoder, att, loss, perplexity


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device=device)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(  # 加快训练
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)

def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.1):
    Q = torch.exp(out / epsilon).t() # K x B
    B = Q.shape[1]
    K = Q.shape[0]

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for _ in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True) #Q k,b k,1
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True) # b,1
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    Q = Q.t()

    indexs = torch.argmax(Q, dim=1)
    # Q = torch.nn.functional.one_hot(indexs, num_classes=Q.shape[1]).float()
    # Q = F.gumbel_softmax(Q, tau=0.5, hard=True)

    return Q, indexs

def distributed_sinkhorn_l1(out, sinkhorn_iterations=3, epsilon=0.03, l1_weight=0.1):
    Q = torch.exp(out / epsilon).t()  # K x B
    K = Q.shape[0]
    B = Q.shape[1]

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for _ in range(sinkhorn_iterations):
        # normalize each row: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True) + 1e-10  # K x B
        Q /= K

        # add L1 regularization to encourage sparsity
        Q *= torch.clamp(torch.norm(Q, p=1, dim=0, keepdim=True) - l1_weight, min=0)

        # normalize each column: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True) + 1e-10 # K x B
        Q /= B

    Q *= B  # the rows must sum to 1 so that Q is an assignment
    indexs = torch.argmax(Q, dim=1)
    # Q = torch.nn.functional.one_hot(indexs, num_classes=Q.shape[1]).float()
    # Q = F.gumbel_softmax(Q, tau=0.5, hard=True)

    return Q, indexs



def distributed_sinkhorn_topk(out, sinkhorn_iterations=3, epsilon=0.03, sparsity=2):
    Q = torch.exp(out / epsilon).t() # K x B
    B = Q.shape[1]
    K = Q.shape[0]

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for _ in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)+ 1e-10
        Q /= sum_of_rows
        Q /= K

        # apply top-k soft thresholding to each row
        Q = Q.t() # B,K
        topk_values, _ = torch.topk(Q, k=sparsity, dim=1)
        Q[Q < topk_values[:, [-1]]] = 0
        # Q[Q > topk_values[:, [-1]]] = 1/sparsity
        Q = Q.t()

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)+ 1e-10
        Q /= B

    Q *= B # the columns must sum to 1 so that Q is an assignment
    Q = Q.t()

    return Q


def distributed_sinkhorn_topk_grad(out, sinkhorn_iterations=3, epsilon=0.03, sparsity=2):
    Q = torch.exp(out / epsilon).t() # K x B
    B = Q.shape[1]
    K = Q.shape[0]

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q = Q / sum_Q

    for _ in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True) + 1e-10
        Q = Q / sum_of_rows
        Q = Q / K

        # apply top-k soft thresholding to each row
        Q = Q.t() # B,K
        topk_values, _ = torch.topk(Q, k=sparsity, dim=1)
        Q = Q * (Q > topk_values[:, [-1]]).float() # soft thresholding
        Q = F.gumbel_softmax(Q, tau=1, dim=1)  # Gumbel-Softmax trick
        Q = Q.t()

        # normalize each column: total weight per sample must be 1/B
        Q = Q / (torch.sum(Q, dim=0, keepdim=True) + 1e-10)
        Q = Q / B

    Q *= B # the columns must sum to 1 so that Q is an assignment
    Q = Q.t()

    return Q



def feat2prob(feat, center, alpha=1.0):
    q = 1.0 / (1.0 + torch.sum(
        torch.pow(feat.unsqueeze(1) - center, 2), 2) / alpha)
    q = q.pow((alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    return q

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_assignment(w.max() - w)
    #return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size

def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2

def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update

def init_prob_kmeans(model, eval_loader, n_clusters, args):
    # torch.manual_seed(args.seed)
    # model = model.to(args.device)
    # cluster parameter initiate
    from tqdm import tqdm
    model.eval()
    targets = np.zeros(len(eval_loader.dataset)* args.nb_primitives)
    feats = np.zeros((len(eval_loader.dataset) * args.nb_primitives, 1024))
    with torch.no_grad():
        for idx, (points, target) in tqdm(enumerate(eval_loader)):
            points = points.transpose(2, 1).cuda()
            outputs = model(points, target, None, 0, flag=0, purn=True)
            feats[idx * args.nb_primitives:(idx + 1) * args.nb_primitives, :] = outputs['ori_part'][0].data.cpu().numpy()
            targets[idx * args.nb_primitives:(idx + 1) * args.nb_primitives] = target.unsqueeze(1).repeat(1, args.nb_primitives).reshape(-1)
            # targets[idx] = label.data.cpu().numpy()
    # evaluate clustering performance
    # pca = PCA(n_components=n_clusters)
    # feats = pca.fit_transform(feats)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(feats)
    np.save('models/kmeans_{}_{}_{}.npy'.format(n_clusters, args.nb_primitives, args.number_points), kmeans.cluster_centers_)

    class_specific_cluster = []
    for i in range(args.known_class):
        class_feats = feats[targets == i]
        kmeans = KMeans(n_clusters=5, n_init=20)
        y_pred = kmeans.fit_predict(class_feats)
        class_specific_cluster.append(kmeans.cluster_centers_)
    class_specific_cluster = torch.cat(class_specific_cluster, dim=0)
    np.save('models/class_specific_kmeans_{}_{}_{}.npy'.format(n_clusters, args.nb_primitives, args.number_points), class_specific_cluster)


    # acc, nmi, ari = cluster_acc(targets, y_pred), nmi_score(targets, y_pred), ari_score(targets, y_pred)
    # print('Init acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    # probs = feat2prob(torch.from_numpy(feats), torch.from_numpy(kmeans.cluster_centers_))
    return  kmeans.cluster_centers_ #, probs, acc, nmi, ari,


class uniform_loss(nn.Module):
    def __init__(self, t=0.07):
        super(uniform_loss, self).__init__()
        self.t = t

    def forward(self, x):
        return x.matmul(x.T).div(self.t).exp().sum(dim=-1).log().mean()

from torch.autograd import Variable
import torch.optim as optim

def part_generation(num_part, emd_size, N_iter=1000):
    print("N =", num_part)
    print("M =", emd_size)
    criterion = uniform_loss()
    x = Variable(torch.randn(num_part, emd_size).float(), requires_grad=True)
    optimizer = optim.Adam([x], lr=1e-1)
    min_loss = 100
    optimal_target = None
    for i in range(N_iter):
        optimizer.zero_grad()
        x_norm = F.normalize(x, dim=1)
        loss = criterion(x_norm)
        if i % 100 == 0:
            print(i, loss.item())
        if loss.item() < min_loss:
            min_loss = loss.item()
            optimal_target = x_norm
        loss.backward()
        optimizer.step()
    import os
    os.makedirs('models', exist_ok=True)
    np.save('models/optimal_{}_{}.npy'.format(num_part, emd_size), optimal_target.detach().numpy())

    print("optimal loss = ", criterion(optimal_target).item())
    return optimal_target.detach()



class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256 ):
        super(ProjectionHead, self).__init__()


        self.proj = nn.Sequential(
                nn.Conv1d(dim_in, dim_in, 1),
                nn.BatchNorm1d(dim_in),
                nn.ReLU(),
                nn.Conv1d(dim_in, proj_dim, 1)
            )

    def forward(self, x):
        # return F.normalize(self.proj(x), p=2, dim=1)
        return self.proj(x)


class PrtAttLayer(nn.Module):
    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = nn.ReLU(inplace=True)

    def prt_interact(self, sem_prt):
        tgt2 = self.self_attn(sem_prt, sem_prt, value=sem_prt)[0]
        sem_prt = sem_prt + self.dropout1(tgt2)
        return sem_prt

    def prt_assign(self, vis_prt, vis_query):
        vis_prt = self.multihead_attn(query=vis_prt,  # 输出与query的大小一致
                                      key=vis_query,
                                      value=vis_query)[0]
        return vis_prt

    def prt_refine(self, vis_prt):
        new_vis_prt = self.linear2(self.activation(self.linear1(vis_prt)))
        return new_vis_prt + vis_prt

    def forward(self, vis_prt, vis_query):
        # sem_prt: 196*bs*c
        # vis_query: wh*bs*c
        vis_prt = self.prt_assign(vis_prt, vis_query)
        vis_prt = self.prt_refine(vis_prt)
        return vis_prt


class PrtClsLayer(nn.Module):
    def __init__(self, nc, na, dim):
        super().__init__()

        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

        self.fc1 = nn.Linear(dim, dim // na)
        self.fc2 = nn.Linear(dim // na, dim)

        self.weight_bias = nn.Parameter(torch.empty(nc, dim))
        nn.init.kaiming_uniform_(self.weight_bias, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.empty(nc))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_bias)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.activation = nn.ReLU()

    def prt_refine(self, prt):
        w = F.sigmoid(self.fc2(self.activation(self.fc1(prt))))
        prt = self.linear2(self.activation(self.linear1(prt)))
        prt = self.weight_bias + prt * w
        return prt

    def forward(self, query, cls_prt):
        cls_prt = self.prt_refine(cls_prt)
        logit = F.linear(query, cls_prt, self.bias)
        return logit, cls_prt

def knn2(x, y, k):
    inner = -2 * torch.matmul(x, y.transpose(1, 2))
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    yy = torch.sum(y ** 2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=1)[1]  # (batch_size, num_points, k)
    return idx
