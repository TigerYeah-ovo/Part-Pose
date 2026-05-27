import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.vmamba2.build_model import build_vssd_model
from model.mamba_gsc.Mamba import Mamba_gsc
from model.pointncd.pointnet2_utils import farthest_point_sample, index_points, knn2, l2_normalize, distributed_sinkhorn, \
    distributed_sinkhorn_topk, distributed_sinkhorn_l1, distributed_sinkhorn_topk_grad
from model.pointncd.dgcnn_PDG import Feature_Extractor_DGCNN, Classifier_DGCNN
from model.pointncd.pointnet_PDG import Feature_Extractor, Classifier, Projecter
from model.pointncd.pointnet2_utils import momentum_update
from geoopt.manifolds.stereographic import PoincareBall

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class PosE_Initial(nn.Module):
    def __init__(self, in_dim=3, out_dim=72, alpha=1000, beta=1000):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz):
        xyz = xyz.permute(0,2,1)
        B, _, N = xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim).float().cuda()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)

        return position_embed.permute(0,2,1)

def Poincare_dist(x, y, c=1.0):
    """
    Computes Poincare distance btw x and y
    Args:
        x (torch.Tensor): shape (n, d). n usually n_way*n_query
        y (torch.Tensor): shape (m, d). m usually n_way
    Returns:
        torch.Tensor: shape(n, m). For each query, the distances to each centroid
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    manifold = PoincareBall(c=c)

    # x = manifold.projx(x)
    # y = manifold.projx(y)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    hx = manifold.projx(x)
    hy = manifold.projx(y)

    return manifold.dist2(hx, hy)

def count_occurrence(knn_idx, pts_num, feat_dim):
    B = knn_idx.shape[0]
    counts = torch.zeros(B, pts_num, feat_dim, device=knn_idx.device)
    for b in range(B):
        flat_idx = knn_idx[b].contiguous().view(-1)  #  [N*K]
        batch_counts = torch.bincount(flat_idx, minlength=pts_num)
        batch_counts[batch_counts == 0] = 1
        batch_counts = batch_counts.unsqueeze(1).repeat(1, feat_dim)
        counts[b] = batch_counts
    return counts

def part_feat_restitution(knn_idx, part_feat, pts_num, feat_dim):
    """
    Input:
        knn_idx: knn indices, [B, n, K]
        part_feat: part feature of n key points, [B, n, C]
    Return:
        part_feat_map: restituted point-wise part feature, [B, N, C]
    """
    knn_idx = knn_idx.transpose(1, 2)
    B, N, K = knn_idx.shape
    device = part_feat.device
    
    center_features = part_feat.unsqueeze(2).repeat(1, 1, K, 1)
    
    neighbor_idx = knn_idx.contiguous().view(B, N * K)
    index = neighbor_idx.unsqueeze(-1).repeat(1, 1, feat_dim)
    
    src = center_features.contiguous().view(B, N * K, feat_dim)
    
    part_feat_map = torch.zeros(B, pts_num, feat_dim, dtype=torch.float32, device=device)
    part_feat_map.scatter_add_(dim=1, index=index, src=src)
    counts = count_occurrence(knn_idx, pts_num, feat_dim)
    
    return part_feat_map / counts

def patch_feat_visualize(knn_idx, part_feat_, pts_num, feat_dim, vis_layer, pts):
    """
    Input:
        knn_idx: knn indices, [B, n, K]
        part_feat: part feature of n key points, [B, n, C]
    Return:
        part_feat_map: restituted point-wise part feature, [B, N, C]
    """
    ret = {}
    pos = 48
    knn_idx = knn_idx.transpose(1, 2)
    B, N, K = knn_idx.shape
    device = part_feat_.device
    
    part_feat = part_feat_.cpu().detach()
    BB, NN, KK = part_feat.shape
    part_feat = part_feat.reshape(BB * NN, KK)

    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = StandardScaler()
    part_feat = scaler.fit_transform(part_feat)
    pca = PCA(n_components=3)
    part_feat = pca.fit_transform(part_feat)
    part_feat = min_max_scaler.fit_transform(part_feat)

    part_feat = part_feat.reshape(BB, NN, 3)
    part_feat = torch.from_numpy(part_feat)

    part_feat[:, :pos, :] = 0
    part_feat[:, pos:, :] = 0
    center_features = part_feat.unsqueeze(2).repeat(1, 1, K, 1)
        
    neighbor_idx = knn_idx.contiguous().view(B, N * K)
    index = neighbor_idx.unsqueeze(-1).repeat(1, 1, 3).cpu()
        
    src = center_features.contiguous().view(B, N * K, 3).cpu()
        
    part_feat_map = torch.zeros(B, pts_num, 3, dtype=torch.float64, device="cpu")
    part_feat_map.scatter_add_(dim=1, index=index, src=src)
    # part_feat_map = vis_layer(part_feat_map)

    ret['pts'] = pts.cpu().detach().numpy()
    ret['color'] = part_feat_map.cpu().detach().numpy()
    
    return ret

class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, knn_xyz, knn_x):
        B, _, G, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim).float().cuda()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=5).flatten(4)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).reshape(B, self.out_dim, G, K)

        # Weigh
        knn_x_w = torch.cat([knn_x, position_embed],dim=1)
        # knn_x_w *= position_embed

        return knn_x_w

class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta):
        super().__init__()
        # self.geo_extract = PosE_Geo(3, 72, alpha, beta)

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x): #(cent_xyz, cent_feat, part_xyz, part_feat)
        knn_x = knn_x.permute(0,2,1,3)
        # Normalize x (features) and xyz (coordinates)
        mean_x = lc_x.unsqueeze(dim=2)
        std_x = torch.std(knn_x - mean_x)

        mean_xyz = lc_xyz.unsqueeze(dim=2)
        std_xyz = torch.std(knn_xyz - mean_xyz)

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        # knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)

        # Geometry Extraction
        # knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        # knn_x = knn_x.permute(0, 3, 1, 2)
        # knn_x_w = self.geo_extract(knn_xyz, knn_x)

        return knn_x
        #return knn_x_w.permute(0, 2, 3, 1)

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        # self.fc_q = nn.Linear(d_model, h * d_k)
        # self.fc_k = nn.Linear(d_model, h * d_k)
        # self.fc_v = nn.Linear(d_model, h * d_v)
        # self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        # self.init_weights()

    # def init_weights(self):
    #     nn.init.xavier_uniform_(self.fc_q.weight)
    #     nn.init.xavier_uniform_(self.fc_k.weight)
    #     nn.init.xavier_uniform_(self.fc_v.weight)
    #     nn.init.xavier_uniform_(self.fc_o.weight)
    #     nn.init.constant_(self.fc_q.bias, 0)
    #     nn.init.constant_(self.fc_k.bias, 0)
    #     nn.init.constant_(self.fc_v.bias, 0)
    #     nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, mode='known'):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[0]

        # dot
        # q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        # k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        # v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        #cos
        keys = (keys).unsqueeze(0).repeat(b_s, 1, 1)
        q = l2_normalize(queries.view(b_s, nq, self.d_k))  # (b_s, h, nq, d_k)
        k = l2_normalize(keys.view(b_s, nk, self.d_k)).permute(0, 2, 1)  # (b_s, h, d_k, nk)
        values = (values).unsqueeze(0).repeat(b_s, 1, 1)
        att = torch.matmul(q, k) #/ np.sqrt(self.d_k)  # (b_s, h, nq, nk) #FIXME cos similarity

        # # hyperbolic
        # q = queries.view(-1, self.d_model)  # (b_s, h, nq, d_k)
        # k = keys.view(-1, self.d_model) # (b_s, h, d_k, nk)
        # q = self.fc_q(q)
        # k = self.fc_k(k)
        # att = Poincare_dist(q, k)

        if mode == 'known':
            with torch.no_grad():
                # sk_att = distributed_sinkhorn_topk(att.detach().reshape(-1, att.shape[-1]), 30, sparsity=5)  # q:n,m  index:n
                # sk_att = sk_att.reshape(b_s, self.h, nq, nk)
                # topk_values, _ = torch.topk(att, k=5, dim=-1)
                # attention_mask = att < topk_values[..., [-1]]
                sk_att = None
                attention_mask = None #att<0
                # attention_mask = sk_att == 0
        else:
            attention_mask = None
            sk_att = None
        att2 = att / 0.01
        # att2 = att
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att2 = att2.masked_fill(attention_mask, -np.inf)
        att2 = torch.softmax(att2, -1)
        out = torch.matmul(att2, values).contiguous()# (b_s, nq, h*d_v)
        # out = self.fc_o(out)  # (b_s, nq, d_model)
        att2 = att2.view(b_s, nq, -1)
        out = out.view(b_s, nq, self.d_model)
        return out, att2, sk_att


class get_part_feat_relate(nn.Module):
    def __init__(self, k=8, num_points=512, input_dim=512, emb_dim=1024, cfg_mafa=None):
        super(get_part_feat_relate, self).__init__()
        self.CrossAtt = ScaledDotProductAttention(d_model=emb_dim, d_k=emb_dim, d_v=emb_dim, h=1)
        self.PartFormer = nn.TransformerEncoderLayer(emb_dim, 4, emb_dim, 0.5, batch_first=True)
        self.visulizer = nn.Linear(emb_dim * 2, 3, bias=False)

        #------在这里加入MAFA来影响CodeBook
        # self.MAFA = build_vssd_model(cfg_mafa)
        #---------------------------------

        self.PosEm = LGA(emb_dim*2, 1000, 1000)
        self.k = k
        self.num_points = num_points

        self.part_projection = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            # nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            # nn.ReLU(),
        )
    
    def forward(self, xyz, feat_xyz, part_concepts, cent_xyz=None, cent_feat=None, mode='known'):
        B = feat_xyz.shape[0]
        xyz = xyz.transpose(1, 2) # (B, 3, N) -> (B, N, 3)
        feat_xyz = feat_xyz.transpose(1, 2) # (B, C, N) -> (B, N, C)

        if cent_xyz == None:
            cent_index= farthest_point_sample(xyz, self.k) # (B, N)
            cent_xyz = index_points(xyz, cent_index) # (B, N, 3)
            cent_feat = index_points(feat_xyz, cent_index) # （B, N, C = 256）

        id = knn2(xyz, cent_xyz, self.num_points) # (B, N, K)
        
        
        # np.save(f"/data/yecl24/ExperimentalResults/6dpe/part_registration/log/pts.npy", xyz.cpu().numpy())
        # np.save(f"/data/yecl24/ExperimentalResults/6dpe/part_registration/log/id.npy", id.cpu().numpy())
        # np.save(f"/data/yecl24/ExperimentalResults/6dpe/part_registration/log/kpt.npy", cent_xyz.cpu().detach().numpy())
        # exit(0)


        part_feat = index_points(feat_xyz, id)

        part_xyz = index_points(xyz, id)
        part_xyz = part_xyz.transpose(1, 2).contiguous()
        # delta_xyz = self.cal_delta(part_xyz,cent_xyz, part_feat)

        # cent_xyz += delta_xyz
        # id = knn2(xyz, cent_xyz, self.num_points)
        # part_feat = index_points(feat_xyz, id)
        # part_xyz = index_points(xyz, id)
        # part_xyz = part_xyz.transpose(1, 2).contiguous()
        part_feat_max = torch.max(part_feat, 1)[0]

        part_related = self.PosEm(cent_xyz, cent_feat, part_xyz, part_feat)
        part_related = self.part_projection(part_related.reshape(part_related.shape[0] * self.k * self.num_points ,-1)).reshape(B, self.k, self.num_points, -1).contiguous()
        part_related = torch.max(part_related, 2)[0] # (B, kpt_num, C = 128)

        part_relate_emb = self.PartFormer(part_feat_max)
        #---使用MAFA先聚合邻域特征，让MAFA模块能够影响CodeBook
        # part_related = self.MAFA(part_related)
        #-------------
        transformed_part_feat, att2, part_target = self.CrossAtt(part_related, part_concepts, part_concepts, mode= mode)

        part_feat_map = None
        #---将部件特征返还到逐点上，得到部件特征图
        # part_feat_map = part_feat_restitution(id, torch.cat((transformed_part_feat, part_relate_emb), dim=-1), xyz.shape[1], transformed_part_feat.shape[-1] + part_relate_emb.shape[-1])
        #---

        vis_list = None
        #---部件特征可视化
        # vis_list = patch_feat_visualize(id, torch.cat((transformed_part_feat, part_relate_emb), dim=-1), xyz.shape[1], transformed_part_feat.shape[-1] + part_relate_emb.shape[-1], self.visulizer, xyz)
        #-----------

        return transformed_part_feat, cent_xyz, part_feat_max, part_xyz, att2, part_target, part_relate_emb, part_related, part_feat_map, vis_list

class get_model(nn.Module):
    def __init__(self, args, num_class, num_unknown_class, cfg_mafa=None, normal_channel=False):
        super(get_model, self).__init__()
        self.args = args
        # args.emb_dims = 256

        model_name = 'pointnet'
        if model_name == 'pointnet':
            self.encoder = Feature_Extractor(args.emb_dims)
            # self.cls = Projecter(args.emb_dims , num_class)
            # self.novel_cls = Projecter(args.emb_dims , num_unknown_class)
            # self.cls_part = Projecter(args.emb_dims, num_class)
            # self.novel_part_cls = Projecter(args.emb_dims, num_unknown_class)
        # elif model_name == 'dgcnn':
        #     self.encoder = Feature_Extractor_DGCNN(args.emb_dims)
        #     self.cls = Projecter(args.emb_dims, num_class)
        #     self.novel_cls = Projecter(args.emb_dims, num_unknown_class)
        # elif model_name == 'pointnext':
        #     self.encoder = pointnext_PDG.get_model()
        #     self.cls = Projecter(args.emb_dims, num_class)
        #     self.novel_cls = Projecter(args.emb_dims, num_unknown_class)
        # elif model_name == 'pointnet2':


        if os.path.exists('models/optimal_{}_{}.npy'.format(args.part_concept_num * num_class, args.emb_dims)):
            part_gen = np.load('models/optimal_{}_{}.npy'.format(args.part_concept_num * num_class, args.emb_dims))
        else:
            from model.pointncd.pointnet2_utils import part_generation
            part_gen = part_generation(args.part_concept_num * num_class, args.emb_dims)

        self.part_prototypes = nn.Parameter(torch.Tensor(part_gen), requires_grad=True)
        # torch.nn.init.xavier_normal_(self.part_prototypes)
        self.get_part_feat = get_part_feat_relate(k=args.part_num, num_points=args.part_point_num, input_dim=args.kpt_feat_dim, emb_dim=args.emb_dims, cfg_mafa=cfg_mafa)

        self.projection = nn.Sequential(
            nn.Linear(args.emb_dims , args.emb_dims),
            nn.BatchNorm1d(args.emb_dims),
            # nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(args.emb_dims, args.emb_dims),
            nn.BatchNorm1d(args.emb_dims),
            # nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(args.emb_dims, args.emb_dims),
            # nn.ReLU(),
        )


        self.cat_prototypes = nn.Parameter(torch.zeros(num_class, args.emb_dims),
                                           requires_grad=True)
        self.novel_prototypes = nn.Parameter(torch.zeros(num_unknown_class, args.emb_dims),
                                             requires_grad=True)
        self.train_flag = True
        #
        # trunc_normal_(self.part_prototypes, std=100, a=-100, b=100)
        # trunc_normal_(self.novel_prototypes, std=100, a=-100, b=100)
        torch.nn.init.xavier_normal_(self.cat_prototypes)
        torch.nn.init.xavier_normal_(self.novel_prototypes)

        self.gamma = 0.6

    def forward(self, pts, point_local_feat, kpt=None, kpt_feat=None, flag=0):
        B = pts.shape[0]
        if point_local_feat is not None:
            points_feat = point_local_feat
        else:    
            points_feat = self.encoder(pts)  # (B, C = 256, N = 1024)
            
        if flag == 0:   # att.shape[2] == num_classes * part_concept_num
            q1, cent_q1, part_feat, part_xyz, att, part_target, part_pos_emb, part_related, part_feat_map, vis_list = self.get_part_feat(pts, points_feat, self.part_prototypes, kpt, kpt_feat, mode='known')

        part_xyz = part_xyz.view(B, -1, 3)
        part_score = att.sum(1)# b,100

        if flag == 0:

            feat_q1 = self.multi_pool(q1, part_pos_emb, part_related)
            # feat_q1 = self.cls(feat_q1)
            feat_q1 = l2_normalize(feat_q1)
            out_q1 = torch.einsum('bc,kc->bk', feat_q1,
                                 l2_normalize(self.cat_prototypes))

            return {#'logits': out_q1,
                    'part_logits': att,
                    # 'part_target': part_target,
                    # 'vis_emd': feat_q1,
                    'part_protos': l2_normalize(self.part_prototypes),
                    # 'feature': feat_q1,
                    # 'res_points': None,
                    # 'part_xyz': part_xyz,
                    # 'cat_protos': l2_normalize(self.cat_prototypes),
                    # 'novel_protos': l2_normalize(self.novel_prototypes),
                    # 'ori_part': part_feat,
                    'part_feat_map': part_feat_map,
                    'pos_inv_part_feat': q1,
                    'pos_awa_part_feat': part_pos_emb,
                    # 'part_score': part_score,
                    # 'cent_idx':cent_q1,
                    'vis_list': vis_list
                    }
    

    def update_usage(self, part_score, train=False):
        # part_score B,part,code
        if train:
            with torch.no_grad():
                part_score = part_score.reshape(-1, part_score.shape[-1])
                topk_values, topk_indices = torch.topk(part_score, 2, dim=1)
                result = torch.sum(torch.where(part_score.scatter(1, topk_indices, 1)== 1, 1, 0), dim=0)
                mask = (result != 0).float()

                self.code_usage = 0.99 * self.code_usage + (1 - 0.99) * mask
                # print('update_usage', self.code_usage)
                # self.code_usage_cur[self.code_usage_cur != 0] = 0
                #
                # self.code_usage_cur += mask

    def update_usage_epoch(self):
            cur_mask = (self.code_usage_cur != 0).float()
            self.code_usage = 0.99 * self.code_usage + (1 - 0.99) * cur_mask
            print('update_usage', self.code_usage)
            self.code_usage_cur[self.code_usage_cur != 0] = 0
            # self.code_usage += torch.mean(torch.masked_fill(part_score, mask, 0), dim=0)
            # self.code_usage /= 2
    def _tile(self, x):
        n, c = x.shape
        code_size = self.part_prototypes.shape[0]
        idx = torch.randperm(n)
        x = x[idx]
        if n < code_size and n>0:
            n_repeats = code_size // n
            remainder = code_size % n
            std = 0.01 / np.sqrt(c)
            x = torch.cat([x] * n_repeats + [x[:remainder]], dim=0)
            x = x + torch.randn_like(x) * std
        elif n == 0:
            x = self.part_prototypes
        else:
            x = x[:code_size]
        return x

    def update_code(self, dis_sim_part_feat, train):
        if train:
            with torch.no_grad():
                mask = (self.code_usage < 0.4).float().unsqueeze(-1)
                if len(dis_sim_part_feat) > 0 and torch.sum(mask>0) > 0:
                    # print('update_code', torch.sum(mask>0), self.code_usage)
                    dis_sim_part_feat = self._tile(dis_sim_part_feat)
                    protos = self.part_prototypes.data.clone()
                    protos = (1-mask) * protos + mask * dis_sim_part_feat
                    self.part_prototypes = nn.Parameter(protos, requires_grad=True)
                    self.code_usage[mask.squeeze().bool()] = 1
            # usage也要变一下

    def multi_pool(self, feat, part_pos_emb, part_relate):
        B, N, D = feat.shape
        feat = torch.cat([feat], dim=-1).reshape(B * N, -1)
        feat = self.projection(feat).reshape(B, N, -1)
        pooled_x_max = torch.max(feat, 1)[0]
        return pooled_x_max



