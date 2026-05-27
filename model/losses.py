import torch
import torch.nn.functional as F

def SmoothL1Dis(p1, p2, threshold=0.1):
    '''
    p1: b*n*3
    p2: b*n*3
    '''
    diff = torch.abs(p1 - p2)
    less = torch.pow(diff, 2) / (2.0 * threshold)
    higher = diff - threshold / 2.0
    dis = torch.where(diff > threshold, higher, less)
    dis = torch.mean(torch.sum(dis, dim=2))
    return dis

def ChamferDis(p1, p2):
    '''
    p1: b*n1*3
    p2: b*n2*3
    '''
    dis = torch.norm(p1.unsqueeze(2) - p2.unsqueeze(1), dim=3)
    dis1 = torch.min(dis, 2)[0]
    dis2 = torch.min(dis, 1)[0]
    dis = 0.5*dis1.mean(1) + 0.5*dis2.mean(1)
    return dis.mean()

def ChamferDis_wo_Batch(p1, p2):
    """
    Args:
        p1: (n1, 3)
        p2: (n2, 3)
    """
    dis = torch.norm(p1.unsqueeze(1) - p2.unsqueeze(0), dim=2) # (n1, n2)
    dis1 = torch.min(dis, 1)[0] # (n1, )
    dis2 = torch.min(dis, 0)[0] # (n2, )
    dis = 0.5*dis1.mean() + 0.5*dis2.mean()
    return dis

def PoseDis(r1, t1, s1, r2, t2, s2):
    '''
    r1, r2: b*3*3
    t1, t2: b*3
    s1, s2: b*3
    '''
    dis_r = torch.mean(torch.norm(r1 - r2, dim=1))
    dis_t = torch.mean(torch.norm(t1 - t2, dim=1))
    dis_s = torch.mean(torch.norm(s1 - s2, dim=1))

    return dis_r + dis_t + dis_s

def UniChamferDis(p1, p2):
    '''
    p1: b, n1, 3
    p2: b, n2, 3
    '''
    # (b, n1, n2)
    dis = torch.norm(p1.unsqueeze(2) - p2.unsqueeze(1), dim=3)
    dis = torch.min(dis, 2)[0]

    return dis.mean()

#--------------------
def PartProtoDiff(part_prototype, dis=0.0):
    part_diff = torch.einsum('mc,nc->mn', part_prototype, part_prototype) - torch.eye(part_prototype.shape[0],part_prototype.shape[0]).cuda() - dis
    loss_diff = torch.sum(part_diff[part_diff>0])

    return loss_diff

def Consistency_Loss(feats, att):
    B, N, K = att.shape
    consistency_loss = 0.0
    feats = feats.reshape(B * N, -1)
    att = att.reshape(B * N, -1)
    for i in range(K):
        indices = (att.argmax(1) == i)
        need = feats[indices]
        if indices.sum() > 0:
            consistency_loss += F.cosine_similarity(need[None, :, :], need[:, None, :], dim=-1).mean()
    return 1 - consistency_loss / K


def Distinctiveness_Loss(feats, att):
    B, N, K = att.shape
    records = []
    feats = feats.reshape(B * N, -1)
    att = att.reshape(B * N, -1)
    for i in range(K):
        indices = (att.argmax(1) == i)
        need = feats[indices]
        if indices.sum() > 0:
            records.append(torch.mean(need, dim=0, keepdim=True))
    records = torch.cat(records, dim=0)
    distinctive_loss = F.cosine_similarity(records[None, :, :], records[:, None, :], dim=-1).mean()
    return distinctive_loss
#--------------------