import torch
import tqdm
from torch.optim import Adam
from time import time
from torch.cuda.amp import autocast, GradScaler
import opt
from utils import *
from encoder import *
from McMVC import McMVC
from data_loader import load_data
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        #mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
class Loss(nn.Module):
    def __init__(self, batch_size, temperature_f, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature_f = temperature_f
        self.device = device
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def Structure_guided_Contrastive_Loss(self, h_i, h_j, S):
        S_1 = S.repeat(2, 2)
        all_one = torch.ones(self.batch_size*2, self.batch_size*2).to('cuda')
        S_2 = all_one - S_1
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)
        sim = torch.matmul(h, h.T) / self.temperature_f
        sim1 = torch.multiply(sim, S_2)
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim1[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
def FL_Loss1(clusters, h1, centers, label, h2=[], temperature=0.5):
    """
    Feature-level contrastive learning
    """
    loss = 0
    h1 = F.normalize(h1, p=2, dim=1)
    h2 = F.normalize(h2, p=2, dim=1)
    centers = F.normalize(centers, p=2, dim=1)

    indicator = torch.ones(h1.size(0), clusters).to(h1.device)
    for i in range(clusters):
        indicator[i, label[i]] = 0

    sim_positive = torch.exp(torch.mm(h1, h2.T) / temperature)
    sim_negative= torch.mul(
        torch.exp(torch.mm(h1, centers.T) / temperature), indicator)
    # sim_negative2 = torch.mul(
    #     torch.exp(torch.mm(h2, centers.T) / temperature), indicator)
    # sim_negative = sim_negative1+sim_negative2
    loss = - torch.log(torch.diagonal(sim_positive) /
                       (torch.diagonal(sim_positive) + torch.sum(sim_negative, dim=1)))
    return loss.mean()


def FL_Lossnew( h1, centers, labels, h2=[], temperature=0.5):
    """
    Feature-level contrastive learning
    """
    # 归一化特征和中心
    h1 = F.normalize(h1, p=2, dim=1)
    h2 = F.normalize(h2, p=2, dim=1)
    centers = F.normalize(centers, p=2, dim=1)

    num_samples = h1.size(0)

    # 创建正样本和负样本的掩码
    labels = labels.contiguous().view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).float().to(h1.device)
    negative_mask = 1 - positive_mask

    # 计算正样本相似度（同一个簇）
    sim_positive = torch.exp(torch.mm(h1, h2.T) / temperature)
    sim_positive = sim_positive * positive_mask

    # 计算 h1 和 centers 的负样本相似度
    sim_negative_h1 = torch.exp(torch.mm(h1, centers.T) / temperature)
    sim_negative_h1 = torch.mm(negative_mask, sim_negative_h1)

    # 计算 h2 和 centers 的负样本相似度
    sim_negative_h2 = torch.exp(torch.mm(h2, centers.T) / temperature)
    sim_negative_h2 = torch.mm(negative_mask, sim_negative_h2)

    # 合并两个负样本相似度
    sim_negative = sim_negative_h1 + sim_negative_h2

    # 计算对比学习损失
    loss = -torch.log(torch.diagonal(sim_positive) /
                      (torch.diagonal(sim_positive) + torch.sum(sim_negative, dim=1)))
    return loss.mean()


def FL_Loss2(clusters, h1, centers, label, h2=[], temperature=0.5):
    """
    Feature-level contrastive learning
    """
    loss = 0
    h1 = F.normalize(h1, p=2, dim=1)
    h2 = F.normalize(h2, p=2, dim=1)
    centers = F.normalize(centers, p=2, dim=1)

    indicator = torch.ones(h1.size(0), clusters).to(h1.device)
    for i in range(clusters):
        indicator[i, label[i]] = 0

    sim_positive = torch.exp(torch.mm(h1, h2.T) / temperature)
    sim_negative1= torch.mul(
        torch.exp(torch.mm(h1, centers.T) / temperature), indicator)
    sim_negative2 = torch.mul(
        torch.exp(torch.mm(h2, centers.T) / temperature), indicator)
    sim_negative = sim_negative1+sim_negative2
    loss = - torch.log(torch.diagonal(sim_positive) /
                       (torch.diagonal(sim_positive) + torch.sum(sim_negative, dim=1)))
    return loss.mean()
def compute_contrastive_loss(H, labels, temperature=0.5, epsilon=1e-10):

    # 获取 H 的设备
    device = H.device

    # 将 labels 转换为 PyTorch 张量，并移动到相同的设备
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels, device=device)
    else:
        labels = labels.to(device)

    N, d = H.size()
    # 计算相似性矩阵
    sim_matrix = torch.matmul(H, H.t()) / temperature

    # 创建相同簇的掩码
    mask = labels.expand(N, N).eq(labels.expand(N, N).t())

    # 正样本相似性
    pos_sim = torch.exp(sim_matrix) * mask.float()
    pos_sim_sum = pos_sim.sum(dim=1) + epsilon  # 加上 epsilon 防止出现 NaN

    # 全部样本相似性（包括自己）
    exp_sim_matrix = torch.exp(sim_matrix)
    sim_sum = exp_sim_matrix.sum(dim=1) + epsilon  # 加上 epsilon 防止出现 NaN

    # 计算对比损失
    log_prob = torch.log(pos_sim_sum / sim_sum + epsilon)  # 加上 epsilon 防止 log(0)
    loss = -log_prob.mean()

    return loss

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, nb_layers=2, nb_units=256, concat_every_fcl=True):
        super(Generator, self).__init__()
        self.nb_layers = nb_layers
        self.concat_every_fcl = concat_every_fcl
        self.input_dim = input_dim

        # 生成全连接层和激活层的序列
        layers = []
        current_dim = input_dim

        for i in range(nb_layers):
            layers.append(nn.Linear(current_dim, nb_units))
            layers.append(nn.LeakyReLU(0.2))
            if concat_every_fcl:
                current_dim = nb_units + input_dim  # 每一层之后拼接输入特征
            else:
                current_dim = nb_units

        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.Identity())  # 可以选择使用 nn.Tanh() 或其他激活函数，根据需要

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if self.concat_every_fcl:
            original_x = x
            for layer in self.network:
                if isinstance(layer, nn.Linear) and x.shape[1] != layer.in_features:
                    # 在全连接层前拼接原始输入
                    x = torch.cat([x, original_x], dim=1)
                x = layer(x)
            return x
        else:
            return self.network(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, nb_layers=2, nb_units=256):
        super(Discriminator, self).__init__()
        self.network = nn.ModuleList()  # 使用ModuleList来支持动态网络层添加

        # 添加第一层，使用LeakyReLU激活
        self.network.append(nn.Linear(input_dim, nb_units))
        self.network.append(nn.LeakyReLU(0.2))

        # 循环添加中间层
        for _ in range(nb_layers - 1):
            self.network.append(nn.Linear(nb_units, nb_units))
            self.network.append(nn.BatchNorm1d(nb_units))
            self.network.append(nn.Tanh())

        # 添加最后的输出层
        self.network.append(nn.Linear(nb_units, 1))
        self.network.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x

def calc_loss(x, x_aug, temperature=0.2, sym=True):
    batch_size = x.shape[0]
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)

    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]

    if sym:

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        #    print(pos_sim,sim_matrix.sum(dim=0))
        loss_0 = - torch.log(loss_0).mean()
        loss_1 = - torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
    else:
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

    return loss
def cross_correlation(Z_v1, Z_v2):

    return torch.mm(F.normalize(Z_v1, dim=1), F.normalize(Z_v2, dim=1).t())



class InstanceLoss(nn.Module):

    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()

        self.batch_size = batch_size

        self.temperature = temperature

        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)

        self.criterion = nn.CrossEntropyLoss(reduction="sum")  # 每一个batch的loss相加

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size

        mask = torch.ones((N, N))

        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0

            mask[batch_size + i, i] = 0

        mask = mask.bool()

        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature

        # torch.matmul(features, features.T)# 计算两两样本间点乘相似度

        sim_i_j = torch.diag(sim, self.batch_size)

        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)  # [2*batch_size, 1]

        negative_samples = sim[self.mask].reshape(N, -1)  # [2*batch_size, 2*batch_size-2]

        labels = torch.zeros(N).to(positive_samples.device).long()

        logits = torch.cat((positive_samples, negative_samples), dim=1)  # [2*batch_size, 2*batch_size-1]

        loss = self.criterion(logits, labels)

        loss /= N  # batch 求均值

        # https://blog.csdn.net/weixin_44966641/article/details/120382198

        return loss


SMALL_NUM = np.log(1e-45)


class DCL(object):
    """

    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf

    weight: the weighting function of the positive sample loss

    temperature: temperature to control the sharpness of the distribution

    """

    def __init__(self, temperature=0.5, weight_fn=None):
        super(DCL, self).__init__()

        self.temperature = temperature

        self.weight_fn = weight_fn

    def __call__(self, z1, z2):
        """

        Calculate one way DCL loss

        :param z1: first embedding vector

        :param z2: second embedding vector

        :return: one-way loss

        """

        cross_view_distance = torch.mm(z1, z2.t())

        positive_loss = -torch.diag(cross_view_distance) / self.temperature

        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)

        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / self.temperature

        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)

        negative_loss = torch.logsumexp(neg_similarity + neg_mask * SMALL_NUM, dim=1, keepdim=False)

        return (positive_loss + negative_loss).mean()


class ClusterLoss(nn.Module):

    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()

        self.class_num = class_num

        self.temperature = temperature

        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num

        mask = torch.ones((N, N))

        mask = mask.fill_diagonal_(0)

        for i in range(class_num):
            mask[i, class_num + i] = 0

            mask[class_num + i, i] = 0

        mask = mask.bool()

        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)

        p_i /= p_i.sum()

        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()

        p_j = c_j.sum(0).view(-1)

        p_j /= p_j.sum()

        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()

        ne_loss = ne_i + ne_j

        c_i = c_i.t()

        c_j = c_j.t()

        N = 2 * self.class_num

        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.class_num)

        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()

        logits = torch.cat((positive_clusters, negative_clusters), dim=1)

        loss = self.criterion(logits, labels)

        loss /= N

        return loss + ne_loss




#criterion_cluster = contrastive_loss.ClusterLoss(cluster_number, args.cluster_temperature, loss_device).to(loss_device)






def cluster_loss_function(c_i, c_j, class_num, temperature, device):
    N = 2 * class_num
    mask = torch.ones((N, N), dtype=torch.bool, device=device)
    mask.fill_diagonal_(0)
    for i in range(class_num):
        mask[i, class_num + i] = 0
        mask[class_num + i, i] = 0

    p_i = c_i.sum(0).view(-1) / c_i.sum()
    ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
    p_j = c_j.sum(0).view(-1) / c_j.sum()
    ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
    ne_loss = ne_i + ne_j

    c = torch.cat((c_i.t(), c_j.t()), dim=0)
    sim = F.cosine_similarity(c.unsqueeze(1), c.unsqueeze(0), dim=2) / temperature
    sim_i_j = torch.diag(sim, class_num)
    sim_j_i = torch.diag(sim, -class_num)

    positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    negative_clusters = sim[mask].reshape(N, -1)

    labels = torch.zeros(N, device=device).long()
    logits = torch.cat((positive_clusters, negative_clusters), dim=1)
    loss = F.cross_entropy(logits, labels, reduction="sum") / N

    return loss + ne_loss

def correlation_reduction_loss(S):

    return torch.diagonal(S).add(-1).pow(2).mean() + off_diagonal(S).pow(2).mean()


def drr_loss1(z1,z2):
    S_N = cross_correlation(z1, z2)
    L_N = correlation_reduction_loss(S_N)
    loss_drr = opt.args.lambda1 * L_N

    return loss_drr
def calc_loss(x, x_aug, temperature=0.2, sym=True):
    batch_size = x.shape[0]
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]

    if sym:

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        #    print(pos_sim,sim_matrix.sum(dim=0))
        loss_0 = - torch.log(loss_0).mean()
        loss_1 = - torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
    else:
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

    return loss
def CL_Loss(u1, u2, temperature=0.5):
    """
    cluster-level contrastive learning
    """
    # u1 = torch.mm(z1.T, h1)
    # u1 = F.normalize(u1, p=2, dim=1)
    #
    # u2 = torch.mm(z2.T, h1)
    # u2 = F.normalize(u2, p=2, dim=1)

    sim_cross1 = torch.exp(torch.mm(u1, u2.T) / temperature)
    sim_cross2 = torch.exp(torch.mm(u2, u1.T) / temperature)

    sim_same1 = torch.exp(torch.mm(u1, u1.T) / temperature)
    sim_same2 = torch.exp(torch.mm(u2, u2.T) / temperature)

    loss = 0
    loss += -torch.log(torch.diagonal(sim_cross1) / torch.sum(sim_cross1, dim=1)
                       ) + (torch.sum(sim_same1, dim=1) - torch.diagonal(sim_same1))
    loss += -torch.log(torch.diagonal(sim_cross2) / torch.sum(sim_cross2, dim=1)
                       ) + (torch.sum(sim_same2, dim=1) - torch.diagonal(sim_same2))

    return loss.mean()

def drr_loss2(z1,z2):
    S_F = cross_correlation(z1, z2)
    L_F = correlation_reduction_loss(S_F)

    loss_drr = + opt.args.lambda2 * L_F

    return loss_drr
def pretrain_ae(model, x):
    print("Pretraining AE...")
    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    for epoch in tqdm.tqdm(range(opt.args.rec_epoch)):
        z = model.encoder(x)
        x_hat = model.decoder(z)
        loss = F.mse_loss(x_hat, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def pretrain_gae(model, x, adj):
    print("Pretraining GAE...")
    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    for epoch in tqdm.tqdm(range(opt.args.rec_epoch)):
        z, a = model.encoder(x, adj)
        z_hat, z_adj_hat = model.decoder(z, adj)
        a_hat = a + z_adj_hat
        loss_w = F.mse_loss(z_hat, torch.spmm(adj, x))
        loss_a = F.mse_loss(a_hat, adj.to_dense())
        loss = loss_w + opt.args.alpha_value * loss_a

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), './model_pretrained/{}_pretrain1.pkl'.format(opt.args.name))

def GL_Loss(A, h1, par=2e+3):
    """
    Loss of graph learning
    """
    A= A.to_dense
    # degree = A.sum(dim=1)
    # L = torch.diag(degree) - A
    D = np.diag(A.sum(0))
    L = D - A
    loss = torch.trace(h1.T.matmul(L).matmul(h1)) + (par/2) * torch.norm(A)**2
    return loss.mean()
def pre_train(model, X1, A1, X2, A2):
    print("Pretraining fusion model...")
    MSE = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    for epoch in tqdm.tqdm(range(opt.args.fus_epoch)):

        # input & output
        hat1, z_fused_hat1, hat2, z_fused_hat2, _, _, _, x_rec, x_rec_, _, _, _, x_rec_f,x_rec_f_,u1,u2,u,u1_,u2_,u_  = model(X1, A1, X2, A2, pretrain=True)
        l1 = MSE(X1, x_rec_f) + MSE(X2, x_rec_f_)
        # L_DRR = drr_loss(cons)
        L_REC1 = reconstruction_loss(X1, A1, hat1)
        L_REC2 = reconstruction_loss(X2, A2, hat2)
        L_REC3 = reconstruction_loss(X1, A1, z_fused_hat1)
        L_REC4 = reconstruction_loss(X2, A2, z_fused_hat2)
        loss = L_REC1 + L_REC2 + L_REC3 + L_REC4+l1

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), './model_pretrained/{}_pretrain1.pkl'.format(opt.args.name))
    

def train(model, X1, A1, X2, A2, y):

    if not opt.args.pretrain:
        #loading pretrained model
        model.load_state_dict(torch.load('./model_pretrained/{}_pretrain1.pkl'.format(opt.args.name), map_location='cpu'))

    print("Training...")
    # best_epoch=0
    # epoch=0
    optimizer = Adam(model.parameters(), lr=(opt.args.lr))
    MSE = nn.MSELoss()
    #scaler = GradScaler()
    pbar = tqdm.tqdm(range(opt.args.epoch), ncols=200)
    instanceLoss = InstanceLoss(opt.args.n_clusters, 0.5, device)
    supConLoss = SupConLoss()
    kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=1,
                  init='k-means++')


    for epoch in pbar:

        # input & output
        hat1, z_fused_hat1, hat2, z_fused_hat2, z1, z2, z_fused, x_rec, x_rec_, h2, h2_, h_fuzed, x_rec_f, x_rec_f_,u1,u2,u,u1_,u2_,u_  = model(
            X1, A1, X2, A2)


        with torch.no_grad():
            km_h = kmeans.fit_predict(h_fuzed.data.cpu().numpy())
            km_z = kmeans.fit_predict(z_fused.data.cpu().numpy())

            km_h = torch.tensor(km_h).cuda()
            km_h = F.one_hot(km_h.long()).float()
            maskh = (km_h @ km_h.T).long().float()  # 自相似矩阵

            km_z = torch.tensor(km_z).cuda()
            km_z = F.one_hot(km_z.long()).float()
            maskz = (km_z @ km_z.T).long().float()


        batch_size, feature_number = z_fused.size()

        z1_r = z1.reshape(batch_size, 1, feature_number)
        z2_r = z2.reshape(batch_size, 1, feature_number)
        h2_r = h2.reshape(batch_size, 1, feature_number)
        h2__r = h2_.reshape(batch_size, 1, feature_number)

        z_r = torch.cat([z1_r, z2_r], dim=1)
        h_r = torch.cat([h2_r, h2__r], dim=1)
        l3 =  supConLoss(features=h_r, mask=maskz)
        #l3 = supConLoss(features=z_r, mask=maskh)  + supConLoss(features=h_r, mask=maskz)
        l_mse = MSE(X1,x_rec_f)+MSE(X2,x_rec_f_)
        L_REC1 = reconstruction_loss(X1, A1, hat1)
        L_REC2 = reconstruction_loss(X2, A2, hat2)
        L_REC3 = reconstruction_loss(X1, A1, z_fused_hat1)
        L_REC4 = reconstruction_loss(X2, A2, z_fused_hat2)
        l2 = instanceLoss(u1, u) + instanceLoss(u2, u)
        l2_ = instanceLoss(u1_, u_) + instanceLoss(u2_, u_)
        l_gaer = L_REC1 + L_REC2 + L_REC3 + L_REC4
        loss = l_gaer + l_mse + l3  +l2  + l2_

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=5)
        y_pred = kmeans.fit_predict(h_fuzed.data.cpu().numpy())

        ari, nmi, ami, acc = eva(y, y_pred, show_details=False)

        pbar.set_postfix({'loss':'{0:1.4f}'.format(loss), 'ARI':'{0:1.4f}'.format(ari),'NMI':'{0:1.4f}'.format(nmi),
                          'AMI':'{0:1.4f}'.format(ami),'ACC':'{0:1.4f}'.format(acc)})

        if acc > opt.args.acc:
            opt.args.acc = acc
            opt.args.nmi = nmi
            opt.args.ari = ari
            opt.args.ami = ami
            best_epoch = epoch

    pbar.close()

    print("Best_epoch: {},".format(best_epoch),"ARI: {:.4f},".format(opt.args.ari), "NMI: {:.4f},".format(opt.args.nmi),
            "AMI: {:.4f}".format(opt.args.ami), "ACC: {:.4f}".format(opt.args.acc))

    print("Final_epoch: {},".format(epoch),"ARI: {:.4f},".format(ari), "NMI: {:.4f},".format(nmi),
            "AMI: {:.4f}".format(ami), "ACC: {:.4f}".format(acc))

    np.save('phlabel.npy'.format(opt.args.name, opt.args.seed), y_pred)
    np.save('H.npy'.format(opt.args.name, opt.args.seed), (h_fuzed).cpu().detach().numpy())
    #np.save('Z.npy'.format(opt.args.name, opt.args.seed), (z_fused).cpu().detach().numpy())


if __name__ == '__main__':
    # setup
    print("setting:")

    setup_seed(opt.args.seed)

    opt.args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("------------------------------")
    print("dataset       : {}".format(opt.args.name))
    print("device        : {}".format(opt.args.device))
    print("random seed   : {}".format(opt.args.seed))
    print("lambda1 value : {}".format(opt.args.lambda1))
    print("lambda2 value : {}".format(opt.args.lambda2))
    print("lambda3 value : {}".format(opt.args.lambda3))
    print("alpha value   : {:.0e}".format(opt.args.alpha_value))
    print("k value       : {}".format(opt.args.k))
    print("learning rate : {:.0e}".format(opt.args.lr))
    print("------------------------------")

    # load data
    Xr, y, Ar = load_data(opt.args.name, 'RNA', opt.args.method, opt.args.k, show_details=False)

    Xa, y, Aa = load_data(opt.args.name, 'ATAC', opt.args.method, opt.args.k, show_details=False)
    opt.args.n_clusters = int(max(y) - min(y) + 1)

    Xr = numpy_to_torch(Xr).to(opt.args.device)
    Ar = numpy_to_torch(Ar, sparse=True).to(opt.args.device)

    Xa = numpy_to_torch(Xa).to(opt.args.device)
    Aa = numpy_to_torch(Aa, sparse=True).to(opt.args.device)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if opt.args.pretrain:
        opt.args.dropout = 0.4
    gae1 = IGAE(
        gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
        gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
        n_input=opt.args.n_d1, n_z=opt.args.n_z, dropout=opt.args.dropout).to(opt.args.device)

    gae2 = IGAE(
        gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
        gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
        n_input=opt.args.n_d2, n_z=opt.args.n_z, dropout=opt.args.dropout).to(opt.args.device)

    if opt.args.pretrain:
        t0 = time()


        pretrain_gae(gae1, Xr, Ar)
        pretrain_gae(gae2, Xa, Aa)


        model = McMVC(gae1, gae2, n_node=Xr.shape[0]).to(opt.args.device)



        pre_train(model, Xr, Ar, Xa, Aa)
        t1 = time()
        print("Time_cost: {}".format(t1-t0))
    else:
        t0 = time()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = McMVC( gae1, gae2,n_node=Xr.shape[0]).to(opt.args.device)

        train(model, Xr, Ar, Xa, Aa, y)
        t1 = time()
        print("Time_cost: {}".format(t1-t0))
