import opt
from encoder import *
import numpy as np
from sklearn.cluster import KMeans
class AttentionFusionModule(nn.Module):
    def __init__(self, feature_dim, feature_num=2):
        super(AttentionFusionModule, self).__init__()
        self.attention_network = nn.Sequential(
            nn.Linear(feature_num * feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, feature_num),
            nn.Softmax(dim=-1)
        )
        self.feature_num = feature_num

    def forward(self, z: list):
        combined = torch.cat(z, dim=1)
        attention_weights = self.attention_network(combined)
        weighted_z = []
        for i in range(0, self.feature_num):
            weighted_z_i = attention_weights[:, i:i + 1] * z[i]
            weighted_z.append(weighted_z_i)

        fused_z = sum(weighted_z)

        return fused_z
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def reconstruction(z, a_igae, adj, gae_decoder):

    z_hat, z_adj_hat = gae_decoder(z, adj)
    a_hat = a_igae + z_adj_hat
    hat = [ z_hat, a_hat]
    return hat

class McMVC(nn.Module):
    def __init__(self,gae1, gae2,n_node=None):
        super(McMVC, self).__init__()


        self.gae1 = gae1
        self.gae2 = gae2

        self.ec1 = nn.Linear(opt.args.n_d1, opt.args.emb_d, bias=True)
        self.ec2 = nn.Linear(opt.args.emb_d, opt.args.n_z, bias=True)

        self.ec1_ = nn.Linear(opt.args.n_d2, opt.args.emb_d, bias=True)
        self.ec2_ = nn.Linear(opt.args.emb_d, opt.args.n_z, bias=True)

        self.dc1 = nn.Linear(opt.args.n_z, opt.args.emb_d, bias=True)
        self.dc2 = nn.Linear(opt.args.emb_d, opt.args.n_d1, bias=True)

        self.dc1_ = nn.Linear(opt.args.n_z, opt.args.emb_d, bias=True)
        self.dc2_ = nn.Linear(opt.args.emb_d, opt.args.n_d2, bias=True)

        self.attention_fusion = AttentionFusionModule(opt.args.n_z, feature_num=2)


        self.label_contrastive_module = nn.Sequential(
            nn.Linear(opt.args.n_z, opt.args.n_clusters),
            nn.Softmax(dim=1)
        )
        self.label_contrastive_module2 = nn.Sequential(
            nn.Linear(n_node, opt.args.n_clusters),
            nn.Softmax(dim=1)
        )

    def update_cluster_centers(self, new_centers1, new_centers2,new_centers):
        self.cluster_centers1 = nn.Parameter(new_centers1.clone())
        self.cluster_centers2 = nn.Parameter(new_centers2.clone())
        self.cluster_centers = nn.Parameter(new_centers.clone())
    def emb_fusion(self, adj, z_ae, z_igae):
        z_i = self.a * z_ae + (1 - self.a) * z_igae
        z_l = torch.spmm(adj, z_i)
        s = torch.mm(z_l, z_l.t())
        s = F.softmax(s, dim=1)
        z_g = torch.mm(s, z_l)
        z_tilde = self.alpha * z_g + z_l

        return z_tilde

    def forward(self, x1,adj1, x2,adj2, pretrain=False):
        # node embedding encoded by AE
        h2 = F.relu(self.ec1(x1))
        h2 = F.relu(self.ec2(h2))
        h2 = F.normalize(h2, p=2, dim=1)


        h2_ = F.relu(self.ec1_(x2))
        h2_ = F.relu(self.ec2_(h2_))
        h2_ = F.normalize(h2_, p=2, dim=1)
        h_fuzed= self.attention_fusion([ h2, h2_])

        z1, a_igae1 = self.gae1.encoder(x1, adj1)
        z2, a_igae2 = self.gae2.encoder(x2, adj2)
        z_fused = self.attention_fusion([ z1, z2])



        x_rec = F.relu(self.dc1(h2))
        x_rec = F.relu(self.dc2(x_rec))
        x_rec = F.normalize(x_rec, p=2, dim=1)

        x_rec_ = F.relu(self.dc1_(h2_))
        x_rec_ = F.relu(self.dc2_(x_rec_))
        x_rec_ = F.normalize(x_rec_, p=2, dim=1)

        x_rec_f = F.relu(self.dc1(h_fuzed))
        x_rec_f = F.relu(self.dc2(x_rec_f))
        x_rec_f = F.normalize(x_rec_f, p=2, dim=1)

        x_rec_f_ = F.relu(self.dc1_(h_fuzed))
        x_rec_f_ = F.relu(self.dc2_(x_rec_f_))
        x_rec_f_ = F.normalize(x_rec_f_, p=2, dim=1)


        u1 = self.label_contrastive_module2(z1.t())#d*c
        u2 = self.label_contrastive_module2(z2.t())

        u = self.label_contrastive_module2(z_fused.t())

        u1_ = self.label_contrastive_module2(h2.t())  # d*c
        u2_ = self.label_contrastive_module2(h2_.t())

        u_ = self.label_contrastive_module2(h_fuzed.t())

        u1 = u1.t()
        u2 = u2.t()
        u = u.t()

        u1_ = u1_.t()
        u2_ = u2_.t()
        u_ = u_.t()



        hat1 = reconstruction(z1, a_igae1, adj1, self.gae1.decoder)
        z_fused_hat1 = reconstruction(z_fused, a_igae1, adj1, self.gae1.decoder)
        hat2 = reconstruction(z2, a_igae2, adj2, self.gae2.decoder)
        z_fused_hat2 = reconstruction(z_fused, a_igae2, adj2, self.gae2.decoder)



        return hat1, z_fused_hat1, hat2, z_fused_hat2, z1, z2,  z_fused, x_rec, x_rec_, h2, h2_, h_fuzed, x_rec_f, x_rec_f_,u1,u2,u,u1_,u2_,u_ # ,s#,Q1, Q2,q
