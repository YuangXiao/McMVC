import argparse

parser = argparse.ArgumentParser(description='McMVC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# setting
parser.add_argument('--name', type=str, default="GSE")#PBMC-3k
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--rec_epoch', type=int, default=30)
parser.add_argument('--fus_epoch', type=int, default=100)
parser.add_argument('--epoch', type=int, default=250)
parser.add_argument('--pretrain', type=bool, default=False)

# parameters
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--alpha_value', type=float, default=0.1)
parser.add_argument('--lambda1', type=float, default=10)
parser.add_argument('--lambda2', type=float, default=1)
parser.add_argument('--lambda3', type=float, default=10)
parser.add_argument('--method', type=str, default='euc')
parser.add_argument('--first_view', type=str, default='ATAC')
parser.add_argument('--lr', type=float, default=1e-3)

# dimension of input and latent representations
parser.add_argument('--n_d1', type=int, default=1000)#300 10 1000
parser.add_argument('--n_d2', type=int, default=25)#1024 128 49
parser.add_argument('--n_z', type=int, default=25)
parser.add_argument('--in_d', type=int, default=8200)
parser.add_argument('--out_d', type=int, default=20)
parser.add_argument('--emb_d', type=int, default=64)

parser.add_argument('--temperature', type=float, default=0.2)


# AE structure parameter
parser.add_argument('--ae_n_enc_1', type=int, default=256)
parser.add_argument('--ae_n_enc_2', type=int, default=128)
parser.add_argument('--ae_n_dec_1', type=int, default=128)
parser.add_argument('--ae_n_dec_2', type=int, default=256)

# IGAE structure parameter
parser.add_argument('--gae_n_enc_1', type=int, default=256)
parser.add_argument('--gae_n_enc_2', type=int, default=128)
parser.add_argument('--gae_n_dec_1', type=int, default=128)
parser.add_argument('--gae_n_dec_2', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0)

# clustering performance: acc, nmi, ari, f1

parser.add_argument('--acc', type=float, default=0)
parser.add_argument('--nmi', type=float, default=0)
parser.add_argument('--ari', type=float, default=0)
parser.add_argument('--ami', type=float, default=0)

args = parser.parse_args()
