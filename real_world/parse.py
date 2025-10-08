import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Confirmation-Aware Social Dynamic Model.")
    parser.add_argument('--seed', type=int, default=2020,
                        help="random seed")  # 2020,42,123
    parser.add_argument('--dataset', type=str, default="ciao",
                        help="dataset")  # ciao,epinions

    parser.add_argument('--alpha', type=float, default=5.,
                        help="alpha")
    parser.add_argument('--beta', type=float, default=5.,
                        help="beta")
    parser.add_argument('--gamma', type=float, default=0.5,
                        help="gamma")
    parser.add_argument('--epsilon', type=float, default=0.,
                        help="epsilon")

    parser.add_argument('--num_rec', type=int, default=20,
                        help="recommendation list length")
    parser.add_argument('--eta', type=float, default=0.1,
                        help="update rate")
    parser.add_argument('--epoch', type=int, default=1000,
                        help="iteration epoch")
    parser.add_argument('--threshold', type=float, default=0.7,
                        help="Interest threshold")

    parser.add_argument('--name', type=str, default="alpha_5.0",
                        help="name for save")
    parser.add_argument('--gpu', type=int, default=0,
                        help="gpu")

    parser.add_argument('--tsne', type=int, default=0,
                        help="use tsne or not")


    parser.add_argument('--s1', type=int, default=0,
                        help="user adaptive alpha")              
    parser.add_argument('--sigma', type=float, default=10,
                        help="sigma")

    parser.add_argument('--s2', type=int, default=0,
                        help="feedback update adjustment")
    parser.add_argument('--rho', type=float, default=0.,
                        help="rho")   

    parser.add_argument('--s3', type=int, default=0,
                        help="diversity post-processing")
    parser.add_argument('--coarse_ranking_num', type=int, default=1000,
                        help="coarse ranking numbers")
    parser.add_argument('--theta', type=float, default=0.,
                        help="theta")

    parser.add_argument('--s4', type=int, default=0,
                        help="social aggregation reweighting")
    parser.add_argument('--omega', type=float, default=0.,
                        help="omega")
                     
    return parser.parse_args()