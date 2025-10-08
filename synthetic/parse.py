import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Confirmation-Aware Social Dynamic Model.")
    parser.add_argument('--seed', type=int, default=2020,
                        help="random seed")  # 2020,42,123
    parser.add_argument('--dataset', type=str, default="synthetic",
                        help="dataset")  

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

                        
    parser.add_argument('--user_nums', type=int, default=1000,
                        help="user nums")
    parser.add_argument('--social_nums', type=int, default=10000,
                        help="social link nums")
    parser.add_argument('--item_nums', type=int, default=10000,
                        help="item nums")
    parser.add_argument('--cate_nums', type=int, default=10,
                        help="category nums")
    return parser.parse_args()