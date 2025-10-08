import random
import numpy as np
from parse import parse_args
import torch
import random
from evaluate import *
import os
from tsne import *
from collections import defaultdict



def print_args(args):
    args_str = "Experimental Settings:\n"
    for arg, value in vars(args).items():
        args_str += f"{arg}: {value}\n"
    print(args_str)
args = parse_args()
print_args(args)
dataset = args.dataset
  
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

final_results = {}
seeds = [1, 10, 100, 1000, 10000, 5, 50, 500, 5000, 50000]
for seed in seeds:

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    device = torch.device("cuda")

    

    # generate social network
    def generate_network(num_users=1000, num_relations=10000, output_file=dataset + "/network.txt"):
        relations = set()
        
        while len(relations) < num_relations:
            u1 = random.randint(0, num_users - 1)
            u2 = random.randint(0, num_users - 1)
            
            if u1 == u2:
                continue
            
            relations.add((u1, u2))
        
        with open(output_file, "w") as f:
            for u1, u2 in relations:
                f.write(f"{u1} {u2}\n")

    generate_network(num_users=1000, num_relations=args.social_nums)
    print(f"Generated social network, saved to {dataset}/network.txt")
    def process_social_data(file_path):
        social_dict = {}
        u1_list = []
        u2_list = []
        
        with open(file_path, 'r') as file:
            for line in file:
                row, col = map(int, line.strip().split())
                
                if row not in social_dict:
                    social_dict[row] = [col]
                else:
                    social_dict[row].append(col)
                
                u1_list.append(row)
                u2_list.append(col)
        
        return social_dict, u1_list, u2_list

    social_path = dataset + "/network.txt"
    social_dict, u1_list, u2_list = process_social_data(social_path)



    num_users = args.user_nums       
    num_items = args.item_nums  
    vec_length = args.cate_nums    
    output_file = dataset + "/cate_vector_norm.npy"

    # generate item representations
    item_dict = defaultdict(list)

    for item_id in range(num_items):
        vec = [0.0] * vec_length
        hot_pos = random.randint(0, vec_length - 1)
        vec[hot_pos] = 1.0
        item_dict[item_id] = np.array(vec)


    np.save(output_file, dict(item_dict))  

    print(f"Generated category representations for {num_items} items, saved to {output_file}")
    print(f"Example (item 0): {item_dict[0]}")
    print(f"Example (item 9999): {item_dict[9999]}")




    cate_vec = np.load(dataset + "/cate_vector_norm.npy", allow_pickle=True).item()
    cate_num = len(cate_vec[0])


    item_matrix = torch.tensor(
        [cate_vec[iid] for iid in sorted(cate_vec.keys())],
        dtype=torch.float32,
        device=device
    )  # [num_items, cate_num]
    cate_vec = {
        iid: torch.tensor(vec, dtype=torch.float32, device=device)
        for iid, vec in cate_vec.items()
    }


    cate_0_1 = {}  
    for iid in cate_vec.keys():
        cate_0_1[iid] = torch.tensor([1 if value != 0 else 0 for value in cate_vec[iid]],  device=device)






    # generate user representations
    def generate_user_embeddings(num_users, embedding_dim, device="cpu"):
        user_embeddings = {}
        
        for uid in range(num_users):
            vec = torch.randn(embedding_dim, dtype=torch.float32)
            
            vec_normalized = vec / torch.norm(vec, p=2)
            
            user_embeddings[uid] = vec_normalized.to(device)
        
        return user_embeddings

    user_init_vec_dict = generate_user_embeddings(
        num_users=num_users,
        embedding_dim=vec_length,
        device=device
    )

    for uid, vec in user_init_vec_dict.items():
        l2_norm = torch.norm(vec, p=2).item()
        assert abs(l2_norm - 1.0) < 1e-6

    print(f"Generated interest representations for {len(user_init_vec_dict)} users, each with L2 norm=1")  
    print(f"Example (user 0): {user_init_vec_dict[0]}")  
    print(f"L2 norm verification: {torch.norm(user_init_vec_dict[0], p=2)}")


    def get_rec_dict(user_init_vec_dict, item_matrix, social_dict, args, k=4):

        user_matrix = torch.stack(list(user_init_vec_dict.values()))  # [num_users, cate_num]

        social_neighbors = [
            [user_init_vec_dict[u] for u in social_dict.get(uid, [])]
            for uid in user_init_vec_dict.keys()
        ]
        social_vecs = torch.stack([
            torch.mean(torch.stack(neighbors), dim=0) if neighbors else torch.zeros_like(user_matrix[0])
            for neighbors in social_neighbors
        ])  # [num_users, cate_num]
        user_matrix_fused = args.gamma * user_matrix + (1 - args.gamma) * social_vecs



        similarity = torch.softmax(user_matrix_fused @ item_matrix.T * args.alpha, dim=1)
        rec_indices = torch.multinomial(similarity, args.num_rec, replacement=False)  # [num_users, num_rec]

        rec_dict = {
            uid: rec_indices[i].cpu().numpy()
            for i, uid in enumerate(user_init_vec_dict.keys())
        }

        return rec_dict

    def update_user_vectors(user_init_vec_dict, rec_dict, cate_vec, args):
        user_matrix = torch.stack(list(user_init_vec_dict.values()))  # [num_users, cate_num]

        rec_items = torch.stack([
            torch.stack([torch.tensor(cate_vec[iid], device=device) for iid in rec_dict[uid]])
            for uid in user_init_vec_dict.keys()
        ])

        old_vecs = user_matrix.unsqueeze(1)  # [num_users, 1, cate_num]
        dots = torch.sum(rec_items * old_vecs, dim=2)  # [num_users, num_rec]

        prob_positive = ((1 + dots)**args.beta) / \
                    ((1 + dots)**args.beta + (1 - dots)**args.beta) + \
                    args.epsilon/2
        
        decisions = torch.rand_like(prob_positive) <= prob_positive  # [num_users, num_rec]
        
        positive_effect = torch.sum(rec_items * decisions.unsqueeze(2), dim=1)  # [num_users, cate_num]
        negative_effect = torch.sum(rec_items * (~decisions).unsqueeze(2), dim=1)  # [num_users, cate_num]
        
        new_vecs = user_matrix + args.eta * (positive_effect - negative_effect) / args.num_rec
        new_vecs = new_vecs / torch.norm(new_vecs, p=2, dim=1, keepdim=True)  # L2 normalize

        for i, uid in enumerate(user_init_vec_dict.keys()):
            user_init_vec_dict[uid] = new_vecs[i]
        return user_init_vec_dict


    epochs_rec_cate_entropy = []
    epochs_rec_accuracy = []
    epochs_nei_distance = []
    epochs_pair_dis_variance = []
    epochs_topk_similarity = []


    for epoch in range(args.epoch):
        print(f"Iteration Epochs:{epoch}")

        if epoch % 10 == 0 and args.tsne == 1:
            draw_tsne(user_init_vec_dict, args, epoch)


        rec_dict = get_rec_dict(user_init_vec_dict, item_matrix, social_dict, args)


        rec_cate_entropy = calc_rec_cate_entropy(rec_dict, cate_0_1)
        rec_accuracy = calc_rec_accuracy(user_init_vec_dict, args.threshold, rec_dict, cate_vec)
        nei_distance = calc_nei_distance(user_init_vec_dict, u1_list, u2_list)
        pair_dis_variance = calc_pair_dis_variance(user_init_vec_dict)
        topk_similarity = calc_topk_similarity(user_init_vec_dict, args)
        print(rec_cate_entropy)
        print(rec_accuracy)
        print(nei_distance)
        print(pair_dis_variance)
        print(topk_similarity)
        print("\n")
        epochs_rec_cate_entropy.append(rec_cate_entropy)
        epochs_rec_accuracy.append(rec_accuracy)
        epochs_nei_distance.append(nei_distance)
        epochs_pair_dis_variance.append(pair_dis_variance)
        epochs_topk_similarity.append(topk_similarity)

        user_init_vec_dict = update_user_vectors(user_init_vec_dict, rec_dict, cate_vec, args)


    eva_dict = {}
    eva_dict["rec_cate_entropy"] = epochs_rec_cate_entropy
    eva_dict["rec_accuracy"] = epochs_rec_accuracy
    eva_dict["nei_distance"] = epochs_nei_distance
    eva_dict["pair_dis_variance"] = epochs_pair_dis_variance
    eva_dict["topk_similarity"] = epochs_topk_similarity

    final_results[seed] = eva_dict
np.save(dataset + f"/eva_dict_seeds_{args.name}.npy", final_results)