import numpy as np
from parse import parse_args
import torch
import random
import os
from evaluate import *
from tsne import *

def print_args(args):
    args_str = "Experimental Settings:\n"
    for arg, value in vars(args).items():
        args_str += f"{arg}: {value}\n"
    print(args_str)
args = parse_args()
print_args(args)
dataset = args.dataset

os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"  
device = torch.device("cuda")

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False






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

social_path = "./data/" + dataset + "/network.txt"
social_dict, u1_list, u2_list = process_social_data(social_path)


cate_vec = np.load("./data/" + dataset + "/cate_vector_norm.npy", allow_pickle=True).item()
cate_num = len(cate_vec[0])


data_dict = np.load("./data/" + dataset + "/data_dict.npy", allow_pickle=True).item()
user_init_vec_dict = {}
for uid in data_dict.keys():
    pos_list = data_dict[uid][0]
    neg_list = data_dict[uid][1]
    u_init_vec = np.array([0.] * cate_num)
    for iid in pos_list:
        i_vec = cate_vec[iid]
        u_init_vec += np.array(i_vec)
    for iid in neg_list:
        i_vec = cate_vec[iid]
        u_init_vec -= np.array(i_vec)
    user_init_vec_dict[uid] = u_init_vec / np.linalg.norm(u_init_vec, ord=2)
    if not np.abs(np.linalg.norm(user_init_vec_dict[uid], ord=2) - 1) < 0.0000001:
        user_init_vec_dict[uid] = np.array([1.] * cate_num) / np.linalg.norm(np.array([1.] * cate_num), ord=2) 
        assert np.abs(np.linalg.norm(user_init_vec_dict[uid], ord=2) - 1) < 0.0000001
user_init_vec_dict = {
    uid: torch.tensor(vec, dtype=torch.float32, device=device)
    for uid, vec in user_init_vec_dict.items()
}



item_matrix = torch.tensor(
    [cate_vec[iid] for iid in sorted(cate_vec.keys())],
    dtype=torch.float32,
    device=device
)  

cate_vec = {
    iid: torch.tensor(vec, dtype=torch.float32, device=device)
    for iid, vec in cate_vec.items()
}


cate_0_1 = {}  
for iid in cate_vec.keys():
    cate_0_1[iid] = torch.tensor([1 if value != 0 else 0 for value in cate_vec[iid]],  device=device)


def get_user_var(user_vec_dict):
    user_mat = torch.stack(list(user_vec_dict.values()))
    
    mean_vec = torch.mean(user_mat, dim=1, keepdim=True)
    diff = (user_mat - mean_vec) ** 2

    user_var = torch.sum(diff, dim=1)
    
    return user_var

def get_user_var_inverse_norm(user_vec_dict):
    user_mat = torch.stack(list(user_vec_dict.values()))
    
    mean_vec = torch.mean(user_mat, dim=1, keepdim=True)
    diff = (user_mat - mean_vec) ** 2

    user_var = torch.sum(diff, dim=1)
    
    user_var_inverse = (1 / user_var) ** args.sigma
    user_var_inverse = torch.where(torch.isinf(user_var_inverse), torch.tensor(1.0).to(device), user_var_inverse)
    user_var_inverse_norm = user_var_inverse * user_var_inverse.shape[0] / torch.sum(user_var_inverse)

    return user_var_inverse_norm



def get_rec_dict(user_init_vec_dict, item_matrix, social_dict, args, k=4):
    if args.s1 == 1:
        user_var_inverse_norm = get_user_var_inverse_norm(user_init_vec_dict).unsqueeze(1)

        if args.dataset == "ciao":
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



            similarity = torch.softmax(user_matrix_fused @ item_matrix.T * user_var_inverse_norm * args.alpha, dim=1)
            rec_indices = torch.multinomial(similarity, args.num_rec, replacement=False)  # [num_users, num_rec]

            rec_dict = {
                uid: rec_indices[i].cpu().numpy()
                for i, uid in enumerate(user_init_vec_dict.keys())
            }
        elif args.dataset == "epinions":
            user_matrix = torch.stack(list(user_init_vec_dict.values()))  # [N, D]

            social_neighbors = [
                [user_init_vec_dict[u] for u in social_dict.get(uid, [])]
                for uid in user_init_vec_dict.keys()
            ]
            social_vecs = torch.stack([
                torch.mean(torch.stack(neighbors), dim=0) if neighbors else torch.zeros_like(user_matrix[0])
                for neighbors in social_neighbors
            ])
            user_matrix_fused = args.gamma * user_matrix + (1 - args.gamma) * social_vecs


            rec_dict = {}
            N = len(user_matrix_fused)
            chunk_size = (N + k - 1) // k  
            
            for chunk_idx in range(k):
                start = chunk_idx * chunk_size
                end = min((chunk_idx + 1) * chunk_size, N)
                batch_users = user_matrix_fused[start:end]  # [actual_chunk_size, D]
                batch_var = user_var_inverse_norm[start:end]
                with torch.no_grad():
                    chunk_sim = torch.softmax(batch_users @ item_matrix.T * batch_var * args.alpha, dim=1)
                    chunk_rec = torch.multinomial(chunk_sim, args.num_rec, replacement=False)
                
                user_ids_chunk = list(user_init_vec_dict.keys())[start:end]
                for j, uid in enumerate(user_ids_chunk):
                    rec_dict[uid] = chunk_rec[j].cpu().numpy()
                
                del batch_users, chunk_sim, chunk_rec
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    elif args.s3 == 1:
        if args.dataset == "ciao":
            user_matrix = torch.stack(list(user_init_vec_dict.values()))  # [num_users, cate_num]
            social_neighbors = [
                [user_init_vec_dict[u] for u in social_dict.get(uid, [])]
                for uid in user_init_vec_dict.keys()
            ]
            social_vecs = torch.stack([
                torch.mean(torch.stack(neighbors), dim=0) if neighbors else torch.zeros_like(user_matrix[0])
                for neighbors in social_neighbors
            ])  # [num_users, cate_num]

            user_matrix_fused = (args.gamma) * user_matrix + (1 - args.gamma) * social_vecs

            similarity = torch.softmax(user_matrix_fused @ item_matrix.T  * args.alpha, dim=1)
            rec_indices = torch.multinomial(similarity, args.coarse_ranking_num, replacement=False)  # [num_users, 1000]

            item_embeds = item_matrix[rec_indices]  # [num_users, 1000, 28]
            user_embeds = user_matrix.unsqueeze(1)  # [num_users, 1, 28]

            rec_scores = (user_embeds * item_embeds).sum(dim=-1)  # [num_users, 1000]


            diversity_score = torch.zeros((rec_indices.shape[0], args.coarse_ranking_num), dtype=torch.float).to(device) # [num_users, 1000]
            


            rec_post = torch.zeros((rec_indices.shape[0], args.num_rec), dtype=torch.long).to(device)
            for t in range(args.num_rec):

                fuse_scores = (1 - args.theta) * rec_scores - args.theta * diversity_score
                _, max_indices = fuse_scores.max(dim=1)

                rec_post[:, t] = torch.gather(rec_indices, dim=1, index=max_indices.unsqueeze(1)).squeeze(1)
                rec_scores[torch.arange(rec_indices.shape[0]), max_indices] = -9999999

                list_diverity_score = item_matrix[rec_post[:, :t+1]].sum(dim=1)   # [num_users, 28]
                list_diverity_score_norm = list_diverity_score / torch.norm(list_diverity_score, p=2, dim=1, keepdim=True) # [num_users, 28]
                diversity_score = (item_embeds * list_diverity_score_norm.unsqueeze(1)).sum(dim=-1)

            
            rec_dict = {
                uid: rec_post[i].cpu().numpy()
                for i, uid in enumerate(user_init_vec_dict.keys())
            }

        elif args.dataset == "epinions":
            k = 7
            user_matrix = torch.stack(list(user_init_vec_dict.values()))  # [N, D]
            social_neighbors = [
                [user_init_vec_dict[u] for u in social_dict.get(uid, [])]
                for uid in user_init_vec_dict.keys()
            ]
            social_vecs = torch.stack([
                torch.mean(torch.stack(neighbors), dim=0) if neighbors else torch.zeros_like(user_matrix[0])
                for neighbors in social_neighbors
            ])
            user_matrix_fused = (args.gamma) * user_matrix + (1 - args.gamma) * social_vecs


            rec_dict = {}
            N = len(user_matrix_fused)
            chunk_size = (N + k - 1) // k  
            
            for chunk_idx in range(k):
                start = chunk_idx * chunk_size
                end = min((chunk_idx + 1) * chunk_size, N)
                batch_users = user_matrix_fused[start:end]  # [actual_chunk_size, D]
                batch_users_init = user_matrix[start:end]

                with torch.no_grad():
                    chunk_sim = torch.softmax(batch_users @ item_matrix.T * args.alpha, dim=1)
                    chunk_rec = torch.multinomial(chunk_sim, args.coarse_ranking_num, replacement=False)

                    item_embeds = item_matrix[chunk_rec]  # [num_users, 1000, 28]
                    user_embeds = batch_users_init.unsqueeze(1)  # [num_users, 1, 28]

                    rec_scores = (user_embeds * item_embeds).sum(dim=-1)  # [num_users, 1000]


                    diversity_score = torch.zeros((chunk_rec.shape[0], args.coarse_ranking_num), dtype=torch.float).to(device) # [num_users, 1000]
                
                    rec_post = torch.zeros((chunk_rec.shape[0], args.num_rec), dtype=torch.long).to(device)
                    for t in range(args.num_rec):

                        fuse_scores = (1 - args.theta) * rec_scores - args.theta * diversity_score
                        _, max_indices = fuse_scores.max(dim=1)

                        rec_post[:, t] = torch.gather(chunk_rec, dim=1, index=max_indices.unsqueeze(1)).squeeze(1)
                        rec_scores[torch.arange(chunk_rec.shape[0]), max_indices] = -9999999

                        list_diverity_score = item_matrix[rec_post[:, :t+1]].sum(dim=1)   # [num_users, 28]
                        list_diverity_score_norm = list_diverity_score / torch.norm(list_diverity_score, p=2, dim=1, keepdim=True) # [num_users, 28]
                        diversity_score = (item_embeds * list_diverity_score_norm.unsqueeze(1)).sum(dim=-1)
                user_ids_chunk = list(user_init_vec_dict.keys())[start:end]
                for j, uid in enumerate(user_ids_chunk):
                    rec_dict[uid] = rec_post[j].cpu().numpy()
                
                del batch_users, chunk_sim, chunk_rec
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    elif args.s4 == 1:
        if args.dataset == "ciao":
            user_matrix = torch.stack(list(user_init_vec_dict.values()))  # [num_users, cate_num]
            user_var_dict = get_user_var(user_init_vec_dict)

            social_neighbors = [
                [user_init_vec_dict[u] for u in social_dict.get(uid, [])]
                for uid in user_init_vec_dict.keys()
            ]
            var_score = [
                [user_var_dict[u] for u in social_dict.get(uid, [])]
                for uid in user_init_vec_dict.keys()
            ]
            fused_vecs = []
            for idx, neighbors in enumerate(social_neighbors):
                if neighbors:
                    neighbors = torch.stack(neighbors)  # [num_neighbors, cate_num]

                    scores = - torch.tensor(var_score[idx]).to(device) * args.omega
                    weights = torch.softmax(scores, dim=0)  # [num_neighbors]

                    fused_vec = torch.sum(weights.unsqueeze(1) * neighbors, dim=0)  # [cate_num]
                else:
                    fused_vec = torch.zeros_like(user_matrix[0])

                fused_vecs.append(fused_vec)

            social_vecs = torch.stack(fused_vecs)  # [num_users, cate_num]

            user_matrix_fused = (args.gamma) * user_matrix + (1 - args.gamma) * social_vecs


            similarity = torch.softmax(user_matrix_fused @ item_matrix.T * args.alpha, dim=1)
            rec_indices = torch.multinomial(similarity, args.num_rec, replacement=False)  # [num_users, num_rec]
            
            rec_dict = {
                uid: rec_indices[i].cpu().numpy()
                for i, uid in enumerate(user_init_vec_dict.keys())
            }
        elif args.dataset == "epinions":
            user_matrix = torch.stack(list(user_init_vec_dict.values()))  # [N, D]

            user_var_dict = get_user_var(user_init_vec_dict)

            social_neighbors = [
                [user_init_vec_dict[u] for u in social_dict.get(uid, [])]
                for uid in user_init_vec_dict.keys()
            ]
            var_score = [
                [user_var_dict[u] for u in social_dict.get(uid, [])]
                for uid in user_init_vec_dict.keys()
            ]
            fused_vecs = []
            for idx, neighbors in enumerate(social_neighbors):
                if neighbors:
                    neighbors = torch.stack(neighbors)  # [num_neighbors, cate_num]
                    self_vec = user_matrix[idx]  

                    scores = - torch.tensor(var_score[idx]).to(device) * args.omega
                    weights = torch.softmax(scores, dim=0)  # [num_neighbors]

                    fused_vec = torch.sum(weights.unsqueeze(1) * neighbors, dim=0)  # [cate_num]
                else:
                    fused_vec = torch.zeros_like(user_matrix[0])

                fused_vecs.append(fused_vec)

            social_vecs = torch.stack(fused_vecs)  # [num_users, cate_num]

            user_matrix_fused = (args.gamma) * user_matrix + (1 - args.gamma) * social_vecs


            rec_dict = {}
            N = len(user_matrix_fused)
            chunk_size = (N + k - 1) // k  
            
            for chunk_idx in range(k):
                start = chunk_idx * chunk_size
                end = min((chunk_idx + 1) * chunk_size, N)
                batch_users = user_matrix_fused[start:end]  # [actual_chunk_size, D]

                with torch.no_grad():
                    chunk_sim = torch.softmax(batch_users @ item_matrix.T * args.alpha, dim=1)
                    chunk_rec = torch.multinomial(chunk_sim, args.num_rec, replacement=False)
                
                user_ids_chunk = list(user_init_vec_dict.keys())[start:end]
                for j, uid in enumerate(user_ids_chunk):
                    rec_dict[uid] = chunk_rec[j].cpu().numpy()
                
                del batch_users, chunk_sim, chunk_rec
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    else:
        if args.dataset == "ciao":
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
        elif args.dataset == "epinions":
            user_matrix = torch.stack(list(user_init_vec_dict.values()))  # [N, D]

            
            social_neighbors = [
                [user_init_vec_dict[u] for u in social_dict.get(uid, [])]
                for uid in user_init_vec_dict.keys()
            ]
            social_vecs = torch.stack([
                torch.mean(torch.stack(neighbors), dim=0) if neighbors else torch.zeros_like(user_matrix[0])
                for neighbors in social_neighbors
            ])
            user_matrix_fused = args.gamma * user_matrix + (1 - args.gamma) * social_vecs


            rec_dict = {}
            N = len(user_matrix_fused)
            chunk_size = (N + k - 1) // k  
            
            for chunk_idx in range(k):
                start = chunk_idx * chunk_size
                end = min((chunk_idx + 1) * chunk_size, N)
                batch_users = user_matrix_fused[start:end]  # [actual_chunk_size, D]
                
                with torch.no_grad():
                    chunk_sim = torch.softmax(batch_users @ item_matrix.T * args.alpha, dim=1)
                    chunk_rec = torch.multinomial(chunk_sim, args.num_rec, replacement=False)
                
                user_ids_chunk = list(user_init_vec_dict.keys())[start:end]
                for j, uid in enumerate(user_ids_chunk):
                    rec_dict[uid] = chunk_rec[j].cpu().numpy()
                
                del batch_users, chunk_sim, chunk_rec
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
    
    if args.s2 == 1:
        positive_effect = torch.sum(rec_items * decisions.unsqueeze(2) * (1 - args.rho), dim=1)  # [num_users, cate_num]
        negative_effect = torch.sum(rec_items * (~decisions).unsqueeze(2) * (1 + args.rho), dim=1)  # [num_users, cate_num]
    else:
        positive_effect = torch.sum(rec_items * decisions.unsqueeze(2), dim=1)  # [num_users, cate_num]
        negative_effect = torch.sum(rec_items * (~decisions).unsqueeze(2), dim=1)  # [num_users, cate_num]
    
    # Update vectors 
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


np.save(dataset + f"_mitigate/eva_dict_{args.name}.npy", eva_dict)