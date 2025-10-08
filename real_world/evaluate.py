import numpy as np
import torch

def calc_rec_cate_entropy(recom_dict, cate_01):
    total_entro = 0.
    for uid, rec_list in recom_dict.items():
        cate_list = torch.stack([cate_01[i] for i in rec_list]).sum(dim=0)
        probability = cate_list / cate_list.sum()
        total_entro += -torch.sum(probability * torch.log2(probability + 1e-10)).item()
    return total_entro / len(recom_dict)


def calc_rec_accuracy(user_vec_dict, threshold, recom_dict, cate_vec):
    total_interest = 0
    total_items = 0
    for uid, u_vec in user_vec_dict.items():
        rec_vec = torch.stack([cate_vec[i] for i in recom_dict[uid]])
        inner_product = torch.matmul(rec_vec, u_vec)
        total_interest += (inner_product >= threshold).sum().item()
        total_items += len(rec_vec)
    return total_interest / total_items

def calc_nei_distance(user_vec_dict, u1_list, u2_list):
    u1_vecs = torch.stack([user_vec_dict[i] for i in u1_list])
    u2_vecs = torch.stack([user_vec_dict[i] for i in u2_list])
    diff_norm = torch.norm(u1_vecs - u2_vecs, p=2, dim=1)
    return diff_norm.mean().item()



def calc_pair_dis_variance(user_vec_dict, chunk_size=2000):

    X = torch.stack(list(user_vec_dict.values()))  # [N, D]
    N, D = X.shape
    device = X.device
    
    all_distances = []
    
    for i in range(0, N, chunk_size):
        start_i = i
        end_i = min(i + chunk_size, N)
        chunk_i = X[start_i:end_i]
        
        diff_intra = None  
        if len(chunk_i) > 1:
            diff_intra = chunk_i.unsqueeze(1) - chunk_i.unsqueeze(0)
            dist_intra = torch.norm(diff_intra, p=2, dim=-1)
            mask = torch.triu(torch.ones_like(dist_intra), diagonal=1).bool()
            all_distances.append(dist_intra[mask])
        
        diff_inter = None 
        for j in range(i + chunk_size, N, chunk_size):
            start_j = j
            end_j = min(j + chunk_size, N)
            chunk_j = X[start_j:end_j]
            
            diff_inter = chunk_i.unsqueeze(1) - chunk_j.unsqueeze(0)
            dist_inter = torch.norm(diff_inter, p=2, dim=-1)
            all_distances.append(dist_inter.flatten())
        
        vars_to_del = ['chunk_i']
        if diff_intra is not None:
            vars_to_del.extend(['diff_intra', 'dist_intra'])
        if diff_inter is not None:
            vars_to_del.extend(['diff_inter', 'dist_inter'])
            
        for var in vars_to_del:
            if var in locals():
                del locals()[var]
        torch.cuda.empty_cache()
    
    if not all_distances:
        return 0.0
    return torch.var(torch.cat(all_distances), unbiased=False).item()


def calc_topk_similarity(user_vec_dict, args):
    X = torch.stack(list(user_vec_dict.values()))
    
    sim_mat = torch.mm(X, X.T)  
    
    sim_mat = sim_mat - torch.eye(sim_mat.size(0), device=sim_mat.device)
    
    if args.dataset == "ciao":
        topk_values, _ = torch.topk(sim_mat, k=300, dim=1)  
    elif args.dataset == "epinions":
        topk_values, _ = torch.topk(sim_mat, k=900, dim=1)  
    row_max_avg = topk_values.mean()
    return row_max_avg.item()  