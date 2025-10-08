import numpy as np

cate_vec = np.load("cate_vector.npy", allow_pickle=True).item()


for i in cate_vec.keys():
    ori_vec = cate_vec[i]
    new_vec = ori_vec / np.linalg.norm(ori_vec, ord=2)
    cate_vec[i] = new_vec

for i in cate_vec.keys():
    new_vec = cate_vec[i]
    # print(np.linalg.norm(new_vec, ord=2))
    assert np.abs(np.linalg.norm(new_vec, ord=2) - 1.) <= 0.0000001


np.save("cate_vector_norm.npy", cate_vec)









