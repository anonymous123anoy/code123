from scipy import stats
import numpy as np

dataset = "ciao"

metric_name = ['entropy', 'proportion', 'neighbor', 'disvar', 'maxsim']

if dataset == "ciao":
    ori_name = dataset + "_mitigate/eva_dict_alpha_5.0"
    solve1_name = dataset + "_mitigate/eva_dict_s1_10"
    solve2_name = dataset + "_mitigate/eva_dict_s2_0.02"
    solve3_name = dataset + "_mitigate/eva_dict_s3_0.501"
    solve4_name = dataset + "_mitigate/eva_dict_s4_1000"
    
    solve_list = [solve1_name, solve2_name, solve3_name, solve4_name]
    metrics = ['RCE', 'RA', 'ND', 'PDV', 'TS@300']

elif dataset == "epinions":
    ori_name = dataset + "_mitigate/eva_dict_alpha_5.0"
    solve1_name = dataset + "_mitigate/eva_dict_s1_10"
    solve2_name = dataset + "_mitigate/eva_dict_s2_0.02"
    solve3_name = dataset + "_mitigate/eva_dict_s3_0.501"
    solve4_name = dataset + "_mitigate/eva_dict_s4_150"
    
    solve_list = [solve1_name, solve2_name, solve3_name, solve4_name]
    metrics = ['RCE', 'RA', 'ND', 'PDV', 'TS@900']

mitigate_list = ["User Adaptive alpha", "Feedback Update Adjustment", "Diversity Post-Processing", "Social Aggregation Reweighting"]

for i in range(len(solve_list)):
    solve_name = solve_list[i]
    mitigate = mitigate_list[i]
    print(f"Mitigate: {mitigate}")
    for j, metric in enumerate(metric_name):
        ori = [np.mean(np.load(ori_name + ".npy", allow_pickle=True).item()[metric]), np.mean(np.load(ori_name + "_42.npy", allow_pickle=True).item()[metric]), np.mean(np.load(ori_name + "_123.npy", allow_pickle=True).item()[metric])]
        solve = [np.mean(np.load(solve_name + ".npy", allow_pickle=True).item()[metric]), np.mean(np.load(solve_name + "_42.npy", allow_pickle=True).item()[metric]), np.mean(np.load(solve_name + "_123.npy", allow_pickle=True).item()[metric])]
        t_stat, p_value = stats.ttest_rel(solve, ori)  
        ori_mean = np.mean(ori)
        ori_std = np.std(ori)
        solve_mean = np.mean(solve)
        solve_std = np.std(solve)
        if j == 3 or j == 4:
            improve = -(solve_mean - ori_mean) / ori_mean * 100
        else:
            improve = (solve_mean - ori_mean) / ori_mean * 100
        print(f"Metric: {metrics[j]}, Ori: {ori_mean:.2f}±{ori_std:.3f}, Solve: {solve_mean:.2f}±{solve_std:.3f}, Improve: {improve:.2f}%, P-value: {p_value:.2f}")
    print("\n")
