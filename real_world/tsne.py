import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  
import seaborn as sns


def plot(x, args, epoch):
    palette = np.array(sns.color_palette("tab20"))
    selected_colors = [palette[i] for i in [0, 2, 4, 6, 8, 10, 12]]
    f = plt.figure(figsize=(10, 10))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=10, c=sns.color_palette("tab20")[6])
    plt.savefig(f'{args.dataset}_tsne/{args.dataset}_alpha_{args.alpha}_beta_{args.beta}_gamma_{args.gamma}_epoch_{epoch}.png', dpi=120)
    return f, ax

def draw_tsne(user_vec_dict, args, epoch):
    user_embed = np.array([i.cpu().numpy() for i in user_vec_dict.values()])
    embed_final = TSNE(perplexity=30).fit_transform(user_embed) 
    plot(embed_final, args, epoch)

