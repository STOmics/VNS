import numpy as np
import random
import time
import copy
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances, calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score
import argparse

# VNS Functions

def initial_solution(k, n):
    return np.random.choice(np.arange(1, n+1), size=k, replace=False)

def objective_function(x, distance):
    min_dist_vec = np.min(distance[:, x - 1], axis=1)
    fx = np.sum(min_dist_vec)
    return fx

def shaking(x, l):
    x1 = copy.deepcopy(x)
    indices = random.sample(range(0, k), l)
    clust = np.arange(1, n + 1)
    clust = np.delete(clust, x1 - 1)
    clust_ind = np.random.choice(len(clust), size=l, replace=False)
    x1[indices] = clust_ind
    return x1

def local_search(x, distance, m, parameter):
    fx_best = objective_function(x, distance)
    count = 0
    while True:
        x1 = copy.deepcopy(x)
        improvement = 0
        while count < parameter:
            for i in range(len(x1)):
                ind = random.randint(0, m)
                while ind == 1:
                    ind = random.randint(0, m)
                if ind == 0:
                    continue
                ind_closest = np.argpartition(distance[x1[i] - 1, :], ind)[:ind+1]
                if np.all(np.isin(ind_closest + 1, x1)):
                    index = random.randint(0, n - 1)
                    while index + 1 in x1:
                        index = random.randint(0, n - 1)
                else:
                    index = np.random.choice(ind_closest)
                    while index + 1 in x1:
                        index = np.random.choice(ind_closest)
            
                x1[i] = index + 1
            
            count += 1
            f1 = objective_function(x1, distance)
            if f1 < fx_best:
                x = copy.deepcopy(x1)
                fx_best = f1
                improvement = 1
                break
        if improvement == 0:
            break
    
    return x

def vns(max_iter, parameter, m, alpha, s, seed):
    random.seed(seed)
    np.random.seed(seed)
    distance = alpha * distance_gene + (1 - alpha) * coord
    
    x = initial_solution(k, n)
    f = objective_function(x, distance)
    x_best = x
    f_best = f
    i = 1

    while i <= max_iter:
        l = 1
        while l <= s:
            x1 = shaking(x, l)
            x2 = local_search(x1, distance, m, parameter)
            f2 = objective_function(x2, distance)

            if f_best > f2:
                f_best = f2
                x_best = copy.deepcopy(x2)
                l = 1
            else:
                l += 1
        i = i + 1
    return x_best, f_best

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Description of your script')

    # Add command-line arguments
    parser.add_argument('--adata_path', type=str, default='/goofys/BCO/Benchmark/SS200000128TR_E2_benchmark.h5ad', help='Path to adata file')
    parser.add_argument('--ground_truth', type=str, default='celltype_pred', help='Ground truth annotation in adata.obs')
    parser.add_argument('--emb', type=str, default='X_pca', choices=['CCST', 'GraphST', 'STAGATE', 'X_pca'], help='Choose embeddings')
    parser.add_argument('-k', type=int, default=20, help='Number of clusters')
    parser.add_argument('--max_iter', type=int, default=10, help='Maximum number of iterations')
    parser.add_argument('--parameter', type=int, default=12, help='Parameter value')
    parser.add_argument('-m', type=int, default=15, help='Value of m')
    parser.add_argument('--alpha', type=int, default=1, help='Value of alpha')
    parser.add_argument('-s', type=int, default=10, help='Value of s')
    parser.add_argument('--seed', type=int, default=4639, help='Seed value')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the command-line arguments
    adata_path = args.adata_path
    ground_truth = args.ground_truth
    emb = args.emb
    k = args.k
    max_iter = args.max_iter
    parameter = args.parameter
    m = args.m
    alpha = args.alpha
    s = args.s
    seed = args.seed

    adata = sc.read_h5ad(adata_path)
    n = adata.shape[0]
    print("Number of clusters: ", k)
    print("Number of cells: ", n)

    # Variable Neighborhood Search - calculating the distance

    df = pd.DataFrame(adata.obsm[emb])
    matrix = df.values
    metr = 'cosine'
    distance_gene = pdist(matrix, metric=metr)
    distance_gene = squareform(distance_gene)

    scaler = MinMaxScaler()
    distance_gene = scaler.fit_transform(distance_gene)

    x1 = adata.obsm['spatial'][:,0]
    y1 = adata.obsm['spatial'][:,1]
    coord = pd.DataFrame({'X': np.array(x1), 'Y': np.array(y1)})
    coord = cdist(coord, coord, metric='euclidean')
    coord = scaler.fit_transform(coord)

    distance = alpha * distance_gene + (1 - alpha) * coord

    # VNS Implementation

    print(f'max_iter, parameter, m, alpha, s, seed = [{max_iter}, {parameter}, {m}, {alpha}, {s}, {seed}]')
    start_time = time.time()
    x_best, f_best = vns(max_iter, parameter, m, alpha, s, seed)
    end_time = time.time()
    print('Best solution: ', x_best)
    print('Best objective function: ', f_best)

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    cell_solution = []
    for i in range(0, n):
        dist_list = []
        for j in range(0, k):
            dist_list.append(distance[i, x_best[j] - 1])

        min_value = min(dist_list)
        min_index = dist_list.index(min_value)
        cell_solution.append(x_best[min_index])

    adata.obs['VNS'] = cell_solution


