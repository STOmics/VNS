# Variable Neighborhood Search Approach for Cell Clustering
The Variable Neighborhood Search (VNS) method is well known metaheuristic method, which starts from one point from the search space, explores its neighborhoods and repeats the whole process until better solution is found or some stopping criteria is reached. Leveraging the well-established foundation of VNS, first we present a comprehensive solution for the cell clustering problem in the form of the Integer Linear Programming (ILP) minimization problem, which is based on the p-median classification. The proposed algorithm exhibits the ability to organize cells into clusters, utilizing information from both gene expression matrices and spatial coordinates.

## Parameters
| Name  | Type | Definition | Default |
| ------------- | ------------- | ------------- | ------------- |
| adata_path  | str  | Path to adata file  | data.h5ad  |
| emb | str  | Choose embeddings ('CCST', 'GraphST', 'STAGATE', 'X_pca', 'STAligner')  | X_pca  |
| k  | int  | Number of clusters  | 20  |
| max_iter  | int  | Maximal number of iterations   | 20  |
| p  | int  | _LocalSearch_ parameter  | 12  |
| m  | int  | _LocalSearch_ parameter  | 15  |
| alpha  | int | Precentage of the influence of the embedding values  | 1  |
| s  | int  | Maximal number of neighborhoods that should be searched  | 10  |
| seed  | int  | Seed value  | 1234  |

