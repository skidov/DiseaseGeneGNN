import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data


datafile_path = 'datasets/DisGeNET/raw/browser_source_summary_gda_CURATED.tsv'
dz_col, gene_col = "Disease_id", "Gene_id"

df = pd.read_csv(datafile_path, index_col=dz_col, sep="\t")
dz_mapping = {index_id: i + 0 for i, index_id in enumerate(df.index.unique())}
df = pd.read_csv(datafile_path, index_col=gene_col, sep="\t")
gene_mapping = {index_id: i + 0 for i, index_id in enumerate(df.index.unique())}

df = pd.read_csv(datafile_path, sep="\t")
dz_nodes = [dz_mapping[index] for index in df[dz_col]]
gene_nodes = [gene_mapping[index] for index in df[gene_col]]
edge_index = torch.tensor([dz_nodes, gene_nodes])
rev_edge_index = torch.tensor([gene_nodes, dz_nodes])

data = Data()
data.num_nodes = len(dz_mapping) + len(gene_mapping)
data.edge_index = torch.cat((edge_index, rev_edge_index), dim=1)
data.x = torch.ones((data.num_nodes, 20))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.15, is_undirected=True,
                        split_labels=True, add_negative_train_samples=True),
])

train_data, val_data, test_data = transform(data)
print(data)
print("Train Data:\n", train_data)
print("Validation Data:\n", val_data)
print("Test Data:\n", test_data)
