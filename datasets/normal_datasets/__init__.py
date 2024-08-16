from datasets.normal_datasets.replica_dataset import ReplicaDataset
from datasets.normal_datasets.scannet_dataset import ScanNetDataset

dataset_dict = {
    'replica': ReplicaDataset,
    'scannet': ScanNetDataset,
}
