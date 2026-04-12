import sys, os, torch

if "ck696" in os.getcwd():
    sys.path.append("/share/hariharan/ck696/allclear")
else:
    sys.path.append("/share/hariharan/cloud_removal/allclear")

from dataset.dataloader_v1 import CRDataset
from torch.utils.data import DataLoader, Dataset

class CRDatasetWrapper(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        batch = self.original_dataset[idx]
        
        return {"x": batch["input_images"].permute(1,0,2,3), 
                "y": batch["target"].permute(1,0,2,3),
                "masks": batch["input_cld_shdw"].permute(1,0,2,3).max(dim=1, keepdim=True).values,
                "position_days": batch["time_differences"],
                "days": batch["time_differences"],
                "sample_index": 0,
                "c_index_rgb": torch.Tensor([0,1,2]),
                "c_index_nir": torch.Tensor([3]),
                "cloud_mask": batch["target_cld_shdw"].permute(1,0,2,3).max(dim=1, keepdim=True).values
               }    

## Load and changes
# import json
# with open('/share/hariharan/cloud_removal/metadata/v3/s2s_tx6_v1.json') as f:
#     metadata = json.load(f)
    
# for i in range(len(metadata)):
#     for j in range(6):
#         metadata[f"{i}"]["s2_toa"][j][1] = "/share/hariharan/cloud_removal/MultiSensor/dataset_30k_v4/" + metadata[f"{i}"]["s2_toa"][j][1].split("dataset_30k_v4")[1]
#     try:
#         for j in range(6):
#             metadata[f"{i}"]["s1"][j][1] = "/share/hariharan/cloud_removal/MultiSensor/dataset_30k_v4/" + metadata[f"{i}"]["s1"][j][1].split("dataset_30k_v4")[1]
#     except:
#         pass
# # save metadata to json
# file_name = '/share/hariharan/cloud_removal/metadata/v3/s2s_tx6_v1_bh.json'
# with open(file_name, 'w') as f:
#     json.dump(metadata, f)

def get_loader(config):
    import json
    with open('/share/hariharan/cloud_removal/metadata/v3/s2s_tx6_v1_bh.json') as f:
        metadata = json.load(f)
        
    clds_shdws = torch.ones(1000, 2, 256, 256)
    
    train_data = CRDataset(metadata, 
                        selected_rois="all", 
                        main_sensor="s2_toa", 
                        aux_sensors=[],
                        aux_data=["cld_shdw"],
                        format="stp",
                        target="s2s",
                        clds_shdws=clds_shdws,
                        tx=6,
                        s2_toa_channels=[4,3,2,8]
                        )
    
    wrapped_train_data = CRDatasetWrapper(train_data)
    phase_loader = DataLoader(wrapped_train_data, batch_size=config["training_settings"]["batch_size"], shuffle=True, num_workers=config["misc"]["num_workers"], pin_memory=True)
    return phase_loader