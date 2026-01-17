from logging import error
import os 
import pdb
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class STPDataset(Dataset):
    def __init__(self, root_dir, list_dir, transform=None, seq_len=12, img_size=192, return_type='vil'):
        '''初始化方法
        
        - params:
            - root_dir: 数据根目录
            - list_dir: 所有时间的id
            - transform: 数据变换
        '''
        self.root_dir = root_dir
        self.transform = transform
        self.seq_len = seq_len
        self.img_size = img_size
        self.return_type = return_type

        # read event list
        with open(list_dir, 'r', encoding='utf-8') as f:
            self.events = [line.strip() for line in f if line.strip()]

        self.ir069 = os.path.join(self.root_dir, 'ir069_npz')
        self.ir107 = os.path.join(self.root_dir, 'ir107_npz')
        self.vil = os.path.join(self.root_dir, 'vil_npz')
        self.lght = os.path.join(self.root_dir, 'lght_npz')

        self._validate_events()

    def _validate_events(self):
        '''验证事件是否合法:
        - 验证是否四种模态的数据都有对应数据存在
        - 将合法的事件id作为最终数据的id
        '''
        valid_events = []
        for event_id in self.events:
            files_exist = all(os.path.exists(os.path.join(data_dir, f"{event_id}.npz"))
                for data_dir in [self.ir069, self.ir107, self.vil, self.lght])
            if files_exist:
                valid_events.append(event_id)
            else:
                print(f"Warning: Event {event_id} lose some modality，skip")
        
        if len(valid_events) == 0:
            raise RuntimeError("No valid event is found! Please check the data directory and event list.")
        
        print(f"{len(valid_events)} valid events")
        self.events = valid_events

    def __len__(self):
        return len(self.events)

    def _load_and_stack_npz(self, file_path):
        '''堆叠时序序列
        '''
        with np.load(file_path) as data:
            arrays = [data[k] for k in sorted(data.keys())]
            stacked = np.stack(arrays, axis=0) # [1, T, H, W]
            return np.expand_dims(stacked, axis=-1) # [1, T, H, W, 1]

    def __getitem__(self, idx):
        '''
        返回样本信息

        parameters
        --------------
        idx : int
            事件下标
        
        returns
        -------------
        input_norm : [::6,4,192,192]
            输入的标准化后的数据
        label_norm : [6::,4,192,192]
            标签数据
        event_id : str
            事件名称
        '''
        event_id = self.events[idx]

        ir069_paths = os.path.join(self.ir069, f"{event_id}.npz")
        ir107_paths = os.path.join(self.ir107, f"{event_id}.npz")
        vil_paths = os.path.join(self.vil, f"{event_id}.npz")
        lght_paths = os.path.join(self.lght, f"{event_id}.npz")

        ir069_frames = self._load_and_stack_npz(ir069_paths).squeeze(0)  # (T, 192, 192, 1)
        ir107_frames = self._load_and_stack_npz(ir107_paths).squeeze(0)  # (T, 192, 192, 1)
        vil_frames = self._load_and_stack_npz(vil_paths).squeeze(0)  # (T, H, W, 1)
        lght_frames = self._load_and_stack_npz(lght_paths).squeeze(0)  # (T, 48, 48, 1)

        ir069_tensor = torch.from_numpy(ir069_frames).permute(0, 3, 1, 2).float()
        ir107_tensor = torch.from_numpy(ir107_frames).permute(0, 3, 1, 2).float()
        vil_tensor = torch.from_numpy(vil_frames).permute(0, 3, 1, 2).float()
        lght_tensor = torch.from_numpy(lght_frames).permute(0, 3, 1, 2).float()

        # resize to target img_size
        vil_tensor = F.interpolate(vil_tensor, size=(self.img_size, self.img_size), mode='area')
        lght_tensor = F.interpolate(lght_tensor, size=(self.img_size, self.img_size), mode='nearest')
        lght_tensor = TF.gaussian_blur(lght_tensor, kernel_size=5, sigma=1.5)

        # clamp lighting
        lght_tensor = torch.clamp(lght_tensor, max=60.0)

        # normalize
        ir069_norm = (ir069_tensor + 3830.0) / 1527.0
        ir107_norm = (ir107_tensor + 1373.0) / 3173.0
        vil_norm = vil_tensor / 255.0
        lght_norm = torch.log2(lght_tensor + 1) / 6.0

        if self.return_type == 'lighting':
            # Return the `lght` modality as the sequence tensor instead of `vil`.
            # Ensure temporal length equals self.seq_len by trimming or padding.
            T = lght_norm.shape[0]
            if T >= self.seq_len:
                seq = lght_norm[-self.seq_len:]
            else:
                pad_len = self.seq_len - T
                pad_tensor = torch.zeros((pad_len, lght_norm.shape[1], self.img_size, self.img_size), dtype=lght_norm.dtype)
                seq = torch.cat((pad_tensor, lght_norm), dim=0)
        elif self.return_type == 'vil':
            # We only return the `vil` modality as a sequence tensor.
            # Ensure temporal length equals self.seq_len by trimming or padding.
            T = vil_norm.shape[0]
            if T >= self.seq_len:
                seq = vil_norm[-self.seq_len:]
            else:
                pad_len = self.seq_len - T
                pad_tensor = torch.zeros((pad_len, vil_norm.shape[1], self.img_size, self.img_size), dtype=vil_norm.dtype)
                seq = torch.cat((pad_tensor, vil_norm), dim=0)
        else:
            raise ValueError(f"Invalid return type: {self.return_type}")
        # final shape: (T, C, H, W)
        return seq