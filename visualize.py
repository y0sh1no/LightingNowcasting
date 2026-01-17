
import os
import torch
import numpy as np
import argparse
from datasets.get_datasets import get_dataset
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='lighting', help="dataset name")
    parser.add_argument("--img_size", type=int, default=192, help="image size")
    parser.add_argument("--seq_len", type=int, default=12, help="sequence length")
    parser.add_argument("--save_dir", type=str, default='./vis_results', help="directory to save visualization")
    args = parser.parse_args()

    print(f"Initializing dataset: {args.dataset}...")
    
    # 获取数据集
    # 注意：get_dataset 返回的是 (train, val, test, color_fn, pixel_scale, thresholds)
    train_dataset, val_dataset, test_dataset, color_fn, pixel_scale, thresholds = get_dataset(
        data_name=args.dataset,
        img_size=args.img_size,
        seq_len=args.seq_len
    )

    print(f"Train dataset size: {len(train_dataset)}")
    
    # 创建 DataLoader 取一个 batch
    loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    # 获取一个样本
    for i, batch in enumerate(loader):
        # batch shape: [B, T, C, H, W]
        print(f"Batch shape: {batch.shape}")
        
        # 拆分为输入和输出 (这里只是为了可视化，我们把前6帧当做输入，后6帧当做GT，或者全画出来)
        # 假设我们要可视化整个序列
        seq = batch[0] # [T, C, H, W]
        
        # 为 color_fn 准备数据
        # color_fn 需要 (pred_seq, gt_seq, save_path, ...)
        # 我们把 seq 当做 gt, 另外造一个 zeros 当做 pred (或者也用 seq) 来演示
        
        # 可视化前5个样本
        if i >= 5:
            break
            
        save_path = os.path.join(args.save_dir, f"sample_{i}")
        os.makedirs(save_path, exist_ok=True)
        
        print(f"Visualizing sample {i} to {save_path}...")
        
        # 调用 get_dataset 返回的 color_fn
        # 注意：color_fn 签名是 (pred_seq, gt_seq, save_path, data_type='vil', save_grays=False, do_hmf=False, save_colored=False, save_gif=False)
        # lighting 模式下 data_type 应该已经在 color_fn 中被 partial 绑定了? 
        # 查看 get_datasets.py: 
        # color_fn = partial(vis_res, pixel_scale=..., thresholds=..., gray2color=...)
        # 并没有绑定 data_type。lighting 的 get_dataset 分支设定了 gray2color 为 lighting_gray2color。
        # 但是 vis_res 里面默认 gray2color 调用时传了 data_type。
        
        # 由于我们传入的 gray2color (lighting_gray2color) 接受 data_type参数 (虽然可能不用), 所以传递 data_type='lighting' 比较安全
        
        color_fn(
            pred_seq=seq,      # 这里为了演示，把 GT 同时作为 pred 传进去，方便看图
            gt_seq=seq, 
            save_path=save_path, 
            data_type='lighting',
            save_grays=True, 
            save_colored=True, 
            save_gif=True
        )
        
    print("Done!")

if __name__ == "__main__":
    main()
