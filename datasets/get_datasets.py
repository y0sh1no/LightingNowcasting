
import torch
import os
import numpy as np
import os.path as osp
import datetime

from functools import partial
from matplotlib import colors
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from matplotlib.colors import LinearSegmentedColormap

HMF_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219]
]) / 255


def vis_res(pred_seq, gt_seq, save_path, data_type='vil',
            save_grays=False, do_hmf=False, save_colored=False,save_gif=False,
            pixel_scale = None, thresholds = None, gray2color = None
            ):
    # pred_seq: ndarray, [T, C, H, W], value range: [0, 1] float
    if isinstance(pred_seq, torch.Tensor) or isinstance(gt_seq, torch.Tensor):
        pred_seq = pred_seq.detach().cpu().numpy()
        gt_seq = gt_seq.detach().cpu().numpy()
    pred_seq = pred_seq.squeeze()
    gt_seq = gt_seq.squeeze()
    os.makedirs(save_path, exist_ok=True)

    if save_grays:
        os.makedirs(osp.join(save_path, 'pred'), exist_ok=True)
        os.makedirs(osp.join(save_path, 'targets'), exist_ok=True)
        for i, (pred, gt) in enumerate(zip(pred_seq, gt_seq)):
            
            # cv2.imwrite(osp.join(save_path, 'pred', f'{i}.png'), (pred * PIXEL_SCALE).astype(np.uint8))
            # cv2.imwrite(osp.join(save_path, 'targets', f'{i}.png'), (gt * PIXEL_SCALE).astype(np.uint8))
            
            plt.imsave(osp.join(save_path, 'pred', f'{i}.png'), pred, cmap='gray', vmax=1.0, vmin=0.0)
            plt.imsave(osp.join(save_path, 'targets', f'{i}.png'), gt, cmap='gray', vmax=1.0, vmin=0.0)


    if data_type=='vil':
        pred_seq = pred_seq * pixel_scale
        pred_seq = pred_seq.astype(np.uint8)
        gt_seq = gt_seq * pixel_scale
        gt_seq = gt_seq.astype(np.uint8)
    
    colored_pred = np.array([gray2color(pred_seq[i], data_type=data_type) for i in range(len(pred_seq))], dtype=np.float64)
    colored_gt =  np.array([gray2color(gt_seq[i], data_type=data_type) for i in range(len(gt_seq))],dtype=np.float64)

    if save_colored:
        os.makedirs(osp.join(save_path, 'pred_colored'), exist_ok=True)
        os.makedirs(osp.join(save_path, 'targets_colored'), exist_ok=True)
        for i, (pred, gt) in enumerate(zip(colored_pred, colored_gt)):
            plt.imsave(osp.join(save_path, 'pred_colored', f'{i}.png'), pred)
            plt.imsave(osp.join(save_path, 'targets_colored', f'{i}.png'), gt)


    grid_pred = np.concatenate([
        np.concatenate([i for i in colored_pred], axis=-2),
    ], axis=-3)
    grid_gt = np.concatenate([
        np.concatenate([i for i in colored_gt], axis=-2,),
    ], axis=-3)
    
    grid_concat = np.concatenate([grid_pred, grid_gt], axis=-3,)
    plt.imsave(osp.join(save_path, 'all.png'), grid_concat)
    
    if save_gif:
        clip = ImageSequenceClip(list(colored_pred * 255), fps=4)
        clip.write_gif(osp.join(save_path, 'pred.gif'), fps=4, verbose=False)
        clip = ImageSequenceClip(list(colored_gt * 255), fps=4)
        clip.write_gif(osp.join(save_path, 'targets.gif'), fps=4, verbose=False)
    
    if do_hmf:
        def hit_miss_fa(y_true, y_pred, thres):
            mask = np.zeros_like(y_true)
            mask[np.logical_and(y_true >= thres, y_pred >= thres)] = 4
            mask[np.logical_and(y_true >= thres, y_pred < thres)] = 3
            mask[np.logical_and(y_true < thres, y_pred >= thres)] = 2
            mask[np.logical_and(y_true < thres, y_pred < thres)] = 1
            return mask
            
        grid_pred = np.concatenate([
            np.concatenate([i for i in pred_seq], axis=-1),
        ], axis=-2)
        grid_gt = np.concatenate([
            np.concatenate([i for i in gt_seq], axis=-1),
        ], axis=-2)

        hmf_mask = hit_miss_fa(grid_pred, grid_gt, thres=thresholds[2])
        plt.axis('off')
        plt.imsave(osp.join(save_path, 'hmf.png'), hmf_mask, cmap=colors.ListedColormap(HMF_COLORS))

def lighting_gray2color(img, data_type='lighting'):
    """
    参考 MCGLN 风格的可视化：
    1. 频次为 0 或极低时显示为白色 (White)
    2. 随着频次增加，颜色从 蓝 -> 绿 -> 黄 -> 红 (类似 Jet 但起始点为白)
    """
    # 确保 img 是 2D (H, W)
    if img.ndim == 3:
        img = img.squeeze()
        
    # 归一化并限制范围
    img = np.clip(img, 0.0, 1.0)

    # 定义颜色转折点：从白色开始，然后进入传统的强对流天气配色
    # 0.0: 白色 (无闪电)
    # 0.01 - 0.2: 淡蓝色/蓝色 (低频)
    # 0.5: 绿色/黄色 (中频)
    # 1.0: 深红色 (高频)
    colors = [
        (1.0, 1.0, 1.0), # 0: 白色
        (0.0, 0.0, 1.0), # 0.2: 蓝色
        (0.0, 1.0, 1.0), # 0.4: 青色
        (0.0, 1.0, 0.0), # 0.6: 绿色
        (1.0, 1.0, 0.0), # 0.8: 黄色
        (1.0, 0.0, 0.0)  # 1.0: 红色
    ]
    
    # 创建自定义 Colormap
    mcgln_cmap = LinearSegmentedColormap.from_list('mcgln_style', colors, N=256)
    
    # 获取 RGB 图像
    color_img = mcgln_cmap(img)[:, :, :3]
    
    return color_img

DATAPATH = {
    'cikm' : './datasets/cikm.h5',
    'shanghai' : './datasets/shanghai.h5',
    'meteo' : './datasets/meteo_radar.h5',
    'sevir' : './datasets/sevir',
    'lighting' : './datasets/sevir'
}

def get_dataset(data_name, img_size, seq_len, **kwargs):
    dataset_name = data_name.lower()
    train = val = test = None

    if dataset_name == 'cikm':
        from .dataset_cikm import CIKM, gray2color, PIXEL_SCALE, THRESHOLDS
        
        train = CIKM(DATAPATH[data_name], 'train', img_size)
        val = CIKM(DATAPATH[data_name], 'valid', img_size)
        test = CIKM(DATAPATH[data_name], 'test', img_size)
        
    elif data_name == 'shanghai':
        from .dataset_shanghai import Shanghai, gray2color, THRESHOLDS, PIXEL_SCALE
        train = Shanghai(DATAPATH[data_name], type='train', img_size=img_size)
        val = Shanghai(DATAPATH[data_name], type='val', img_size=img_size)
        test = Shanghai(DATAPATH[data_name], type='test', img_size=img_size)
    
    elif data_name == 'meteo':
        from .dataset_meteonet import Meteo, gray2color, THRESHOLDS, PIXEL_SCALE
        train = Meteo(DATAPATH[data_name], type='train', img_size=img_size)
        val = Meteo(DATAPATH[data_name], type='val', img_size=img_size)
        test = Meteo(DATAPATH[data_name], type='test', img_size=img_size)
        
    elif dataset_name == 'sevir':
        from .dataset_sevir import gray2color, PIXEL_SCALE, THRESHOLDS
        from .my_dataloader import STPDataset

        # Expecting event id list files under the SEVIR folder:
        # train_list.txt, val_list.txt, test_list.txt

        root = DATAPATH[data_name]
        train_path = osp.join(root, 'training')
        val_path = osp.join(root, 'validation')
        test_path = osp.join(root, 'testing')

        train_list = osp.join(train_path, 'training.txt')
        val_list = osp.join(val_path, 'validation.txt')
        test_list = osp.join(test_path, 'testing.txt')

        train = STPDataset(train_path, train_list, transform=None, seq_len=seq_len, img_size=img_size, return_type='vil')
        val = STPDataset(val_path, val_list, transform=None, seq_len=seq_len, img_size=img_size, return_type='vil')
        test = STPDataset(test_path, test_list, transform=None, seq_len=seq_len, img_size=img_size, return_type='vil')
    
    # --- 新增 lighting 分支 ---
    elif dataset_name == 'lighting':
        from .my_dataloader import STPDataset  # 复用你的 STPDataset
        
        # 参数配置
        PIXEL_SCALE = 1.0  # 已经是 log 归一化后的数据，保持 1.0 即可
        # 对应原始闪电次数 [1, 5, 15] 的 log2 归一化值: log2(x+1)/6
        THRESHOLDS = [0.16, 0.43, 0.66] 
        
        gray2color = lighting_gray2color  # 使用上面定义的配色函数

        root = DATAPATH[data_name]
        train_path = osp.join(root, 'training')
        val_path = osp.join(root, 'validation')
        test_path = osp.join(root, 'testing')

        # 确保你的 STPDataset 初始化参数与这里一致
        # 如果没有 txt 列表文件，请修改 STPDataset 代码或在这里传入 None
        train_list = osp.join(train_path, 'training.txt')
        val_list = osp.join(val_path, 'validation.txt')
        test_list = osp.join(test_path, 'testing.txt')

        train = STPDataset(train_path, train_list, transform=None, seq_len=seq_len, img_size=img_size, return_type='lighting')
        val = STPDataset(val_path, val_list, transform=None, seq_len=seq_len, img_size=img_size, return_type='lighting')
        test = STPDataset(test_path, test_list, transform=None, seq_len=seq_len, img_size=img_size, return_type='lighting')

    color_fn = partial(vis_res, 
                    pixel_scale = PIXEL_SCALE, 
                    thresholds = THRESHOLDS, 
                    gray2color = gray2color)
    
    return train, val, test, color_fn, PIXEL_SCALE, THRESHOLDS
