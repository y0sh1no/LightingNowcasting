import numpy as np
import matplotlib.pyplot as plt

# 配置参数
npz_file_path = "./dataset/training/vil_npz/S737653_1.npz"      # 替换为你的 .npz 文件路径
variable_name = "data"

# 加载 .npz 文件
data = np.load(npz_file_path)
if variable_name not in data:
    raise KeyError(f"NPZ 文件中没有名为 '{variable_name}' 的变量。可用键: {list(data.keys())}")

frames = data[variable_name]
print(f"数据形状: {frames.shape}")

# 检查维度是否符合预期，ir为(12, 192, 192)，vil为(12, 384, 384), lght为(12, 48, 48)
if frames.ndim != 3 or frames.shape[0] != 12 or frames.shape[1:] != (384, 384):
    raise ValueError(f"期望形状为 (12, 384, 384)，但实际为 {frames.shape}")

# 初始化当前帧索引
current_index = 0

# 创建图形和轴
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(frames[current_index], cmap='viridis', origin='upper')
ax.set_title(f"Frame {current_index + 1} / {frames.shape[0]}")
ax.axis('off')

# 定义键盘事件回调函数
def on_key(event):
    global current_index
    if event.key == 'right':
        if current_index < frames.shape[0] - 1:
            current_index += 1
    elif event.key == 'left':
        if current_index > 0:
            current_index -= 1
    else:
        return

    im.set_data(frames[current_index])
    ax.set_title(f"Frame {current_index + 1} / {frames.shape[0]}")
    fig.canvas.draw()

fig.canvas.mpl_connect('key_press_event', on_key)

print("使用 ← 和 → 键切换帧（左/右箭头）")
print("关闭窗口退出")

plt.show()
data.close()
