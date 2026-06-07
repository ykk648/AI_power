# green_screen_matting

基于 [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) 的绿幕抠图与背景替换推理模块。支持视频逐帧抠图，将绿幕视频人物与指定背景合成并输出视频。

---

## 快速开始（3 步）

1. **环境**：安装 Python 3.7+、PyTorch、FFmpeg，安装依赖 `pip install -r requirements.txt`
2. **权重**：下载 `pytorch_resnet50.pth` 放到 `content/pytorch_resnet50.pth`
3. **运行**：
   ```python
   from seg_lib.green_screen_matting import green_screen_matting

   green_screen_matting(video_path, bg_path, new_bg_path, output_dir, video_shape, 'cuda')
   ```
   或命令行：
   ```bash
   python green_screen_matting_api.py --name <名称>
   ```

---

## 模型权重

- 使用 **MattingRefine + ResNet50** 时，需下载预训练权重 `pytorch_resnet50.pth`（可参考 [BackgroundMattingV2 发布页](https://github.com/PeterL1n/BackgroundMattingV2#pre-trained-models)）。
- 将权重放在本模块下的 `content/pytorch_resnet50.pth`。

若修改路径，可在调用时传入 `model_checkpoint` 参数指定自定义路径。

---

## 目录结构约定

```
green_screen_matting/
├── __init__.py
├── green_screen_matting_api.py  # 推理入口
├── content/                     # 模型权重
│   └── pytorch_resnet50.pth
├── data/                        # 默认资源
│   ├── 绿幕1.png
│   └── 0.jpg
├── model/                       # 模型定义
│   ├── model.py
│   ├── decoder.py
│   ├── refiner.py
│   ├── resnet.py
│   ├── mobilenet.py
│   └── utils.py
└── dataset/                     # 数据集相关
    ├── images.py
    ├── video.py
    ├── sample.py
    ├── zip.py
    └── augmentation.py
```

---

## 使用方法

### 程序调用

```python
from seg_lib.green_screen_matting import green_screen_matting

video_path = "path/to/input.mp4"
bg_path = "path/to/绿幕.png"
new_bg_path = "path/to/新背景.jpg"
output_dir = "path/to/output"
video_shape = (height, width, 3)
device_type = "cuda"  # 或 "cpu"

green_screen_matting(
    video_path, bg_path, new_bg_path, output_dir, video_shape, device_type,
    model_checkpoint="optional/path/to/weights.pth"  # 可选
)
```

运行后会在 `output_dir` 下生成 `*_lb.avi` 和（若 FFmpeg 可用）带音频的 `*_lb.mp4`。

### 命令行

```bash
python green_screen_matting_api.py --name <name> [--lb 绿幕图路径] [--new_bg 新背景图路径]
```

参数说明：

| 参数       | 说明 |
|------------|------|
| `--name`   | **必填**。对应 `../result/<name>/test_video/` 下的视频，脚本会处理该目录下所有 `.mp4`。 |
| `--lb`     | 绿幕参考图路径。不填则使用默认 `data/绿幕1.png`。 |
| `--new_bg` | 新背景图路径。不填则使用默认 `data/0.jpg`。 |

---

## 模型与配置说明

- **模型类型**：MattingRefine（Base + Refine）
- **骨干网络**：ResNet50，scale 0.5
- **权重路径**：`content/pytorch_resnet50.pth`（可通过参数覆盖）
- **设备**：默认 `cuda`，通过 `device_type="cpu"` 切换

---

## 常见问题

1. **CUDA out of memory**
   视频分辨率过高时会占较多显存。脚本内对超过 3000 像素的一边做了缩小处理，若仍 OOM，可考虑进一步缩小输入分辨率或使用 CPU。

2. **无 FFmpeg**
   未安装时无法从 mp4 提取音频并合成最终 mp4，仅有 avi 输出。请安装 FFmpeg 并加入 PATH。

3. **Windows 兼容**
   临时文件使用 Python 的 `os.remove` 删除，无需额外命令。若 FFmpeg 报错，请确认已安装并加入 PATH。
