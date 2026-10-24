# 抽烟检测 Smoking Detection

输入一个视频地址或文件路径，使用 YOLO11 ONNX 模型显示疑似抽烟检测框。适合学习视频读取、模型推理和简单的跨帧显示处理。

支持 RTSP、HTTP/HTTPS 视频直链和本地视频文件。提供重叠分块检测、短时历史框保留和 Windows 窗口居中。使用 CPU 推理，无需安装 PyTorch 或准备 NVIDIA 显卡。

## 快速开始 Windows

建议使用 Python 3.12。在项目文件夹打开 PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_model.py
.\.venv\Scripts\python.exe app.py
```

随后粘贴视频地址，按回车开始。预览窗口按 `Q` 或 `Esc` 退出。

安装完成后，也可以双击 `run.bat` 启动。PyCharm 用户选择 `.venv\Scripts\python.exe` 作为解释器，运行 `app.py`。如果 PyCharm 2022.3 的安装工具提示找不到 `pkg_resources`，可安装 `requirements-pycharm.txt`。

Linux/macOS 可使用 `python3 -m venv .venv`，后续用 `.venv/bin/python` 执行同样的安装、下载和启动命令。Windows 自动居中功能只在 Windows 启用；其他系统仍可显示窗口。需要可用的桌面图形环境，服务器测试可加 `--headless`。

## 视频地址怎么填

本地视频：

```powershell
.\.venv\Scripts\python.exe app.py "D:\videos\sample.mp4"
```

网络视频：运行 `app.py` 后，在提示中输入实际 RTSP 地址或 HTTP 视频直链。普通网站的播放页面地址不能直接当作视频流。

```text
rtsp://<用户名>:<密码>@<摄像头IP>:554/<流路径>
https://<服务器>/<视频文件>.mp4
http://<服务器>:<端口>/<视频流>.flv
```

电脑必须能访问对应视频服务。摄像头需要开启 RTSP；可先用 VLC 验证地址、密码与编码。含特殊字符的用户名和密码需分别进行 URL 编码。本项目没有内置任何真实设备地址或密码。

也可以通过 `VIDEO_SOURCE` 环境变量传入视频地址，程序不会主动打印来源 URL。不要把带密码的地址写入公开代码或提交记录。

## 检测结果怎么看

| 显示 | 含义 |
| --- | --- |
| 红框 Smoking candidate | 当前帧检出疑似抽烟目标，附模型分数 |
| 黄框 Last seen ... ago | 当前帧未检出，短暂显示之前检出的位置 |
| No smoking candidate detected | 当前没有显示候选，不能据此证明没有吸烟 |

默认保留历史框 `0.6` 秒。未检出不会刷新这个计时，超过时限即清除。历史框不预测运动、不确认遮挡原因，也不能作为新的阳性证据或个人身份判断。

## 可调参数

```powershell
# 降低计算量：关闭分块增强
.\.venv\Scripts\python.exe app.py "D:\videos\sample.mp4" --no-tiles

# 调整分数阈值和历史框保留时间
.\.venv\Scripts\python.exe app.py "D:\videos\sample.mp4" --confidence 0.5 --hold-seconds 0.6

# 不打开窗口，处理前 100 帧，用于检查推理能否运行
.\.venv\Scripts\python.exe app.py "D:\videos\sample.mp4" --headless --max-frames 100

# 指定同类兼容的 ONNX 模型
.\.venv\Scripts\python.exe app.py "D:\videos\sample.mp4" --model "models\smoking.onnx"
```

完整参数用 `python app.py --help` 查看。降低阈值可能增加误报，不应把分数解释成真实准确率。

分块增强会检测整图和四个重叠区域，并用 NMS 合并重复框，每帧最多进行 5 次推理。有助于部分小目标，但更耗时；实时视频可能积累播放延迟。本地文件按帧号/帧率估算历史框时间，网络来源按电脑单调时间计时，均不代表已校验的设备事件时间。

## 模型来源

模型来自 [Ali Hassan 的 Smoking-detection-yolo11](https://github.com/alihassanml/Smoking-detection-yolo11)。固定提交为 `bbef478fad3d6d18e4a18bf02fadce1a09f55e15`，原文件为 `best.onnx`。模型的单个类别名是 `Smooking`，代码保留该原始拼写。

`download_model.py` 通过 GitHub 官方 API 获取固定版本，检查 Git blob 哈希和预期 SHA-256，并保存来源记录。模型文件约 12 MB，不随本仓库提交，首次使用时下载。

模型 SHA-256：

```text
ddc8cae492f0c28e1ccbadb11560de49b436dd4544f4382a361d3b047f02f38f
```

模型输入是固定大小的 float32 NCHW 图像，输出需要符合 `[1, 4+类别数, 候选框数]` 的 YOLOv8/11 风格。本项目不保证兼容任意 ONNX 文件。

本项目代码采用 [MIT License](LICENSE)。模型作者仓库标注的 MIT 许可证保留在 [third_party/model-LICENSE.txt](third_party/model-LICENSE.txt)。模型、训练数据与基础模型的适用条件参见 [第三方说明](THIRD_PARTY_NOTICES.md)。

## 测试

离线单元测试不要求下载模型，也不连接摄像头：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

覆盖历史框到期、时间倒退、重新检出、分块坐标还原、重复框合并、负类别过滤、补边坐标还原和命令行参数。GitHub Actions 执行相同测试。模型推理的实际验证需另外下载权重并提供本地测试视频；合成测试不等于识别准确率评测。

## 项目结构

```text
app.py                       视频入口、分块检测、预览与短时记忆
detector.py                  ONNX 模型加载、预处理及框解析
download_model.py            固定版本模型下载与哈希校验
requirements.txt             运行依赖
requirements-pycharm.txt     旧版 PyCharm 的可选兼容依赖
run.bat                      Windows 快捷启动
models/README.md             权重位置与来源说明
tests/                       离线测试
.github/workflows/tests.yml  持续集成
```

## 能力范围

程序输出的是疑似抽烟候选，不确认香烟是否点燃或人员是否实际吸烟。模型可能漏掉被遮挡、小尺寸或不同外观的烟，也可能误认水印、手部物品等。应使用有烟、无烟、喝水、吃东西、不同光照和遮挡视频进行场景评估。

当前不跟踪人物身份，不保存告警证据，不录制视频，不自动重连，不发送 HTTP 或 JT808。没有附带任何第三方测试视频、人物截图、现场设备配置或业务资料。
