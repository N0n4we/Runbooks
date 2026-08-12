该项目是runbook集合

## 换源与安装

- 不要假设 `download.pytorch.org` 在所有目标容器都可用；每台新机器部署前先做小文件/目标 wheel 的连通性与下载速度预检，失败或明显卡顿时切换 PyPI 镜像。
- PyTorch 2.8 在阿里源/清华源的默认 PyPI 包通常就是 cu128 build，可用 `torch==2.8.0`（不带 `+cu128`）替代官方 cu128 源；安装后必须验证 `torch.__version__`、`torch.version.cuda`、`torch.cuda.is_available()` 和 GPU 名称。
- pip 使用阿里源时：`-i https://mirrors.aliyun.com/pypi/simple/ --retries 30 --timeout 60`；wheel 下载可能间歇超时，长任务应使用 retry-loop 或可断点/缓存策略。
- `uv pip` 不支持 pip 的 `--retries`、`--timeout` 参数，也不要使用短参数 `-i`；使用完整的 `--index-url` / `--default-index`，并自行用 shell 超时与重试包裹长任务。
- 先装并验证关键的 torch/torchaudio，再安装其余依赖；任何单次下载超过 10 分钟无有效进展应终止并更换源，不要盲等。
