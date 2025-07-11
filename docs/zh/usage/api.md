# API 调用 或 可视化调用

1. 使用python api直接调用：[Python 调用示例](https://github.com/opendatalab/MinerU/blob/master/demo/demo.py)
2. 使用fast api方式调用：
    ```bash
    mineru-api --host 127.0.0.1 --port 8000
    ```
    在浏览器中访问 http://127.0.0.1:8000/docs 查看API文档。

3. 使用gradio webui 或 gradio api调用
    ```bash
    # 使用 pipeline/vlm-transformers/vlm-sglang-client 后端
    mineru-gradio --server-name 127.0.0.1 --server-port 7860
    # 或使用 vlm-sglang-engine/pipeline 后端
    mineru-gradio --server-name 127.0.0.1 --server-port 7860 --enable-sglang-engine true
    ```
    在浏览器中访问 http://127.0.0.1:7860 使用 Gradio WebUI 或访问 http://127.0.0.1:7860/?view=api 使用 Gradio API。

> [!TIP]
> - 以下是一些使用sglang加速模式的建议和注意事项：
> - sglang加速模式目前支持在最低8G显存的Turing架构显卡上运行，但在显存<24G的显卡上可能会遇到显存不足的问题, 可以通过使用以下参数来优化显存使用：
>   - 如果您使用单张显卡遇到显存不足的情况时，可能需要调低KV缓存大小，`--mem-fraction-static 0.5`，如仍出现显存不足问题，可尝试进一步降低到`0.4`或更低。
>   - 如您有两张以上显卡，可尝试通过张量并行（TP）模式简单扩充可用显存：`--tp-size 2`
> - 如果您已经可以正常使用sglang对vlm模型进行加速推理，但仍然希望进一步提升推理速度，可以尝试以下参数：
>   - 如果您有超过多张显卡，可以使用sglang的多卡并行模式来增加吞吐量：`--dp-size 2`
>   - 同时您可以启用`torch.compile`来将推理速度加速约15%：`--enable-torch-compile`
> - 如果您想了解更多有关`sglang`的参数使用方法，请参考 [sglang官方文档](https://docs.sglang.ai/backend/server_arguments.html#common-launch-commands)
> - 所有sglang官方支持的参数都可用通过命令行参数传递给 MinerU，包括以下命令:`mineru`、`mineru-sglang-server`、`mineru-gradio`、`mineru-api`

> [!TIP]
> - 任何情况下，您都可以通过在命令行的开头添加`CUDA_VISIBLE_DEVICES` 环境变量来指定可见的 GPU 设备。例如：
>   ```bash
>   CUDA_VISIBLE_DEVICES=1 mineru -p <input_path> -o <output_path>
>   ```
> - 这种指定方式对所有的命令行调用都有效，包括 `mineru`、`mineru-sglang-server`、`mineru-gradio` 和 `mineru-api`，且对`pipeline`、`vlm`后端均适用。
> - 以下是一些常见的 `CUDA_VISIBLE_DEVICES` 设置示例：
>   ```bash
>   CUDA_VISIBLE_DEVICES=1 Only device 1 will be seen
>   CUDA_VISIBLE_DEVICES=0,1 Devices 0 and 1 will be visible
>   CUDA_VISIBLE_DEVICES=“0,1” Same as above, quotation marks are optional
>   CUDA_VISIBLE_DEVICES=0,2,3 Devices 0, 2, 3 will be visible; device 1 is masked
>   CUDA_VISIBLE_DEVICES="" No GPU will be visible
>   ```
> - 以下是一些可能的使用场景：
>   - 如果您有多张显卡，需要指定卡0和卡1，并使用多卡并行来启动'sglang-server'，可以使用以下命令：
>   ```bash
>   CUDA_VISIBLE_DEVICES=0,1 mineru-sglang-server --port 30000 --dp-size 2
>   ```
>   - 如果您有多张显卡，需要在卡0和卡1上启动两个`fastapi`服务，并分别监听不同的端口，可以使用以下命令：
>   ```bash
>   # 在终端1中
>   CUDA_VISIBLE_DEVICES=0 mineru-api --host 127.0.0.1 --port 8000
>   # 在终端2中
>   CUDA_VISIBLE_DEVICES=1 mineru-api --host 127.0.0.1 --port 8001
>   ```

---
