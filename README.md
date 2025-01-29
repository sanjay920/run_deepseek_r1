# DeepSeek-R1 with llama.cpp: Quickstart & Best Practices

Deploy and run **DeepSeek-R1**, a 685B parameter model with dynamic quantization, using [llama.cpp](https://github.com/ggerganov/llama.cpp) on various GPU configurations. This guide covers installation steps, quantization options, hardware recommendations, best practices, **and how to launch an OpenAI-compatible server**.

---

## 1. Introduction

**DeepSeek-R1** is a large language model featuring:

- **685 billion parameters**  
- Multiple **dynamic quantization** options (from 1.58bit up to 2.51bit)  - thanks to [unsloth]([https://huggingface.co/unsloth](https://github.com/unslothai/unsloth))
- Support for both **standard** and **distilled** variants  
- Flexible GPU memory usage for cost-effective deployments  

When combined with **llama.cpp**, DeepSeek-R1 can run on GPUs of various sizes, making it suitable for both **development** and **production** workflows. You can also serve it via an **OpenAI-compatible** REST API to integrate with popular libraries and tools that use OpenAI's API format.

---

## 2. Prerequisites & Requirements

- **Python 3.8 and above+**  
- **pip** installed (for installing `huggingface_hub`)
- **CMake** (for building llama.cpp)
- A **GPU** with sufficient memory (see **Quantization Options** and **Recommended Machines** sections)

> **Note**: Lower-bit quantization (e.g., 1.58bit) uses less GPU memory but sacrifices some quality. Higher-bit quantization (e.g., 2.51bit) provides better quality but requires more GPU memory.

---

## 3. Set Up llama.cpp

Follow these steps to clone and build llama.cpp with CUDA support:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp
    ```

2. **Build with CUDA**:

    ```bash
    cmake -B build -DGGML_CUDA=ON
    GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_F16=1 cmake --build build --config Release
    ```

> **Tip**: Building can take some time. Start downloading the model (next section) in a separate terminal to save time.

---

## 4. Download the Model

1. **Install the Hugging Face CLI**:

    ```bash
    pip install huggingface_hub
    ```

2. **Choose the quantization** level you want (see the table below).  
3. **Download** the model from Hugging Face, ensuring you have enough disk space:

    ```bash
    # 1.58bit Version (~131GB)
    huggingface-cli download unsloth/DeepSeek-R1-GGUF --include "*UD-IQ1_S*" --local-dir DeepSeek-R1-GGUF

    # 1.73bit Version (~158GB)
    huggingface-cli download unsloth/DeepSeek-R1-GGUF --include "*UD-IQ1_M*" --local-dir DeepSeek-R1-GGUF

    # 2.22bit Version (~183GB)
    huggingface-cli download unsloth/DeepSeek-R1-GGUF --include "*UD-IQ2_XXS*" --local-dir DeepSeek-R1-GGUF

    # 2.51bit Version (~212GB)
    huggingface-cli download unsloth/DeepSeek-R1-GGUF --include "*UD-Q2_K_XL*" --local-dir DeepSeek-R1-GGUF
    ```

---

## 5. Quantization Options

DeepSeek-R1 offers four main quantization configurations. Choose one based on your GPU's memory capacity and the quality you need:

| **Quant** | **File Size** | **24GB GPU** | **80GB GPU** | **2×80GB GPU** | **4×80GB GPU** | **8×80GB GPU** | **Type**    | **Quality** | **Down_proj**     |
|-----------|---------------|--------------|--------------|----------------|----------------|----------------|------------|------------|-------------------|
| **1.58bit** | ~131GB      | 7 layers     | 33 layers    | All (61)       | All (61)       | All (61)       | IQ1_S       | Fair       | 2.06/1.56bit      |
| **1.73bit** | ~158GB      | 5 layers     | 26 layers    | 57 layers      | All (61)       | All (61)       | IQ1_M       | Good       | 2.06bit           |
| **2.22bit** | ~183GB      | 4 layers     | 22 layers    | 49 layers      | All (61)       | All (61)       | IQ2_XXS     | Better     | 2.5/2.06bit       |
| **2.51bit** | ~212GB      | 2 layers     | 19 layers    | 32 layers      | 58 layers      | All (61)       | Q2_K_XL     | Best       | 3.5/2.5bit        |

- **File Size**: Approximate download size.  
- **Layers**: How many full layers can fit on a given GPU. If not all layers fit, you must load the remaining layers on additional GPUs or at lower precision.  
- **Quality**: Higher bits yield better performance but require more GPU memory.

---

## 6. Recommended Machines (Lambda Labs or Similar)

Below is a quick reference for popular GPU configurations, their estimated cost (on Lambda Labs), and suggested quantization levels.

| **Configuration**        | **Quantization**           | **Layer Support**                             | **Cost/Hour** | **Best For**                       |
|--------------------------|----------------------------|------------------------------------------------|--------------|------------------------------------|
| **8× H100 SXM (80GB)**   | 2.51bit (Q2_K_XL)          | All layers at highest quality                 | \$23.92       | Large-scale production             |
| **4× H100 SXM (80GB)**   | 2.51bit or 2.22bit         | 58 layers (highest) or all layers (better)    | \$12.36       | Medium-scale production            |
| **2× H100 SXM (80GB)**   | 2.22bit or 1.73bit         | 49-57 layers                                  | \$6.38        | Development/Testing                |
| **1× H100 SXM (80GB)**   | 1.73bit or 1.58bit         | 26-33 layers                                  | \$3.29        | Initial development                |
| **8× A100 SXM (80GB)**   | Various                    | Similar performance to H100 (slightly lower)  | \$14.32       | Cost-optimized production          |
| **4× A100 PCIe (40GB)**  | 1.58bit (IQ1_S)            | Limited layers, best for smaller loads        | \$5.16        | Budget development                 |

> **Not Recommended**:  
>
> - NVIDIA A10 (24GB)  
> - NVIDIA V100 (16GB)  
> - NVIDIA Quadro RTX 6000 (24GB)  
> - Single A100 40GB (for full 33B deployment)

---

## 7. Basic Inference (Interactive Prompt)

Once `llama.cpp` is built and the model is downloaded, you can run a **simple interactive** prompt:

1. **Navigate** to your llama.cpp directory:

    ```bash
    cd /path/to/llama.cpp
    ```

2. **Run** a basic inference command (adjust paths and prompt as needed):

    ```bash
    ./build/bin/llama-cli \
      --model /path/to/DeepSeek-R1-GGUF/your_quantized_model-part-00001-of-0000n.gguf \
      --model ~/central-tx-fs/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL/DeepSeek-R1-UD-Q2_K_XL-00001-of-00005.gguf \
      --cache-type-k q4_0 \
      --threads 16 -no-cnv --n-gpu-layers 8888 --prio 3 \
      --temp 0.6 \
      --ctx-size 8192 \
      --seed 3407 \
      --prompt "what should humans be focused on after agi?"
    ```

3. **Monitor GPU usage** to verify you have enough memory. If you run out, consider:
    - Using a lower-bit quantization  
    - Offloading more layers to CPU

---

## 8. Running an OpenAI-Compatible Server

If you want to use DeepSeek-R1 as a drop-in **OpenAI API** replacement for your applications, you can run an **OpenAI-compliant** HTTP server directly from `llama.cpp`. This lets you connect to the model via the same endpoints and request formats used by the OpenAI Python client and other ecosystem tools.

1. **Launch** the server with the desired parameters:

    ```bash
    ./llama-server \
        -m /path/to/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL-00001-of-00005.gguf \
        --threads 6 \
        --n-gpu-layers 1000 \
        --ctx-size 8192 \
        --host 0.0.0.0 \
        --port 8080 \
        -fa
    ```

   **Key flags** to note:
   - `-m / --model`: Path to your quantized DeepSeek-R1 `.gguf` file  
   - `--threads`: Number of CPU threads for tokenization and scheduling tasks (6 is often sufficient)  
   - `--n-gpu-layers`: Number of layers to run on the GPU (1000 ensures all layers run on GPU)  
   - `--ctx-size`: Sets the maximum context length (in tokens)  
   - `--host` and `--port`: Sets the server's listening address (e.g., `0.0.0.0:8080`)  
   - `-fa`: Force all GPU operations, ensuring maximum GPU utilization

   For additional configuration options, see the [llama.cpp server documentation](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md).

2. **Available Endpoints**:

   The server provides OpenAI-compatible endpoints:
   - `/v1/models`: Get model information
   - `/v1/chat/completions`: Text completions
   - `/v1/completions`: Text completions

3. **Test** your server with an OpenAI-like request. For instance, using `curl`:

    ```bash
    curl http://localhost:8080/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "DeepSeek-R1",
        "prompt": "What is quantum computing?",
        "max_tokens": 128
      }'
    ```

   You should receive a JSON response formatted similarly to the OpenAI API.

4. **Integrate** your application:  
   - **Python**: Set your OpenAI endpoint to `http://<server-ip>:8080/v1`, or override `openai.api_base`.  
   - **Custom Clients**: Use the same request body and headers as you would with the official OpenAI endpoints.

---

## 9. Best Practices & Tips

1. **Distilled vs. Standard**: If using the distilled variant, you may fit more layers in memory.
2. **Batch Processing**: Group inference requests for higher throughput.
3. **Measure & Tune**: Profile GPU memory usage, inference speed, and output quality.

---

## 10. Additional Resources

- **[Unsloth DeepSeek-R1 Model Card](https://huggingface.co/unsloth/DeepSeek-R1-GGUF)**
- **[Official DeepSeek Repository](https://github.com/deepseek-ai/DeepSeek-R1)**
- **[llama.cpp GitHub Repo](https://github.com/ggerganov/llama.cpp)**
- **[Lambda Labs Documentation](https://docs.lambdalabs.com)**

---

### Final Thoughts

DeepSeek-R1's dynamic quantization makes it highly adaptable for different hardware setups—from single-GPU dev boxes to multi-GPU production servers. By picking the right quantization level and hardware configuration, you can strike the perfect balance between **speed**, **quality**, and **cost**.

Now, with an **OpenAI-compatible** server interface, it's even easier to integrate DeepSeek-R1 into applications that already use the OpenAI API format.

Feel free to open up an issue if you have any questions or feedback!
