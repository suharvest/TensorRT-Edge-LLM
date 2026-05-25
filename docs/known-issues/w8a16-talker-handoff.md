# Handoff: W8A16 Talker Quantization for Qwen3-TTS-CustomVoice

**Date**: 2026-05-25
**Goal**: 让 Qwen3-TTS-12Hz-0.6B-CustomVoice talker engine 跑 W8A16 量化，省 ~400MB 显存，让 Orin NX 16GB 上 ASR+TTS+LLM 能同驻

## TL;DR

Naive per-output-channel max-abs INT8 量化（脚本 `scripts/quantize_onnx_matmul_w8a16.py`）**完全跑不通** — engine 能 build，但生成 EOS token 永远不被采样，跑到 max_audio_length=4096 帧（实测 511 chunks，4 分钟代替 2 秒短句）。

需要换成 **outlier-aware 量化**（AWQ / SmoothQuant / GPTQ）。

## Why this matters

| 模型 | Engine size | 内存压力 |
|---|---|---|
| FP16 talker (现状) | 863 MB | Orin NX 16GB unified mem 跟 ASR + LLM 同驻时吃紧（前测 phase1 peak 15.1G/15.6G, 96.7%） |
| W8A16 talker (目标) | ~436 MB | 省 ~427MB，关键 |

LLM 路径用的是 Qwen3.5-4B-AWQ（已经 AWQ 量化），不是 naive PTQ —— **这印证了"naive PTQ 不够"是常识**。

## 已尝试的（不要重复）

### 1. Naive `quantize_onnx_matmul_w8a16.py` baseline
```bash
python3 scripts/quantize_onnx_matmul_w8a16.py \
  --input model.onnx --output ... \
  --size-threshold 1024 --check --cast-plugin-inputs-to-fp16
```
- 量化了 197 个 MatMul（887MB → 424MB）
- engine build 成功
- runtime: EOS 失效，跑飞

### 2. Exclude lm_head (codex 推荐 #1)
```bash
... --exclude '^node_linear_196$'
```
- 通过 `find_lm_head2.py` 确认 `node_linear_196` 真是 lm_head (产出 logits output, vocab=3072)
- 量化 196 个（跳 1 个），engine 仍能 build
- runtime: **EOS 仍失效** —— exclude lm_head 不够

### 3. `--cast-plugin-outputs-to-fp32` (codex 推荐 #3)
- TensorRT 报错：plugin output FP32 跟下游 SUM 的 FP16 类型不匹配，**ONNX 解析失败**
- 不可用

### 4. Weight layout check
- 用 `check_w8a16_attrs.py` 验证 plugin nodes：`gemm_n / gemm_k` 一致，`scale_mode=0`（per-output-channel）, `group_size=0`, qweight shape `[K,N]` 跟 plugin 默认 layout=0 匹配
- 不是 layout 问题

## fork 的"成功" W8A16 engine 是 fake history

- `/home/harvest/qwen3-models/engines/orin-nx/highperf/talker_w8a16_outputk/talker_decode_w8a16_outputk.engine` 是 fork 2026-05-10/11 build 的
- 那批 engine 是给 **更老的 Qwen3-TTS 模型**（不是 CustomVoice）
- 而且 fork 自己 doc 也提到反序列化失败：`/Users/harvest/project/qwen3-edgellm-jetson/docs/performance/qwen3-orin-nx-clean-room-2026-05-11.md:226`
- CustomVoice 模型是 2026-05-24 之后才出现，**fork 从没成功量化过 CustomVoice talker**

User 记忆里"highperf 跑通 W8A16"指的是老模型不是 CustomVoice。

## 推荐下一步方案

### Option A: AWQ via ModelOpt (推荐)

NVIDIA modelopt 支持 AWQ for TRT-LLM 系列。spike v0.7.1 已含 `experimental/llm_loader/` 量化入口。

参考：
- `experimental/quantization/quantization_configs.py` 含 `_LM_HEAD_CFG_MAP`，看 LLM 那边 AWQ 配置方式
- `tensorrt_edgellm/quantization/llm_quantization.py` 主入口
- LLM 路径已成功 export Qwen3.5-4B-AWQ，应该可借鉴

Calibration data：用 CustomVoice 训练数据子集（语音 → text → 文本作 calib prompt）

预估：3-5 工程日（含调试）。

### Option B: SmoothQuant

简单些，先把 activation 大值"smooth"到 weight，再普通 PTQ。可以基于现有 `quantize_onnx_matmul_w8a16.py` 改。需要 calibration data 收集 activation statistics。

预估：2-3 工程日。

### Option C: GPTQ

GPTQ 不需要 calibration on input，只用 weight 的 Hessian 信息。但 ONNX 操作复杂度高。

预估：5-7 工程日。

### Option D: 跳过 W8A16（短期）

- 用 FP16 + 别处省内存（如 KV cache 压缩 / 减小 max_seq_len / 拆 ASR+TTS 到不同进程串行跑）
- Trade-off：失去同驻并发能力

## 关键 file:line + paths

### 量化脚本
- `mac:/Users/harvest/project/tensorrt-edge-llm/scripts/quantize_onnx_matmul_w8a16.py`
- Plugin runtime: `cpp/plugins/w8A16LinearPlugin/w8A16LinearPlugin.cpp`
- Plugin kernel: `cpp/kernels/w8A16LinearKernels/w8A16Linear.cu`

### ONNX source (FP16 talker)
- `orin-nx:/home/harvest/edgellm-workspace/qwen3-tts-onnx-minimal-fix/llm/model.onnx + model.onnx.data` (847 MB)
- md5: `7e182cf65b8639e5e10ad2a9f303f0cb`

### 当前 FP16 baseline engine（可用）
- `orin-nx:/home/harvest/qwen3-tts-export-workspace/Qwen3-TTS-12Hz-0.6B-CustomVoice/engines-nx/talker/llm.engine` (863 MB)

### 验证脚本
- `mac:/tmp/find_lm_head2.py` — trace from `logits` output → `node_linear_196` (lm_head)
- `mac:/tmp/check_w8a16_attrs.py` — print plugin attrs / qweight shape

### Smoke command (用 streaming worker)
```bash
fleet exec orin-nx -- "
EDGELLM_PLUGIN_PATH=/home/harvest/spike-v071-nx/build/libNvInfer_edgellm_plugin.so
echo '{\"id\":\"t\",\"text\":\"今天天气真不错\",\"speaker\":\"Vivian\",\"first_chunk_frames\":8}' | \
/home/harvest/spike-v071-nx/build/examples/omni/qwen3_tts_streaming_worker \
  --talkerEngineDir <NEW_W8A16_ENGINE_DIR> \
  --code2wavEngineDir /home/harvest/qwen3-tts-export-workspace/Qwen3-TTS-12Hz-0.6B-CustomVoice/engines-nx/code2wav \
  --tokenizerDir <NEW_W8A16_ENGINE_DIR>"
```

预期成功：
- `done` event total ≤ 5000ms（不是 230000ms）
- chunks ≤ 30（不是 511）
- 生成 wav 时长 ~2-3s（短句）

### 当前 v071 branch HEAD
`v071/customvoice-product` HEAD `11112049` （runtime + worker + CUDA graph capture commits）

## Validation criteria

成功的 W8A16 engine 必须满足：
1. `llm_build` 出 engine（size ~440 MB）
2. smoke 短句 EOS 正常触发（chunks ≤ 30，total ≤ 5s）
3. 音频质量主观可接受（让用户听一下 wav）
4. 跟 FP16 baseline 对比 TTFA：W8A16 应该不慢于 FP16（理论应该快 ~150-200ms）

## 相关 task 状态（已 close）

- Task #13 / #16 / #17 都已标 done，原因：naive PTQ 死路。如果新 agent 接手做 AWQ/SmoothQuant，建议建新 task 跟踪。

## 不要做的

- 不要再试 naive max-abs `quantize_onnx_matmul_w8a16.py` 任何参数组合
- 不要复用 fork 老 W8A16 engine（不同模型，且反序列化已知失败）
- 不要量化 lm_head（已确认是 `node_linear_196`）

## Suggested skills

- **codex:codex-rescue** — 调研 AWQ / SmoothQuant 在 v0.7.1 codebase 的接入点（看 modelopt / quantization configs），出 spec
- **general-purpose** — 实施量化 pipeline（下 calib data → 量化 → build engine → smoke）
- **Explore** — 探索 `experimental/quantization/` 和 `tensorrt_edgellm/quantization/llm_quantization.py` 看 LLM AWQ 路径，借鉴到 TTS

## Context references

- 主对话 session: `/Users/harvest/.claude/projects/-Users-harvest-project-qwen3-edgellm-jetson/534045f9-85a5-4045-ae9e-758b30c53c61.jsonl`
- spike v071 branch: `mac:/Users/harvest/project/tensorrt-edge-llm` @ `v071/customvoice-product`
- TTS 设计 docs: `/Users/harvest/project/qwen3-edgellm-jetson/docs/plans/qwen3-w8a16-tensorcore-design-2026-05-10.md`（fork 当年的 W8A16 设计，部分仍 valid）
