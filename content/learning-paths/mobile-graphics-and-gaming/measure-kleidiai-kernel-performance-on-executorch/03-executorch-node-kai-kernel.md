---
title: KleidiAI micro-kernels support in ExecuTorch
weight: 4

### FIXED, DO NOT MODIFY
layout: learningpathall
---

Only a subset of KleidiAI SME micro-kernels has been integrated into XNNPACK, which serves as the CPU backend for ExecuTorch.

These micro-kernels accelerate operators with specific data types and quantization configurations in ExecuTorch models. Operators not covered by KleidiAI fall back to XNNPACK’s default implementations during inference.

In Excutorch v1.0.0, the supported nodes:

* XNNFullyConnected
* XNNDepthwiseConv2d
* XNNConv2d
* XNNBatchMatrixMultiply

Not all of the nodes listed above are eligible for acceleration by Kleidiai. 

Below is a detailed information of the nodes that can benefit from Kleidiai acceleration.

### XNNFullyConnected 

| XNNPACK GEMM Config | Activations DataType| Weights DataType | Output DataType                      |
| ------------------  | ---------------------------- | --------------------------------------- | ---------------------------- |
| pf16_gemm_config    | FP16                         | FP16                                    | FP16                         |
| pf32_gemm_config    | FP32                         | FP32                                    | FP32                         |
| qp8_f32_qc8w_gemm_config | Asymmetric INT8 quantization | Per-channel symmetric INT8 quantization | FP32                         |
| pqs8_qc8w_gemm_config    | Asymmetric INT8 quantization | Per-channel symmetric INT8 quantization | Asymmetric INT8 quantization |
| qp8_f32_qb4w_gemm_config | FP32                         | Per-channel symmetric INT4 quantization | FP32                         |


### XNNDepthwiseConv2d 
| XNNPACK GEMM Config | Input DataType| Filter DataType | Output DataType                      |
| ------------------  | ---------------------------- | --------------------------------------- | ---------------------------- |
| pqs8_qc8w_gemm_config | Asymmetric INT8 quantization(NHWC) | Per-channel or per-tensor symmetric INT8 quantization | Asymmetric INT8 quantization(NHWC) |

### XNNConv2d
| XNNPACK GEMM Config | Input DataType| Filter DataType | Output DataType                      |
| ------------------  | ---------------------------- | --------------------------------------- | ---------------------------- |
| pf32_gemm_config    | FP32                         | FP32, pointwise (1×1)                   | FP32                         |


### XNNBatchMatrixMultiply
| XNNPACK GEMM Config | Input A DataType| Input B DataType |Output DataType |
| ------------------  | ---------------------------- | --------------------------------------- |--------------------------------------- |
| pf32_gemm_config    | FP32                         | FP32                         | FP32 | 
| pf16_gemm_config    | FP16                         | FP16                         | FP16 |



