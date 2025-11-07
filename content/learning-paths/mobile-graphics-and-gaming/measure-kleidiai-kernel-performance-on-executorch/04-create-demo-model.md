---
title: Create and quantize demo models
weight: 5

### FIXED, DO NOT MODIFY
layout: learningpathall
---

To evaluate the per-operator performance of SME2 acceleration, we first need to create a set of simple PyTorch models that can demonstrate which nodes in the computation graph are accelerated by Kleidiai microkernels.

To showcase a wider range of nodes benefiting from Kleidiai acceleration, we also apply lightweight quantization to these models.

Finally, the models are exported into a format compatible with ExecuTorch, enabling end-to-end execution and validation of the accelerated kernels.


### Example PyTorch model

In the following example models, we use simple model to generate nodes that can be accelerated by Kleidiai. 

By adjusting some of the model’s input parameters, we can also simulate the behavior of nodes that appear in real-world models.


#### FullyConnected example model 

```python
import torch
import torch.nn as nn
class DemoLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256,256)

    def forward(self, x):
        y = self.linear(x)
        return y

    def get_example_inputs(self,dtype=torch.float32):
        return (torch.randn(1, 256, dtype=dtype),)

```

#### DepthwiseConv2d example model 

```python
import torch
import torch.nn as nn
class DemoDepthWiseConv2dModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwiseconv = torch.nn.Conv2d(3, 6, 3,groups=3)

    def forward(self,x):
         x = self.depthwiseconv(x)
         return x

    def get_example_inputs(self,dtype=torch.float32):
        return (torch.randn(1, 3, 16, 16, dtype=dtype),)
```

#### Conv2d example model 

```python
import torch
import torch.nn as nn
class DemoConv2dModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bitwiseconv = torch.nn.Conv2d(3, 2, 1,groups=1)

    def forward(self,x):
         x = self.bitwiseconv(x)
         return x

    def get_example_inputs(self,dtype=torch.float32):
        return (torch.randn(1, 3, 16, 16, dtype=dtype),)

```

#### BatchMatrixMultiply example model

```python
import torch
import torch.nn as nn
class DemoBatchMatMulModel(nn.Module):
    def forward(self, x,y):
        return torch.bmm(x, y)

    def get_example_inputs(self,dtype=torch.float32):
        return (torch.randn(1, 256, 256, dtype=dtype),torch.randn(1, 256, 256, dtype=dtype))

```



### Quantize model

we use the XNNPACK quantizer to perform int8 quantization on the model. The corresponding code is shown below:

```python
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

def xnn_quantize(model,
                 example_inputs,
                 per_channel: bool = False,
                 dynamic: bool=False
                 ):
    quantizer = XNNPACKQuantizer()
    operator_config = get_symmetric_quantization_config(
        is_per_channel=per_channel,
        is_dynamic=dynamic
    )

    quantizer.set_global(operator_config)
    m = prepare_pt2e(model, quantizer)
    m(*example_inputs)
    m = convert_pt2e(m)
    return m

```

### Export and lower the model  

Lowering the example model above using the XNNPACK delegate for mobile CPU performance can be done as follows:

```python
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.partition.config.xnnpack_config import ConfigPrecisionType
from executorch.exir import to_edge_transform_and_lower

def export_to_executorch(model,
                         sample_inputs,
                         pte_file: str,
                         etr_file: str,
                         quantize: bool = True,
                         per_channel: bool = False,
                         dynamic: bool=False
                         ):

    exported_program = torch.export.export(model, sample_inputs)
    if quantize:
        quantize_model = xnn_quantize(exported_program.module(),
                                      sample_inputs,
                                      per_channel=per_channel,
                                      dynamic=dynamic)
        exported_program = torch.export.export(quantize_model, sample_inputs)

    partitioner = XnnpackPartitioner()
    edge_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[partitioner],
        generate_etrecord=True
    )

    et_program = edge_program.to_executorch()
    with open(pte_file, "wb") as f:
        f.write(et_program.buffer)

    # Get and save ETRecord
    etrecord = et_program.get_etrecord()
    etrecord.save(etr_file)

```

When exporting the model, we enable the generate_etrecord option. This allows us to generate the model’s etrecord file at the same time, which will later be used for model analysis.

The complete source code is available [here](../create-demo-model.py).

After running this script, both the PTE model file and the etrecord file are generated.

