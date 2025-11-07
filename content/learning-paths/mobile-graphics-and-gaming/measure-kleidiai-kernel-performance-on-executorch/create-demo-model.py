import torch
import torch.nn as nn
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.partition.config.xnnpack_config import ConfigPrecisionType
from executorch.exir import to_edge_transform_and_lower

from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)


class DemoBatchMatMulModel(nn.Module):
    def forward(self, x,y):
        return torch.bmm(x, y)  

    def get_example_inputs(self,dtype=torch.float32): 
        return (torch.randn(1, 256, 256, dtype=dtype),torch.randn(1, 256, 256, dtype=dtype))

class DemoConv2dModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bitwiseconv = torch.nn.Conv2d(3, 2, 1,groups=1)


    def forward(self,x):
         x = self.bitwiseconv(x)
         return x

    def get_example_inputs(self,dtype=torch.float32): 
        return (torch.randn(1, 3, 16, 16, dtype=dtype),)

class DemoDepthWiseConv2dModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwiseconv = torch.nn.Conv2d(3, 6, 3,groups=3)

    def forward(self,x):
         x = self.depthwiseconv(x)
         return x

    def get_example_inputs(self,dtype=torch.float32):
        return (torch.randn(1, 3, 16, 16, dtype=dtype),)


class DemoConvTranspose2dModel(nn.Module):
    def __init__(self):
        super().__init__()
        # in_channels=1, out_channels=8, kernel_size=3, stride=2 -> 常见配置
        self.deconv = nn.ConvTranspose2d(1, 8, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        return self.deconv(x)

    def get_example_inputs(self,dtype=torch.float32): 
        return (torch.randn(1, 1, 16, 16, dtype=dtype),)

class DemoLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256,256)

    def forward(self, x):
        y = self.linear(x)
        return y

    def get_example_inputs(self,dtype=torch.float32):
        return (torch.randn(1, 256, dtype=dtype),)

def create_demo_model(ModelClass, dtype=torch.float32):
    model = ModelClass().eval().to(dtype)
    inputs = model.get_example_inputs(dtype)
    return model, inputs

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


def export_to_executorch(model, 
                         sample_inputs,
                         pte_file: str, 
                         etr_file: str, 
                         quantize: bool = True,
                         per_channel: bool = False,
                         dynamic: bool=False
                         ):

    exported_program = torch.export.export(model, sample_inputs)
    printmodel = exported_program.module()
    if quantize:
        quantize_model = xnn_quantize(exported_program.module(), 
                                      sample_inputs,
                                      per_channel=per_channel,
                                      dynamic=dynamic)
        printmodel = quantize_model
        exported_program = torch.export.export(quantize_model, sample_inputs)

    output_file = pte_file + ".g"
    with open(output_file, "w") as f:
        f.write(str(printmodel.print_readable()) + "\n\n\n")
        f.write(str(printmodel) + "\n")


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

def export_to_torchscript(ModelClass,pt_file: str):
    model, sample_inputs = create_demo_model(ModelClass)
    traced_model = torch.jit.trace(model, sample_inputs)
    traced_model.save(pt_file)

def export_with_dtype(ModelClass, 
                      pte_file: str, 
                      etr_file: str,
                      dtype=torch.float32, 
                      quantize: bool = True,
                      per_channel: bool = False,
                      dynamic: bool=False
                      ):
    model, sample_inputs = create_demo_model(ModelClass, dtype)
    export_to_executorch(model, sample_inputs, pte_file, etr_file, 
                                      quantize,
                                      per_channel=per_channel,
                                      dynamic=dynamic)

model_configs = [
    (DemoLinearModel,"linear_model"),
    (DemoConv2dModel,"conv2d_model"),
    (DemoDepthWiseConv2dModel,"depthwiseconv2d_model"),
    (DemoBatchMatMulModel,"batchmatmul_model"),
]

for model_class,mode_name in model_configs:
    mode_file_name = "model/" + mode_name

    export_to_torchscript(model_class,f"{mode_file_name}.pt")

    export_with_dtype(model_class,
                      f"{mode_file_name}_f32.pte", 
                      f"{mode_file_name}_f32.etrecord", 
                      quantize=False)

    export_with_dtype(model_class,
                      f"{mode_file_name}_f16.pte",
                      f"{mode_file_name}_f16.etrecord",
                      dtype=torch.float16, 
                      quantize=False)

    quant_configs = [
        ("qint8",  False, False),   # per_channel=False, dynamic=False
        ("qcint8_static", True,  False),  # per_channel=True,  dynamic=False
        ("qcint8_dynamic", True, True) # per_channel=True,  dynamic=True
    ]

    for suffix, per_channel, dynamic in quant_configs:
        export_with_dtype(model_class,
                          f"{mode_file_name}_f32_{suffix}.pte", 
                          f"{mode_file_name}_f32_{suffix}.etrecord", 
                          quantize=True,
                          per_channel=per_channel,
                          dynamic=dynamic)
    
        export_with_dtype(model_class,
                          f"{mode_file_name}_f16_{suffix}.pte", 
                          f"{mode_file_name}_f16_{suffix}.etrecord", 
                          dtype=torch.float16,
                          quantize=True,
                          per_channel=per_channel,
                          dynamic=dynamic)
