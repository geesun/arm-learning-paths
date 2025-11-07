---
title: How to Measure Kleidai Kernel Performance in ExecuTorch

minutes_to_complete: 30

who_is_this_for: This article is intended for advanced developers who want to leverage Kleidai to accelerate ExecuTorch model inference on the AArch64 platform.

learning_objectives: 
    - Cross-compile ExecuTorch for the ARM64 platform, enabling XNNPACK and KleidiAI with SME2 support. 
    - Create ExecuTorch models that can be accelerated by SME2 through KleidiAI. 
    - Use the executor_runner tool to generate ETDump profiling data. 
    - Analyze the contents of ETRecord and ETDump using the ExecuTorch Inspector API. 

prerequisites:
    - An x86_64 Linux machine running Ubuntu with approximately 15GB of free space.
    - An Arm64 system with support SME/SME2. 

author: Qixiang Xu

### Tags
skilllevels: Advanced
subjects: ML
armips:
    - Cortex-A
    - SME
    - Kleidai

tools_software_languages:
    - Python
    - cmake
    - XNNPACK

operatingsystems:
    - Linux


further_reading:
    - resource:
        title: Executorch User Guide 
        link: https://docs.pytorch.org/executorch/stable/intro-section.html
        type: documentation



### FIXED, DO NOT MODIFY
# ================================================================================
weight: 1                       # _index.md always has weight of 1 to order correctly
layout: "learningpathall"       # All files under learning paths have this same wrapper
learning_path_main_page: "yes"  # This should be surfaced when looking for related content. Only set for _index.md of learning path content.
---
