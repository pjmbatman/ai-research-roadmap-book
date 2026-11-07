import re


def summarize(smi_text, nvcc_text, torch_cuda):
    """nvidia-smi·nvcc 출력과 torch.version.cuda로 CUDA 계층 정보를 정리한다.

    smi_text: nvidia-smi 출력 문자열 또는 None
    nvcc_text: nvcc --version 출력 문자열 또는 None
    torch_cuda: torch.version.cuda 문자열(예 "12.1") 또는 None
    반환: {"driver", "driver_cuda", "toolkit", "torch_ok"} dict
    """
    raise NotImplementedError("세 출력을 파싱하고 torch_ok를 판정하세요")
