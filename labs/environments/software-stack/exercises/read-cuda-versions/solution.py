import re


def _ver(s):
    return tuple(int(p) for p in s.split(".")[:2])   # "12.4" -> (12, 4)


def summarize(smi_text, nvcc_text, torch_cuda):
    """nvidia-smi·nvcc 출력과 torch.version.cuda로 CUDA 계층 정보를 정리한다.

    smi_text: nvidia-smi 출력 문자열 또는 None
    nvcc_text: nvcc --version 출력 문자열 또는 None
    torch_cuda: torch.version.cuda 문자열(예 "12.1") 또는 None
    반환: {"driver", "driver_cuda", "toolkit", "torch_ok"} dict
    """
    driver = driver_cuda = toolkit = None
    if smi_text:
        m = re.search(r"Driver Version:\s*([\d.]+)", smi_text)
        driver = m.group(1) if m else None
        m = re.search(r"CUDA Version:\s*([\d.]+)", smi_text)
        driver_cuda = m.group(1) if m else None
    if nvcc_text:
        m = re.search(r"release\s+([\d.]+)", nvcc_text)
        toolkit = m.group(1) if m else None

    if torch_cuda is None:
        torch_ok = None
    elif driver_cuda is None:
        torch_ok = False
    else:
        torch_ok = _ver(torch_cuda) <= _ver(driver_cuda)
    return {"driver": driver, "driver_cuda": driver_cuda, "toolkit": toolkit, "torch_ok": torch_ok}
