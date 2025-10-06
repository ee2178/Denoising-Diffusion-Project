import torch



print("torch version:", torch.__version__)

print("CUDA available:", torch.cuda.is_available())

print("CUDA device count:", torch.cuda.device_count())



if torch.cuda.is_available():

    print("GPU is available and detected by PyTorch.")

    print("GPU name:", torch.cuda.get_device_name(0))

else:

    print("GPU is not available or not detected by PyTorch.")


