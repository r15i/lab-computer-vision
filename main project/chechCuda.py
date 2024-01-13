import torch

try:
    print("Versione PyTorch:", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA e disponibile. Numero di GPU:", torch.cuda.device_count())
        print("GPU corrente:", torch.cuda.get_device_name(0))
    else:
        print("CUDA non e disponibile. Utilizzera la CPU.")
except Exception as e:
    print("Si e verificato un errore durante la verifica di PyTorch:", e)
