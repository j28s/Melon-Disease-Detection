import torch

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':50,
    'LEARNING_RATE':2e-5,
    'BATCH_SIZE':8,
    'SEED':42,
    'DEVICE': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}