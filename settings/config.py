import torch


class Config:
    # Training
    seed = 13
    epochs = 120
    learning_rate = 2e-3
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Data preparation
    data_path = "data/Extracted Faces/Extracted Faces/"
    model_path = "models/"
    log_dir = "logs/"
    batch_size = 96
    num_workers = 4

    # Data transformation
    resize_to = (128, 128)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
