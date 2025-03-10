import torch 

class Config:
    seed = 42
    img_size = (512, 512)  # adjust base on your dataset and cuda device
    batch_size = 8
    lr = 3e-4
    epochs = 100
    encoder = 'resnet50'  
    encoder_weights = 'imagenet'
    activation = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_path = "/data/stu/dagang/my_datasets/IDRID/1. Original Images/a. Training Set"
    test_path = "/data/stu/dagang/my_datasets/IDRID/1. Original Images/b. Testing Set"
    train_od = "/data/stu/dagang/my_datasets/IDRID/2. All Segmentation Groundtruths/a. Training Set/5. Optic Disc"
    test_od = "/data/stu/dagang/my_datasets/IDRID/2. All Segmentation Groundtruths/b. Testing Set/5. Optic Disc"
