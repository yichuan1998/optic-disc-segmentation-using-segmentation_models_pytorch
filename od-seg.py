import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from segmentation_models_pytorch.utils.metrics import IoU, Fscore, Accuracy
from PIL import Image
from config import Config


# encoder_names = smp.encoders.get_encoder_names() 
# print(encoder_names)

str_color = {
    'RED' : '\033[91m',
    'GREEN' : '\033[92m',
    'YELLOW' : '\033[93m',
    'BLUE' : '\033[94m',
    'MAGENTA' : '\033[95m',
    'CYAN' : '\033[96m',
    'ENDC' : '\033[0m' 
}


train_transform = A.Compose([
    A.Resize(*Config.img_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(*Config.img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class RetinaDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ids = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_name = self.img_ids[idx]
        mask_name = img_name.replace('.jpg', '_OD.tif')
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path))   
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].float()
            
        return image, mask

# define model 
model = smp.UnetPlusPlus(
    encoder_name=Config.encoder,
    encoder_weights=Config.encoder_weights,
    in_channels=3,
    classes=1,
    activation=Config.activation
).to(Config.device)

# define loss
class DiceBCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        bce = self.bce(pred, target)
        pred = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (pred * target).sum()
        dice = (2.*intersection + smooth)/(pred.sum() + target.sum() + smooth)
        return 0.5*bce + (1 - dice)

train_metrics_func = {     
            'train_iou': IoU(threshold=0.5),     
            'train_fscore': Fscore(threshold=0.5),  
            'train_accuracy': Accuracy(threshold=0.5)}
val_metrics_func = {
            'val_iou': IoU(threshold=0.5),
            'val_fscore': Fscore(threshold=0.5),
            'val_accuracy': Accuracy(threshold=0.5)}

criterion = DiceBCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, verbose=True)


# load data
train_dataset = RetinaDataset(Config.train_path, Config.train_od, transform=train_transform)
val_dataset = RetinaDataset(Config.test_path, Config.test_od, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4, pin_memory=True)


best_iou = 0
history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': [], 'train_fscore': [], 'val_fscore': [], 
           'train_accuracy': [], 'val_accuracy': []}


for epoch in range(Config.epochs):
    model.train()
    epoch_loss = 0
    iou_sum = 0
    train_metrics = {'train_iou': 0, 'train_fscore': 0, 'train_accuracy': 0}
    
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        masks = masks.unsqueeze(1)
        images = images.to(Config.device)
        masks = masks.to(Config.device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        for name, metric in train_metrics_func.items():
            train_metrics[name] += metric(preds, masks).item() * images.size(0)

    
    # validatation
    model.eval()
    val_loss = 0
    val_metrics = {'val_iou': 0, 'val_fscore': 0, 'val_accuracy': 0}
    with torch.no_grad():
        for images, masks in val_loader:
            masks = masks.unsqueeze(1)
            images = images.to(Config.device)
            masks = masks.to(Config.device)
            
            outputs = model(images)
            val_loss += criterion(outputs, masks).item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            for name, metric in val_metrics_func.items():
                val_metrics[name] += metric(preds, masks).item() * images.size(0)
    
    # save metrics
    train_loss = epoch_loss / len(train_dataset)
    val_loss = val_loss / len(val_dataset)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)    

    val_avg_metrics = {k: v / len(val_dataset) for k, v in val_metrics.items()}
    for k in val_avg_metrics:
        history[k].append(val_avg_metrics[k])

    train_avg_metrics = {k: v / len(train_dataset) for k, v in train_metrics.items()}
    for k in train_avg_metrics:
        history[k].append(train_avg_metrics[k])

    
    # save best model
    if val_avg_metrics["val_iou"] > best_iou:
        best_iou = val_avg_metrics['val_iou']
        print(str_color['MAGENTA'] + "best model ever saved!" + str_color['ENDC'])
        torch.save(model.state_dict(), f'best_model_epoch.pth')
    
    # schedule lr
    scheduler.step(val_avg_metrics["val_iou"])
    
    # print 
    print(f"Epoch {epoch+1}/{Config.epochs}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Train IoU: {train_avg_metrics['train_iou']:.4f} | Val IoU: {val_avg_metrics['val_iou']:.4f} | Train F1: {train_avg_metrics['train_fscore']:.4f} | Val F1: {val_avg_metrics['val_fscore']:.4f} | Train Accuracy: {train_avg_metrics['train_accuracy']:.4f} | Val Accuracy: {val_avg_metrics['val_accuracy']:.4f}\n")
    

plt.figure(figsize=(12, 6))
plt.subplot(2,2,1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(2,2,2)
plt.plot(history['train_iou'], label='Train IoU')
plt.plot(history['val_iou'], label='Val IoU')
plt.title('IoU Curve')
plt.legend()

plt.subplot(2,2,3)
plt.plot(history['train_fscore'], label='Train F1')
plt.plot(history['val_fscore'], label='Val F1')
plt.title('F1 Curve')
plt.legend()

plt.subplot(2,2,4)
plt.plot(history['train_accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.savefig('training_metrics.png')
plt.show()
