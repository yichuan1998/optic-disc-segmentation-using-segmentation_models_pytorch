import cv2
import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import Config


def predict_single_image(image_path=None, model_path='best_model_epoch.pth', overlay_color=(255, 0, 0), alpha=0.3, Config=Config):
    # load trained model
    model = smp.UnetPlusPlus(
        encoder_name=Config.encoder,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    ).to(Config.device)
    model.load_state_dict(torch.load(model_path, map_location=Config.device))
    model.eval()

    # preprocess
    transform = A.Compose([
        A.Resize(*Config.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # read image
    orig_image = np.array(Image.open(image_path).convert('RGB'))
    h, w = orig_image.shape[:2]  

    # inference on batch images
    # save_path = ""
    # retinal_images_path = ""
    # for i in tqdm(os.listdir(retinal_images_path)):
    #     img = np.array(Image.open(os.path.join(retinal_images_path, i)))
    #     transformed = transform(image=img)
    #     input_tensor = transformed["image"].unsqueeze(0).to(Config.device)
    #     output = model(input_tensor)
    #     pred_mask = (torch.sigmoid(output) > 0.5).squeeze().cpu().numpy().astype(np.uint8)    
    #     #cv2.imwrite(os.path.join(save_path, i), pred_mask*255)   

    with torch.no_grad():
        transformed = transform(image=orig_image)
        input_tensor = transformed["image"].unsqueeze(0).to(Config.device)
        output = model(input_tensor)
        pred_mask = (torch.sigmoid(output) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
    
    resized_mask = pred_mask
    resized_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    overlay = orig_image.copy()
    colored_overlay = np.zeros_like(overlay)
    colored_overlay[resized_mask == 1] = overlay_color  
    cv2.addWeighted(colored_overlay, alpha, overlay, 1 - alpha, 0, overlay)  

    return resized_mask * 255, overlay  

image_path = "" # your retinal image
mask, overlay_img = predict_single_image(image_path)

cv2.imwrite("mask.png", mask)
Image.fromarray(overlay_img).save("overlay.jpg")

fig, ax = plt.subplots(1, 3, figsize=(15,5))
ax[0].imshow(Image.open(image_path))
ax[0].set_title('Original Image')
ax[1].imshow(mask, cmap='gray')
ax[1].set_title('Segmentation Mask')
ax[2].imshow(overlay_img)
ax[2].set_title('Overlay Visualization')
plt.show()
