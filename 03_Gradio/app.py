import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
from textwrap import wrap

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import cv2

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import timm

import gradio as gr
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class InsectModel(nn.Module):
    def __init__(self,num_classes):
        super(InsectModel, self).__init__()
        self.num_classes = num_classes
        self.model = timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=num_classes)
    def forward(self, image):
        return self.model(image)


class CustomConvNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNet, self).__init__()
        self.num_classes = num_classes
        self.layer1 = self.conv_module(3, 16)
        self.layer2 = self.conv_module(16, 32)
        self.layer3 = self.conv_module(32, 64)
        self.layer4 = self.conv_module(64, 128)
        self.layer5 = self.conv_module(128, 256)
        self.gap = self.global_avg_pool(256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(-1, self.num_classes)

        return out

    def conv_module(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))

num_classes = 5

vit_model = InsectModel(num_classes)
vit_model.load_state_dict(torch.load("./vit-learn_21.pth"))

cnn_model = CustomConvNet(num_classes)
cnn_model.load_state_dict(torch.load('cnn_45.pth'))

input_transform = A.Compose([
    A.Resize(224, 224),
    ToTensorV2()
])


def image_analysis(image, model):
    LABELS = ["정상", "노균병", "노균병 유사", "흰가루병 유사", "흰가루병 유사"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 이미지 전처리
    image = cv2.cvtColor(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = input_transform(image=image)['image']
    image = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

    # 모델 디바이스 이동 및 예측
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(image).softmax(1)[0].cpu().numpy()

    # 결과 반환
    return {LABELS[i]: float(pred[i]) for i in range(len(LABELS))}

def get_model(model_name):
    if model_name == "ViT":
        return vit_model  # 사전에 정의된 PyTorch 모델 객체
    else:
        return cnn_model

# Gradio 인터페이스
demo = gr.Interface(
    fn=lambda image, model_name: image_analysis(image, get_model(model_name)),
    inputs=["image",  gr.Radio(['ViT', "CNN"])],
    outputs="label"
)

demo.launch(share=True)
