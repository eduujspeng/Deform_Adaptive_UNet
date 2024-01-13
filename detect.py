import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from models.Deform_Adaptive_UNet import Deform_Adaptive_UNet


def load_model(model_path):
    model = Deform_Adaptive_UNet(3, 1)  # 模型有3个输入通道和1个输出类别
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location='cpu')
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print('Weights of {} are loaded.'.format(model_path))
    print('%d is not loaded.' % len(no_load_key))

    model.eval()
    return model


def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((192//2, 256//2)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # 增加批处理维度


def postprocess_prediction(prediction):
    prediction = prediction.squeeze().detach().cpu().numpy()
    return prediction


def detect(model, image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)[-1]
    prediction = postprocess_prediction(output)
    return prediction


def main():
    model_path = 'weight/ISIC_DAUnet.pth'
    image_path = 'datasets/ISIC_test/1_img.png'

    model = load_model(model_path)
    prediction = detect(model, image_path)

    plt.imshow(prediction, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
