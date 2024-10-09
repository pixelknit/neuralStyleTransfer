import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path, size=None):
    image = Image.open(image_path).convert('RGB')
    if size is not None:
        image = image.resize((size, size))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(device)

def rgb_to_yuv(image):
    rgb_to_yuv_matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.14713, -0.28886, 0.436],
        [0.615, -0.51499, -0.10001]
    ]).to(device)
    return torch.matmul(image.permute(0, 2, 3, 1), rgb_to_yuv_matrix.t()).permute(0, 3, 1, 2)

def yuv_to_rgb(image):
    yuv_to_rgb_matrix = torch.tensor([
        [1.0, 0.0, 1.13983],
        [1.0, -0.39465, -0.58060],
        [1.0, 2.03211, 0.0]
    ]).to(device)
    return torch.matmul(image.permute(0, 2, 3, 1), yuv_to_rgb_matrix.t()).permute(0, 3, 1, 2)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

def get_style_model_and_losses(cnn, style_img, content_img,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    cnn = copy.deepcopy(cnn)
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    normalization = normalization.to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)

    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_img.requires_grad_()])

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"run {run[0]}:")
                print(f'Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}')

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

def neural_style_transfer(content_path, style_path, output_path, num_steps=300):
    content_img = load_image(content_path)
    style_img = load_image(style_path, size=content_img.shape[2])
    input_img = content_img.clone()

    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    content_yuv = rgb_to_yuv(content_img)

    output = run_style_transfer(cnn, content_img, style_img, input_img, num_steps=num_steps)

    # Convert output to YUV and preserve original UV channels
    output_yuv = rgb_to_yuv(output)
    output_yuv[:, 1:, :, :] = content_yuv[:, 1:, :, :]

    # Convert back to RGB
    color_preserved_output = yuv_to_rgb(output_yuv)

    output_img = color_preserved_output.cpu().squeeze(0)
    output_img = transforms.ToPILImage()(output_img)
    output_img.save(output_path)
    print(f"Color-preserving style transfer complete. Output saved to {output_path}")

if __name__ == "__main__":
    content_path = "./rockcolor_small.jpg"
    style_path = "./oil.jpg"
    output_path = "./rock_color_result.jpg"

    neural_style_transfer(content_path, style_path, output_path, num_steps=600)
