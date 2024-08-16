from lib import *
from config import *
from utils import *
from image_transform import ImageTransform
from PIL import Image

class_index = ["ants", "bees"]

class Predictor():
    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, output):  # [0.9, 0.1]
        max_id = np.argmax(output.detach().numpy())
        predicted_label = self.class_index[max_id]
        return predicted_label

predictor = Predictor(class_index)

def predict(img_path):
    # prepare network
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    net.eval()

    # prepare model
    model = load_model(net, save_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # prepare input img
    img = Image.open(img_path).convert('RGB')  # Read image and convert to RGB
    transform = ImageTransform(resize, mean, std)
    img = transform(img, phase="test")
    img = img.unsqueeze_(0)  # (chan, height, width) -> (1, chan, height, width)
    img = img.to(device)

    # predict
    with torch.no_grad():
        output = model(img)
    response = predictor.predict_max(output)

    return response

path = "images.jpg"
res = predict(path)
print(res)

# Epoch 0/5
# val Loss: 0.7702 Acc: 0.4444
# Epoch 1/5
# train Loss: 0.2313 Acc: 0.9136
# val Loss: 0.1658 Acc: 0.9412
# Epoch 2/5
# train Loss: 0.1614 Acc: 0.9547
# val Loss: 0.1439 Acc: 0.9542
# Epoch 3/5
# train Loss: 0.0368 Acc: 0.9835
# val Loss: 0.1479 Acc: 0.9542
# Epoch 4/5
# train Loss: 0.0575 Acc: 0.9835
# val Loss: 0.1288 Acc: 0.9673

