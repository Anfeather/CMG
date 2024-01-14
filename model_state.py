from torchstat import stat
from torchvision.models import resnet18
from models import SupResNet, SSLResNet, TextNet, VGG_11, DenseNet121, DenseNet169, DenseNet201, VGG_13,VGG_19, VGG_16
model = VGG_19()
stat(model, (3, 224, 224))