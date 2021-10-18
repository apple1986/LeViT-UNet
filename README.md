# LeViT_UNet
For medical image segmentation

1. Our model is based on LeViT (https://github.com/facebookresearch/LeViT). You'd better gitclone its codes.
Thanks for its great job.
2. There are three models which can be build directly. They are LeViT_UNet_128s, LeViT_UNet_192, and LeViT_UNet_384.
You can build one of the models as it follows:

model = Build_LeViT_UNet_192(num_classes=9, pretrained=True)
model.eval()
output = model(torch.randn(1, 1, 224, 224))
