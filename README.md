# LeViT_UNet
For medical image segmentation

1. Our model is based on LeViT (https://github.com/facebookresearch/LeViT). You'd better gitclone its codes.
Thanks for its great job.
2. There are three models which can be build directly. They are LeViT_UNet_128s, LeViT_UNet_192, and LeViT_UNet_384.
You can build one of the models as it follows:

```
model = Build_LeViT_UNet_192(num_classes=9, pretrained=True)

model.eval()

output = model(torch.randn(1, 1, 224, 224))
```

3. The processed Synapse dataset can be downloaded in:
https://drive.google.com/file/d/1-w9q-IjnK28Tvz3ARt6XaSO6fan8PwKQ/view?usp=sharing


If you use this code for your paper, please cite:
```
@article{LeViT-UNet,
  author    = {Guoping Xu and
               Xingrong Wu and
               Xuan Zhang and
               Xinwei He},
  title     = {LeViT-UNet: Make Faster Encoders with Transformer for Medical Image
               Segmentation},
  journal   = {CoRR},
  volume    = {abs/2107.08623},
  year      = {2021},
  url       = {https://arxiv.org/abs/2107.08623},
  eprinttype = {arXiv},
  eprint    = {2107.08623},
  timestamp = {Thu, 22 Jul 2021 11:14:11 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2107-08623.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
