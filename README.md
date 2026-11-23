# LeViT_UNet
For medical image segmentation

1. Our model is based on LeViT ([https://github.com/facebookresearch/LeViT](https://github.com/facebookresearch/LeViT)). You'd better git clone its codes.
Thanks for its great job.

2. There are three models which can be built directly. They are LeViT_UNet_128s, LeViT_UNet_192, and LeViT_UNet_384.
You can build one of the models as it follows:
```python
model = Build_LeViT_UNet_192(num_classes=9, pretrained=True)
model.eval()
output = model(torch.randn(1, 1, 224, 224))
```

3. The processed Synapse dataset can be downloaded in:
[https://drive.google.com/file/d/1-w9q-IjnK28Tvz3ARt6XaSO6fan8PwKQ/view?usp=sharing](https://drive.google.com/file/d/1-w9q-IjnK28Tvz3ARt6XaSO6fan8PwKQ/view?usp=sharing)

If you use this code for your paper, please cite **either the PRCV conference paper or the arXiv preprint**:
```bibtex
@inproceedings{xu2023levitunet,
  author    = {Guoping Xu and
               Xuan Zhang and
               Xinwei He and
               Xinglong Wu},
  title     = {LeViT-UNet: Make Faster Encoders with Transformer for Medical Image Segmentation},
  booktitle = {Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
  pages     = {42--53},
  publisher = {Springer Nature Singapore},
  year      = {2023},
  doi       = {10.1007/978-981-99-8543-2_4},
  url       = {[https://dl.acm.org/doi/abs/10.1007/978-981-99-8543-2_4](https://dl.acm.org/doi/abs/10.1007/978-981-99-8543-2_4)}
}

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
  url       = {[https://arxiv.org/abs/2107.08623](https://arxiv.org/abs/2107.08623)},
  eprinttype = {arXiv},
  eprint    = {2107.08623},
  timestamp = {Thu, 22 Jul 2021 11:14:11 +0200},
  biburl    = {[https://dblp.org/rec/journals/corr/abs-2107-08623.bib](https://dblp.org/rec/journals/corr/abs-2107-08623.bib)},
  bibsource = {dblp computer science bibliography, [https://dblp.org](https://dblp.org)}
}
```

---
## 🚀 Upgrade Version: LeViT‑UNet++
LeViT‑UNet++ is the upgraded successor of LeViT‑UNet. It introduces Haar Wavelet Multi‑Scale Downsampling (HWMD), redesigned lightweight depthwise decoder, and natively supports both 2D slice‑based and 3D volumetric medical segmentation tasks.
- Code repository: [https://github.com/apple1986/levit_unet_pp](https://github.com/apple1986/levit_unet_pp)
- Paper: [https://arxiv.org/abs/2611.22482](https://arxiv.org/abs/2611.22482)
- PDF: [https://arxiv.org/pdf/2611.22482](https://arxiv.org/pdf/2611.22482)

```bibtex
@article{xu2026levitunetpp,
  title={LeViT-UNet++: Towards Balanced Fine-Grained Details and Context for Efficient Medical Image Segmentation},
  author={Guoping Xu, Jinling Zhu, Liuyue Li},
  journal={arXiv preprint arXiv:2611.22482},
  year={2026}
}
```
