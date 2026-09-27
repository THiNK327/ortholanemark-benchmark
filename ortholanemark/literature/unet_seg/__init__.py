"""Plain U-Net binary/3-class segmentation baseline (plan §3.3).

Answers the reviewer question "why not just segment?": a standard U-Net
(Ronneberger et al., MICCAI 2015) trained on exactly the same rasterized
polynomial labels and augmentation as the SCNN faithful baseline (the
dataset class is imported from `scnn_faithful`), decoded by the same
row-wise argmax → polynomial-fit adapter. The only differences from
`scnn_faithful` are the architecture (no spatial message passing, no
ImageNet-pretrained VGG backbone) and the existence rule (no dedicated
existence head — existence is derived from accumulated segmentation
support, i.e. the fraction of rows with confident lane pixels).
"""
