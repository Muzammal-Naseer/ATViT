# Improving-Adversarial-Transferability-of-Vision-Transformers

[Muzammal Naseer](https://scholar.google.ch/citations?user=tM9xKA8AAAAJ&hl=en),
[Kanchana Ranasinghe](https://scholar.google.com/citations?user=K2WBZTwAAAAJ),
[Salman Khan](https://scholar.google.com/citations?user=M59O9lkAAAAJ&hl=en),
[Fahad Khan](https://scholar.google.ch/citations?user=zvaeYnUAAAAJ&hl=en&oi=ao),
[Fatih Porikli](https://scholar.google.com/citations?user=VpB8NZ8AAAAJ&hl=en)

**[arxiv link](https://arxiv.org/pdf/2106.04169.pdf)** 

![demo](.github/demo.png)

> **Abstract:** 
*Vision transformers (ViTs) process input images as sequences of patches via self-attention;  a radically different architecture than convolutional neural networks(CNNs).  This makes it interesting to study the adversarial feature space of ViTmodels and their transferability. In particular, we observe that adversarial patternsfound via conventional adversarial attacks show verylowblack-box transferabilityeven for large ViT models. However, we show that this phenomenon is only dueto the sub-optimal attack procedures that do not leverage the true representationpotential of ViTs. A deep ViT is composed of multiple blocks, with a consistentarchitecture comprising of self-attention and feed-forward layers, where each blockis capable of independently producing a class token. Formulating an attack usingonly the last class token (conventional approach) does not directly leverage thediscriminative information stored in the earlier tokens, leading to poor adversarialtransferability of ViTs. Using the compositional nature of ViT models, we enhancetransferability  of  existing  attacks  by  introducing  two  novel  strategies  specificto the architecture of ViT models.(i) Self-Ensemble:We propose a method tofind multiple discriminative pathways by dissecting a single ViT model into anensemble of networks. This allows explicitly utilizing class-specific informationat each ViT block.(ii) Token Refinement:We then propose to refine the tokensto further enhance the discriminative capacity at each block of ViT. Our tokenrefinement systematically combines the class tokens with structural informationpreserved within the patch tokens.  An adversarial attack when applied to suchrefined tokens within the ensemble of classifiers found in a single vision transformerhas significantly higher transferability and thereby brings out the true generalizationpotential of the ViT’s adversarial space.* 


### We are in the process of cleaning our code. We will update this repo shortly. Here are the highlights of what to expect :)

1) Code for ensemble model strategy and training methodology
2) Code for token refinement module 
3) Pretrained models and code for evaluating transfer attacks
