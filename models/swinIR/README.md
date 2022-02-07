# SwinIR: Image Restoration Using Swin Transformer
paper: https://arxiv.org/abs/2108.10257<br>
official code(pytorch): https://github.com/JingyunLiang/SwinIR

## paper review
* 본 논문 리뷰는 직접 읽어보고 이해한 내용을 바탕으로 작성된 내용입니다. 따라서 잘못된 내용이 있을 수 있으니 위의 링크에 있는 논문을 반드시 읽어보기 바랍니다.
* 본 논문 리뷰는 Introduction과 Method에 대해서만 설명하고 있습니다.
* 본 논문은 computer vision 분야 중 Image Restoration에 관한 논문입니다.
* code의 경우 tensorflow와 keras로 작성되었습니다.
# Introduction
### CNN 연구
* 기존의 Image Restoration(앞으로 IR로 지칭함) 분야는 다른 CV 분야처럼 2010년대 이후부터 CNN 방식의 아키텍처가 높은 성능을 보여주었습니다.
* 하지만 CNN 방식은 2가지의 큰 문제점을 가지고 있습니다.
* 첫번째는, 이미지와 convolution 사이의 상호작용이 이미지 맥락의 관계를 학습하지 못한다는 것입니다.
* 두번째는, CNN 방식은 local processing 원리로 작동하기 때문에, 이미지가 커질경우의 모델에 대해서 효과적이지 않습니다.
### Vision Transformer 연구
* 이에 대한 대안으로 2020년에 공개된 ViT를 비롯한 Vision Transformer 모델들이 attention mechanism을 통해 전체 이미지의 관계에 대해 학습할 수 있다는 장점을 바탕으로 많은 연구가 이루어졌습니다.
* 하지만 ViT를 비롯한 vision transformer 모델들도 고정된 크기의 patch로 이미지를 나누어 학습을 하는 방식을 채택하여 IR분야에선 2가지의 문제점을 발생시켰습니다.
* 첫번째는, patch 경계면의 픽셀들은 이웃하지만 patch 밖에 있는 다른 픽셀들에 대한 학습을 온전히 진행할 수 없습니다.
* 두번째는, restored image에 대해서는 각 patch의 경계면이 제대로 복원이 되지 않습니다.(원문: Second, the restored image may introduce border artifacts around each patch. 이 부분은 확실히 해석하기 힘드네요. 틀릴수도 있습니다.)
### Swin Transformer
* 최근에는 Swin Transformer 구조는 shifted-window를 활용하여 기존의 CNN 구조와 Vision Transformer 구조의 장점을 모두 활용하는 구조가 나타났습니다.
* 논문의 이름에서 알 수 있듯이 본 논문은 swin transformer를 활용하여 Image Restoration을 진행합니다.
* 구조는 크게 shallow feature extraction, deep feature extraction,  high-quality image reconstruction 3가지로 구분되며, Swin Transformer는 deep feature extraction module에서 활용됩니다.

# Method
![SwinIR_archi](https://user-images.githubusercontent.com/70330480/150893515-284dac75-783f-486a-ad9b-6235735bb8b3.png) <br>
### <center>SwinIR 구조 이미지</center>

## Shallow feature extraction
* input이 들어오면 가장 먼저 shallow feature extraction을 통과합니다.
* 들어온 input shape이 H x W x C<sub>input_C</sub>이며, 이를 3 x 3 크기의 kernel size의 CNN을 통과합니다.
* 통과한 output shape은 H x W x C 이며, C는 feature channels number라는 하이퍼 파라메터입니다.(구현 코드에는 'emb_size'라고 되어있음)
* shallow feature extraction을 통과한 tensor는 deep feature extraction을 통과합니다.
## deep feature extraction
* deep feature extraction는 RSTB(Residual Swin Transformer Block)들로 이루어져 있으며 마지막 레이어는 Convolution을 통과합니다.
* RSTB를 살펴보면 각 RSTB는 swin transformer 레이어를 짝수의 수 만큼 가지고 있습니다. 짝수개로 가지고 있는 이유는 swin transformer 자체의 구조가 shifted window를 하지 않은 것, 한 것 2개를 사용해야하기 때문입니다.
* RSTB 각 블럭의 마지막 레이어도 Convolution을 지나며 모든 레이어를 다 지난 output과 입력 받은 input을 residual connection합니다.(이미지 참조)
* 본 논문에서는 각 RSTB 블럭의 Convolution 레이어와 deep feature extraction의 마지막 Convolution 레이어는 1개의 CNN 층을 지나는 버전과 3개의 CNN 층을 지나는 버전을 제시하였습니다.(code 참조)
* 3개의 층을 지나는 버전을 사용하면, 성능은 최대한 보전하면서 파라메터 수를 줄일 수 있습니다.
* 여담으로 공식 github 코드의 질문을 살펴보면 absolute positional encoding을 사용하지 않은 경우가 사용한 경우보다 성능이 더 좋게 나왔다고 합니다.
## high-quality image reconstruction
* high-quality image reconstruction을 통과하기 앞서 deep feature extraction을 통과하기 이전의 input과 통과한 이후의 output을 residual connection합니다.(이미지 참조)
* 본 논문에는 high-quality image reconstruction의 레이어를 구성하는 방식도 여러가지를 소개하였지만, 제가 구현한 코드에는 2가지만 구현하였으며, 나머지는 위의 github 공식 code에서 확인할 수 있습니다.
## Loss function
* Loss function에도 2가지의 버전을 제시하였으며, IR task에 따라 분류됩니다.
* Super Resolution(SR) task에선 L1 Loss를 사용하며, 나머지 분야에서 L2 Loss에 ε<sup>2</sup>을 더한 값을 사용합니다.(ε = 10<sup>-3</sup>)
