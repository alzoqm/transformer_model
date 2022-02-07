# SwinIR: Image Restoration Using Swin Transformer
paper: https://arxiv.org/abs/2108.10257<br>
official code(pytorch): https://github.com/JingyunLiang/SwinIR

## paper review
* 본 논문 리뷰는 직접 읽어보고 이해한 내용을 바탕으로 작성된 내용입니다. 따라서 잘못된 내용이 있을 수 있으니 위의 링크에 있는 논문을 반드시 읽어보기 바랍니다.
* 본 논문 리뷰는 Introduction과 Model에 대해서만 설명하고 있습니다.
* 본 논문은 computer vision 분야 중 Image Restoration에 관한 논문입니다.
* code의 경우 tensorflow와 keras로 작성되었습니다.
## Introduction
### CNN 연구
* 기존의 Image Restoration(앞으로 IR로 지칭함) 분야는 다른 CV 분야처럼 2010년대 이후부터 CNN 방식의 아키텍처가 높은 성능을 보여주었습니다.
* 하지만 CNN 방식은 2가지의 큰 문제점을 가지고 있습니다.
* 첫번째는, 이미지와 convolution 사이의 상호작용이 이미지 맥락의 관계를 학습하지 못한다는 것입니다.
* 두번째는, CNN 방식은 local processing 원리로 작동하기 때문에, 이미지가 커질경우의 모델에 대해서 효과적이지 않습니다.
### vision transformer 연구
* 이에 대한 대안으로 2020년에 공개된 ViT를 비롯한 vision transformer 모델들이 attention mechanism을 통해 전체 이미지의 관계에 대해 학습할 수 있다는 장점을 바탕으로 많은 연구가 이루어졌습니다.
* 하지만 ViT를 비롯한 vision transformer 모델들도 고정된 크기의 patch로 이미지를 나누어 학습을 하는 방식을 채택하여 IR분야에선 2가지의 문제점을 발생시켰습니다.
* 첫번째는, patch 경계면의 픽셀들은 이웃하지만 patch 밖에 있는 다른 픽셀들에 대한 학습을 온전히 진행할 수 없습니다.
* 두번째는, restored image에 대해서는 각 patch의 경계면이 제대로 복원이 되지 않습니다.(원문: Second, the restored image may introduce border artifacts around each patch. 이 부분은 확실히 해석하기 힘드네요. 틀릴수도 있습니다.)
### swin transformer
* 최근에는 swin transformer 구조는 shifted-window를 활용하여 기존의 cnn 구조와 vision transformer 구조의 장점을 모두 활용하는 구조가 나타났습니다.
* 

![SwinIR_archi](https://user-images.githubusercontent.com/70330480/150893515-284dac75-783f-486a-ad9b-6235735bb8b3.png)
