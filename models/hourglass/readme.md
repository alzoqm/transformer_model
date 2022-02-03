# Hierarchical Transformers Are More Efficient Language Models
arxiv: https://arxiv.org/pdf/2110.13711v1.pdf <br>
offciial code: https://github.com/google/trax/blob/master/trax/models/research/hourglass.py
# paper review
* 본 논문 리뷰는 직접 읽어보고 이해한 내용을 바탕으로 작성된 내용입니다. 따라서 잘못된 내용이 있을 수 있으니 위의 링크에 있는 논문을 반드시 읽어보기 바랍니다.
* 본 논문 리뷰는 Introduction과 Model에 대해서만 설명하고 있습니다.
## Introduction
* 트랜스포머 모델들은 요약, 언어 모델, 코드 생성 등 다양한 NLP task에서 우수한 성능을 보여 주었습니다. 이는 트랜스포머 모델의 핵심인 attention 메커니즘을 활용해 긴 문장에서도 높은 성능을 보여준다는 것에 있습니다.
* 하지만, 트랜스포머 모델의 핵심인 attention matrix를 계산하기 위해선, 시간적으로도 공간적으로도 O(L^2d)의 비용이 소모되는 단점이 있습니다. 이는 트랜스포머 모델이 더욱더 긴 문장에서의 학습하는 것을 방해하고 있는 요인입니다.
* 본 논문의 저자분들은 위의 문제를 해결하기 위해 트랜스포머 모델에 hierarchical architecture를 도입하였고, 이를 위해 다양한 downsample과 upsample 방법을 도입하여 실험을 진행하였습니다. 그리고 가장 성능이 좋은 모델의 이름을 "Hourglass"라고 이름을 붙였습니다.
## Model
* 본 논문의 저자 분이 직접 설명한 내용이 유튜브에 있습니다. 링크: https://www.youtube.com/watch?v=soqWNyrdjkw
* 위의 유튜브 링크를 통해 제가 이해한 내용입니다.
* 기본적으로 트랜스포머 모델 중 decoder만을 사용하여 학습합니다.
* downsample과 upsample의 횟수는 동일합니다. 가장 앞과 뒤의 레이어들은 2017년 공개된 바닐라 트랜스포머 모델을 그대로 사용합니다.
* 각 트랜스포머 레이어들 중 마지막 레이어만 동일한 크기의 upsample 후의 output과 바로 다음의 트랜스포머 레이어의 output을 residual connection합니다.(그림의 빨간색 점선 참고)
<img width="1031" alt="Hourglass" src="https://user-images.githubusercontent.com/70330480/152101216-e622cae6-c416-468f-9eb1-13a358293d21.png">

### Methods of shortening the input sequence
* shortening은 downsample을 말합니다. 기본적으로 shape이 (L, d)인 tensor가 들어오면 (L/k, d)인 shape으로 만드는 것입니다. k는 shorten factor라는 hyperparameter입니다.
* 기본적인 downsample 방식은 1D average pooling을 활용한 방식입니다. 하지만 본 논문의 저자 분들은 새로운 downsample 방식으로 linear pooling을 제안하고 있습니다.
* linear pooling의 방식의 경우 먼저 shape(L, d)인 tensor를 (L/k, k*d)인 텐서로 reshape합니다. 그 뒤 linear projection을 진행하여 (L/k, d)인 tensor로 변환하는 방식을 사용합니다.
<br><br><img width="479" alt="LinearPooling" src="https://user-images.githubusercontent.com/70330480/152104854-3e86c141-2415-416e-a3d6-4f8f53685374.png">
* 그 뒤 attention 및 residual connection을 진행합니다. 수식은 아래와 같습니다.
* ***output = Downsample(input) + Attention(Q=Downsample(input), K=V=input)***
### Upsampling methods
* upsample 역시 downsample과 방식은 같습니다. 기본적인 방식은 k번 반복해서 upsample을 하는 방식이거나(논문에는 naive upsample이라고 합니다), 위의 linear pooling와 같은 방식으로 linear upsample을 할 수 있습니다.
* 논문의 저자 분들이 또 다른 방식으로 제안한 것은 attention upsampling입니다. 이는 아래의 수식으로 나타내겠습니다.
* ***output = U(residual_input, input) + Attention(Q=U(residual_input, input) K=V=input)***
* 참고로 위의 residual_input은 위에서 설명했다시피 residual connection해서 오는 값입니다. U()는 upsample 함수를 말하며 아래의 수식으로 나타냅니다.
* ***U(residual_input, input) = residual_input + Upsampling(input)***

