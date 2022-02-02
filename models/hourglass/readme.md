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
