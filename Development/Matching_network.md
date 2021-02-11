
---
title: 'Matching Network 알아보기'
date: 2021-02-11 12:00:00
category: 'development'
draft: false
---

# Matching Network 알아보기 🧐
## 0. 포스트 소개
이번 포스트는 matching network 및 전반적인 meta learning에 대해서 정리해보려고 합니다. 기본적으로 N-Shot Learning을 기본으로 하는 모델임으로 관련한 내용 부터 시작하여 논문 'Matching networks for one shot learning'을 톺아보기로 마무리 하기로 하겠습니다.

## 1.  Learning without big-data
이 포스트를 찾아 보고 있는 여러분이라면 딥러닝에 많은 데이터들이 필요하다는 사실은 굳이 다시 상기할 필요가 없을것 이라고 생각하는데요.  기본적인 CNN구조를 사용하여 네트워크를 만들어 냈을 때 이미지 한장을 그럴듯 하게 training하기 위해서는 약 1만장 가량의 이미지가 필요하며(물론 이보다 더 적거나 많을 수 있다) y레이블의 종류가 많은 이미지의 경우 더욱 많은 데이터가 요구됩니다. 

하지만 이러한 데이터의 생성에는 막대한 비용이 들게 마련인데, 이미지 데이터에 대한 라벨링, annotation, 데이터 취득을 위한 사전 실험등에 드는 비용, 데이터를 보관하기 위한 Data disk의 가격 등을 고려한다면 이는 큰 부담일 수 밖에 없습니다. 

따라서 적은 수의 데이터를 통한 학습은 딥러닝 연구자로서 반드시 고민해보고 탐구해보아야 할 영역임은 분명합니다. (물론 적은 수의 데이터로 '학습'은 가능하다, 하지만 여기서 말하고자 하는 바는 '모델 일반화 성능을 보장하는 학습' 임으로 Overfitting 되지 않은 '가치있는' 모델을 만들 수 있는 학습을 가정한다.)

이러한 학습 방법은 다양하게 제안되어 왔는데 아래와 같다고 할 수 있겠습니다. (요즘은 이와 더불어 'Graph Neural Network-based Approach'와 같은 접근도 있지만 여기서는 다루지 않겠습니다)

<img src="https://github.com/Lukious/BlogPostArchive/blob/main/Content/matchingnet/img1.png?raw=true">

거리 기반의 approch인 Matric방식의 러닝 방법이 제안된 이후, Model기반의 a모델이 제안되었고 이후 Opmization 기반의 모델이 제안되어 왔습니다. 이러한 내용에 대한 정리는 [이 블로그](https://talkingaboutme.tistory.com/entry/DL-Meta-Learning-Learning-to-Learn-Fast)에 너무나도 완벽하게 되어 있기 때문에 해당 내용을 참조함이 더욱 좋겠습니다. 

### History of Meta Learning 
메타러닝 논문들 중 중요한 contribution을 한 논문들 만을 모아보면 아래와 같이 정리 해 볼 수 있는데,
- Metric-based Approach
-- Siamese Neural Network for One-Shot Image Recognition (2015)
-- **Matching networks for one shot learning (2016)**
-- Prototypical networks for few-shot learning (2017)
-- Learning to Compare: Relation Network for Few-Shot Learning (2018)

- Model-based Approach
--Meta-Learning with Memory Augmented Neural Networks (2016)
--Meta Networks (2017)

- Optimization-based Approach
-- Optimization as a Model for Few-Shot Learning (2017)
-- Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (2017)

위에서 정리한 표의 'Matching networks for one shot learning'에서(이번에 review할) 제안된 거리기반 모델 아키텍쳐, 메타러닝에 특화된 에피소드 학습, 미니 이미지넷 데이터 베이스 등은 이후에 저술된 모든 논문에 영향을 미쳤으며, 사실상 메타러닝의 뿌리라고 할 수 있는 정말 중요한 논문이라고 할 수 있겠습니다. 자 이제 본격적으로 matching network를설명해 보겠습니다.

## 2.  Meta strategy  for Meta Learning
메타 러닝은 적은 수의 데이터로 데이터를 학습할 수 있도록 하는 모델을 설계하려고 했습니다. 하지만 기존의 학습방식으로는 적은 데이터 세트를 통한 학습을 아무리 반복해 보아도 과적합(over-fitting)을 피해 갈 수 없는 구조 입니다. 이에 메타러닝에 적합한 새로운 학습 구조를 제안하게 되는데 이것이 바로 Episode Training (Meta strategy) 입니다.
- __Batch Training [Simple strategy] (일반적인 학습법)__
-- Limited data에서 일반적인 방법으로 지도학습이 잘 되지 않음

- __Episode Training [Meta strategy] (메타학습법)__
--Training을  할 때, Testing과 유사한 episode 구성 (Overfitting 방지)
--Training set : Support set(S), Batch set(B) 구성
-- N-Way K-Shot 기반

### N-Way K-Shot ?

<img src="https://github.com/Lukious/BlogPostArchive/blob/main/Content/matchingnet/img2.png?raw=true">


'N-shot Learning'에 대해서는 아마 많은 분들이 들어 보였을것 같지만,(메타러닝 대부분 논문 제목부터가 N-Shot learning이죠) 혹시나 'N-Way K-Shot'의 개념이 아직 무엇인지 모르시는 분들은 위의 이미지를 참고해 주시면 될 것 같습니다.

**Way** 는 학습하고자 하는 라벨의 종류 입니다.
**Shot** 은 Episode Training에서 각 라벨이 학습하는 이미지의 수 입니다.

위의 그림에서는 각 레이블마다 4장의 이미지(4-Shot)로 2개의 데이터를 구분하는(2-Way) 모델이라고 할 수 있겠습니다.

추가로 학습에 사용하는(Batching Training에서의 Training Set)을 Episode Training에서는 **Support Set** 이라고 하며, 구분해야 할 이미지(Batching Training에서 predict할 대상)을 **Batch Set**이라고 합니다. 하지만 여기서 주의 해야할 점은 메타러닝의 접근은 **쉽게 학습될 수 있는 모델을 발견** 한다는 점에 있습니다. 즉 특정 주제에 대해서(우리가 보통 모델을 학습시킬 때 data의 특성에 맞는 모델을 설계하죠) 최적화된 모델을 찾는것이 아닌 **적은 데이터셋으로 학습될 수 있는 모델**을 설계 하는것이 메타러닝의 핵심임을 생각해야 합니다. 따라서 Support Set과 Batch Set을 일반적인 Training / Test data set으로 이해하시면 안되고 실질적인 성능평가를 위한 Training / Test data set은 각 N-Way K-Shot을 이루는 Data set 뭉치가 Training/ Test라고 이해를 하시는것이 맞겠습니다. (위의 이미지에서는 좌측의 두 뭉치가 Training Dataset 우측의 뭉치가 Test Data set이라고 할 수 있겠습니다.
   
### Update Rule
이러한 Episode Training 기반의 모델에서의 Update Rule은 아래 이미지와 같습니다.

<img src="https://github.com/Lukious/BlogPostArchive/blob/main/Content/matchingnet/img3.png?raw=true">

N-Way K-Shot에 대한 내용과 더불어 Eposide Traing에 대한 내용 정리했다고 불 수 있는 내용을 수식으로 보면 문제는 더욱 단순해 집니다. 단순하게  해당 확률 값의 log likelihood를 Maximize하게 파라미터를 업데이트 해나가는 과정이라고 정리할 수 있겠습니다.

이러한 메타 러닝의 흐름들을 정리하면 아래의 이미지와 같습니다.

<img src="https://github.com/Lukious/BlogPostArchive/blob/main/Content/matchingnet/img4.png?raw=true">

## 3.  Matching Network
다양한 Meta Learning 논문들 중에서 조금은 올드하다고 할 수 있는 Matching Network를 리뷰하게 된것은 여러 이유가 있는데 그 중 가장큰 이유는 Matching Network가 가장 메타 러닝의 근간이 되며, 가장 준수한 성능을 내고 있기 때문이다. 2016년에 Publish된 논문이 어째서 아직도 준수하다고 (2년이 지나 나온 RelationNet은 해당 모델이 Matching Newwork보다 20% 정도의 성능 향상이 있다고 서술 했다)하는지에 대해 의문을 가질만 합니다.  이에 대한 대답은 **'실제로 메타 러닝의 성능은 최근들어 정체에 이르고 있다'** 라고 주장해 보겠습니다.  근거는 ICLR에서 Publish된 'A closer look at few-shot classification'을 레퍼런스 해보겠습니다.

<img src="https://github.com/Lukious/BlogPostArchive/blob/main/Content/matchingnet/img5.png?raw=true">

해당 논문을 보면 실제 보고된 논문들의 저자 주장과 달리 실제로 메타러닝 방법에 있어 성능의 큰 차이는 없는 것을 확인 할 수 있습니다. 심지어는 Matching Net의 간결한 구조가MAML보다도 더 괜찮은 성능을 보일 때도 있었는데 이러한 경향성을 보면 제안되고 있는 최신 메타러닝 방법 및 전략들이 기대보다는 유효하지 않을 수 있다고 할 수 있겠습니다. 물론 모든 방법들이 유효하지 않다고 주장하는것은 아닙니다. 모델의 퍼포먼스란 하나의 지료만을 가지고 평가 할 수 없는 것이 사실이고 실제로 비교적 최근에 report 된 논문일 수록 미세하지만 조금씩 성능이 향상되고 있는 것은 사실입니다.

어찌되었던 가장 최근에 제안되었던 논문들과 비교해도 손색없을 정도의 성능을 가진 Meta Learning 학습법인 Matching Network는 어떤 방식으로 적은 수의 데이터만을 가지고 Learning을 해내는지에 대해서 톺아보며 알아 보겠습니다.

<img src="https://github.com/Lukious/BlogPostArchive/blob/main/Content/matchingnet/img6.png?raw=true">

위 그림은 Matching Network의 전반적인 구조 입니다. 왼쪽의 데이터들이 Support Set(S) 아래쪽의 이미지 한장이 Batch Set(B) 이라고 할 수 있겠습니다. 해당 모델은 4장의 입력된 견종 사진 레이블 대해서 입력된 B가 어떤 레이블인지를 구분해 내는 4-Way 1-Shot Learning이라고 할 수 있겠습니다.  

그렇다면 이러한 Matching Network의 구조가 어떤 방식으로 정확히 학습이 이루어 지는지 수식과 함께 살펴보겠습니다.

<img src="https://github.com/Lukious/BlogPostArchive/blob/main/Content/matchingnet/img7.gif?raw=true">

우선적으로 Support Set 그룹에서 하나의 Batch를 선택하여 뽑아내어 그 Batch가 어떤 Label일지 확률적으로 표현해냅니다. 이러한 과정은 하나의 Support Set 데이터와 입력된 Batch Set 데이터 간의 Kernel Density Estimator(KDE) 수식의 대입을 통한 두 데이터 간의 metric(거리) 기반 유사도를 통해 평가 되게 됩니다. 즉 정리하자면 두 이미지간의 거리기반 유사도를 구하여 해당 레이블이 어떤 레이블에 할당 될 수 있는지에 대한 확률 값을 계산해 낸다는 내용입니다. 그렇다면 KDE는 어떤 방식으로 두 데이터간의 거리기반 유사도에 대해 알아내는 걸까요?

### Kernel Density Estimator

<img src="https://github.com/Lukious/BlogPostArchive/blob/main/Content/matchingnet/img8.gif?raw=true">

Kernel Density Estimator가 동작하는 방식은 아래의 그림과 같습니다. Support Set과 Batch Set에서 각각 뽑아낸 Feature들에 대해서 Cosine 함수를 통해 이들의 거리 기반 유사도를 이끌어 냅니다. 여기에 softmax를 통해서 KDE를 구해내고 최종적으로 이를 레이블 예측치들의 Attention 합을 나누어 주는 방식을 통해 구해냅니다.

이렇게 구해진 KDE값을 바탕으로 레이블을 예측해 낸다 라는 것이 Matching Network의 핵심이라고 할 수 있겠습니다.


## 4.  Full Context Embedding
해당 논문에서 제안한 Matching Network의 학습 방법은 꽤나 단순해 보이며, 실제도로 간단한 구조입니다. 하지만 이렇게 모델이 단순해 질 수록 발생하는 문제들이 있습니다. 바로 복잡한 데이터 예측에서의 성능이 떨어진다는 점인데요, 이러한 문제는 모델의 일반화 성능을 낮추게되는 매우 critical한 문제임으로 이러한 문제를 해결 하기 위한 방법이 제안되어야 합니다.

이 논문에서는 이러한 문제를 해결하고자 Full Context Embedding을 제안하는데요, 해당 내용은 꼭 meta learning뿐만이 아니라 다양한 분야에서도 활용 되고 있는 내용이니 추가로 알아가시면 좋을것 같습니다.

문제 제기는 이렇게 시작됩니다, 기존의 feature을 extaction해내는 g'()와 같은 함수의 경우 단순히 CNN(Convolution Neural Network)를 통한 추출(extraction)을 진행하였기 때문에 Support Label간의 연관성(dependent)가 부여되지 않는다는 문제가 있습니다 (실제로는 Support Label간의 연관성이 상당히 존재 함에도 불구하고). 따라서 기존에 g'()를 통한 특징점 추출 방법과 더불어 Bidirectional LSTM을 통한 추가적인 임베딩 과정을 통해 Support Label간의 dependent가 반영된 새로운 feature을 extraction될 수 있게 한 것입니다.

<img src="https://github.com/Lukious/BlogPostArchive/blob/main/Content/matchingnet/img9.gif?raw=true">

이러한 방식을 통해 Supprot Label간의 dependent를 부여하였습니다. 하지만 Batch set은 어떨까요? Batch set에게도 Dependent Embedding 된 Support set 정보를 줄 수 있다면 이는 매우 유용하게 사용 될 수 있을 것입니다.

<img src="https://github.com/Lukious/BlogPostArchive/blob/main/Content/matchingnet/img10.gif?raw=true">

이러한 방식은 위의 이미지와 같은 방법을 통해 가능합니다. 우선적으로 Batch Set(S)의 Data는 기존과 같은 CNN을 활용한 방법을 통해 feature extraction을 진행하게 됩니다. 기존에는 이를 그대로 활용하였지만 이번에는 Full context embedding을 위해 이를 LSTM에 입력하겠습니다. 총 K번의(K는 하이퍼파라미터 입니다) Sequence를 지나게 됩니다. 한번의 Sequence에서는 conv로 뽑은 feature와 LSTM으로 생성된 값과의 residual connection을 통해 새로운 feature을 뽑게 되며 이렇게 뽑힌 feature와 support set에서 나온 feature들간 attention 매커니즘을 통해 새로운 feature를 뽑게됩니다(한 스퀸스의 최종). 이렇게 뽑힌 정보를 다음 sequence로 넘기며 총 K번 진행하게 되고 매 sequence마다 attention을 통해 weight가 유용한 feature에 강조될 수 있습니다(다만 K가 너무 높으면 다른 feature들이 아예 무시당할 수도 있겠죠?). 최종적으로 K번의 Sequence이후 K번째 LSTM에서 나온 Hidden state의 input값과 합하여 Batch set에 대한 feature를 뽑아낼 수 있게 될 것입니다.

이들의 Full Context Embedding은 따라서 아래와 같이 정의 되겠습니다.

<img src="https://github.com/Lukious/BlogPostArchive/blob/main/Content/matchingnet/img11.gif?raw=true">


## 5.  마무리...
Matcing Network와 관련된 한국어 자료가 조금 부족한듯 하여 연구실 세미나에서 발표했던 PPT를 바탕으로 조금 덧붙여 보았습니다. 아무래도 100장이 넘는 PPT를 하나의 page로 요약하였기 때문에 생략된 내용도 있지만 그만큼 matching network논문에서 핵심만을 뽑아서 정리해 보았습니다.

혹시나 질문이나 이해가 힘든 내용이 있으시다면 언제든지 댓글 달아주시면 답변 드리겠습니다. 읽어주셔서 감사합니다!