## kernel
- 필사 or 공부
- [pseudo-labeling-qda](https://www.kaggle.com/cdeotte/pseudo-labeling-qda-0-969) : 필사 1
  - pseudo labeling을 사용
  - 예측모델 : QDA
- https://www.kaggle.com/christofhenkel/graphicallasso-gaussianmixture
- https://www.kaggle.com/cdeotte/3-clusters-per-class-0-975

## Discussion
- [Summary of Techniques](https://www.kaggle.com/c/instant-gratification/discussion/94573)
  - 이 대회에서 사용된 전반적인 방법론들
  - wheezy-copper-turtle-magic 라는 magic feature 발견
  - feature의 variance가 낮은 경우 불필요한 feature
  - QDA model이 여기서는 best 유용
  - Pseudo Labeling 방법
- [QDA Explained](https://www.kaggle.com/c/instant-gratification/discussion/93843)
- [How to Score LB 0.975](https://www.kaggle.com/c/instant-gratification/discussion/96506)
  - 특이했던 점은 binary classification 인데 이를 비지도학습을 이용하여 6개의 cluster를 만들고 이를 반반씩 합쳐서 classification했다
- [Curse of dimension](https://www.kaggle.com/c/instant-gratification/discussion/93379)
  - [Curse of dimension 참고](https://www.visiondummy.com/2014/04/curse-dimensionality-affect-classification/)
  - 실험내용 : binary classification의 상황, random으로 feature값들을 만든다. data의 수와 feature의 수가 비슷해질수록 완벽하게 분류가 가능하다. 즉, feature가 너무 많으면 overfitting이 심해지는 것이다.
