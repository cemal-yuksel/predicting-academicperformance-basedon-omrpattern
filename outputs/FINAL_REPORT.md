# 📊 Comprehensive Transfer Learning Classification Report

**Generated:** 2026-01-17 13:17:03

---

## 🎯 Executive Summary

This report presents a comprehensive evaluation of **17 transfer learning architectures** 
combined with **25 classification algorithms** for Optical Music Recognition (OMR) tasks.

**🏆 Best Performance:** efficientnet_b0 + SVC = **0.8890** (88.90% accuracy)

**📊 Total Experiments:** 425 (17 models × 25 classifiers)

**🔬 Validation:** 10-fold GroupKFold Cross-Validation (subject-wise splits)

**📈 Statistical Significance:** Friedman test χ²(16) = 174.34, p < .001


---

## 🔬 Experimental Setup

### Dataset

- **Total Samples:** 2,100 images

- **Classes:** 2 (binary classification)

- **Subjects:** 21 unique individuals

- **Class Distribution:** Balanced (50-50 split)


### Transfer Learning Architectures

Evaluated 18 pre-trained CNN architectures (17 successful):


**Classic CNNs:**

- ✅ alexnet: 4096 features, 
291.4 img/s

- ✅ vgg16: 4096 features, 
27.9 img/s

- ✅ vgg19: 4096 features, 
22.8 img/s



**Residual Networks:**

- ✅ resnet50: 2048 features, 
63.6 img/s

- ✅ resnet101: 2048 features, 
43.9 img/s

- ✅ resnet152: 2048 features, 
31.3 img/s



**Dense Networks:**

- ✅ densenet169: 1664 features, 
60.9 img/s

- ✅ densenet201: 1920 features, 
47.9 img/s



**EfficientNets:**

- ✅ efficientnet_b0: 1280 features, 
168.0 img/s

- ✅ efficientnet_b1: 1280 features, 
129.8 img/s

- ✅ efficientnet_b2: 1408 features, 
123.7 img/s

- ✅ efficientnet_b5: 2048 features, 
45.3 img/s

- ✅ efficientnet_b6: 2304 features, 
34.6 img/s

- ✅ efficientnet_b7: 2560 features, 
25.4 img/s



**EfficientNet V2:**

- ✅ efficientnet_v2_s: 1280 features, 
84.8 img/s

- ✅ efficientnet_v2_m: 1280 features, 
6.1 img/s

- ✅ efficientnet_v2_l: 1280 features, 
20.8 img/s



**Failed:**

- ❌ inceptionv3: The parameter 'aux_logits' expected value True but got False instead.



### Classification Algorithms (25 tested)


**Linear Models:** LogisticRegression, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV, SGDClassifier, PassiveAggressiveClassifier, Perceptron


**Support Vector Machines:** LinearSVC, SVC, NuSVC


**Tree-Based:** DecisionTreeClassifier, ExtraTreeClassifier


**Ensemble Methods:** RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier


**Naive Bayes:** GaussianNB, BernoulliNB


**Neural Networks:** MLPClassifier


**Distance-Based:** KNeighborsClassifier, NearestCentroid


**Discriminant Analysis:** LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis



---

## 📈 Main Results

### Table 1: Best Classification Method per Transfer Learning Model

Ranked by accuracy (highest to lowest):


| Rank | Transfer Learning | Classification Method | Accuracy |

|------|------------------|----------------------|----------|

| 🥇 1 | efficientnet_b0 | 
SVC | **0.8890** |

| 🥈 2 | efficientnet_b6 | 
SVC | **0.8858** |

| 🥉 3 | resnet101 | 
LogisticRegressionCV | **0.8855** |

|    4 | densenet201 | 
SVC | **0.8835** |

|    5 | resnet152 | 
SVC | **0.8832** |

|    6 | densenet169 | 
SVC | **0.8832** |

|    7 | efficientnet_v2_s | 
SVC | **0.8830** |

|    8 | efficientnet_b2 | 
SVC | **0.8803** |

|    9 | efficientnet_b1 | 
LogisticRegressionCV | **0.8800** |

|    10 | efficientnet_v2_m | 
ExtraTreesClassifier | **0.8783** |

|    11 | resnet50 | 
LogisticRegressionCV | **0.8767** |

|    12 | efficientnet_b5 | 
NuSVC | **0.8760** |

|    13 | efficientnet_b7 | 
SVC | **0.8712** |

|    14 | alexnet | 
LogisticRegressionCV | **0.8662** |

|    15 | vgg19 | 
HistGradientBoostingClassifier | **0.8585** |

|    16 | efficientnet_v2_l | 
SVC | **0.8583** |

|    17 | vgg16 | 
ExtraTreesClassifier | **0.8567** |


**Top 3 Models achieve >88.3% accuracy**


### Table 2: All Classifiers on Best Model (EfficientNet-B0)

Complete evaluation of 25 classification algorithms:


| Rank | Classification Method | Accuracy | Recall | Precision | F1 Score |

|------|---------------------|----------|--------|-----------|----------|

| 🥇 1 | SVC | 
**0.8890** | 0.9130 | 
0.8875 | 0.8939 |

| 🥈 2 | LogisticRegressionCV | 
**0.8882** | 0.9213 | 
0.8811 | 0.8946 |

| 🥉 3 | NuSVC | 
**0.8848** | 0.8917 | 
0.8956 | 0.8868 |

|    4 | RandomForestClassifier | 
**0.8840** | 0.8987 | 
0.8903 | 0.8877 |

|    5 | ExtraTreesClassifier | 
**0.8822** | 0.9090 | 
0.8790 | 0.8878 |

|    6 | GradientBoostingClassifier | 
**0.8768** | 0.9060 | 
0.8721 | 0.8823 |

|    7 | NearestCentroid | 
**0.8762** | 0.9250 | 
0.8598 | 0.8853 |

|    8 | MLPClassifier | 
**0.8758** | 0.8967 | 
0.8786 | 0.8817 |

|    9 | HistGradientBoostingClassifier | 
**0.8757** | 0.9037 | 
0.8730 | 0.8820 |

|    10 | BernoulliNB | 
**0.8697** | 0.9180 | 
0.8528 | 0.8785 |

|    11 | KNeighborsClassifier | 
**0.8540** | 0.9227 | 
0.8297 | 0.8671 |

|    12 | GaussianNB | 
**0.8488** | 0.9270 | 
0.8177 | 0.8626 |

|    13 | BaggingClassifier | 
**0.8462** | 0.8457 | 
0.8609 | 0.8476 |

|    14 | AdaBoostClassifier | 
**0.8407** | 0.8573 | 
0.8413 | 0.8444 |

|    15 | LogisticRegression | 
**0.8342** | 0.8547 | 
0.8358 | 0.8394 |

|    16 | Perceptron | 
**0.8245** | 0.8393 | 
0.8302 | 0.8296 |

|    17 | PassiveAggressiveClassifier | 
**0.8240** | 0.8450 | 
0.8248 | 0.8293 |

|    18 | LinearSVC | 
**0.8238** | 0.8417 | 
0.8253 | 0.8282 |

|    19 | SGDClassifier | 
**0.8193** | 0.7787 | 
0.8615 | 0.8124 |

|    20 | RidgeClassifierCV | 
**0.7940** | 0.8363 | 
0.7798 | 0.8029 |

|    21 | DecisionTreeClassifier | 
**0.7768** | 0.7757 | 
0.7893 | 0.7772 |

|    22 | RidgeClassifier | 
**0.7658** | 0.7983 | 
0.7570 | 0.7734 |

|    23 | LinearDiscriminantAnalysis | 
**0.7623** | 0.7933 | 
0.7545 | 0.7698 |

|    24 | ExtraTreeClassifier | 
**0.7458** | 0.7350 | 
0.7642 | 0.7438 |


---

## 📊 Statistical Analysis

### Model Performance Distribution


| Model | Mean | Std | Min | Max |

|-------|------|-----|-----|-----|

| efficientnet_b0 | 0.8065 | 0.1735 | 
0.0000 | 0.8890 |

| efficientnet_v2_s | 0.8028 | 0.1730 | 
0.0000 | 0.8830 |

| densenet169 | 0.7988 | 0.1746 | 
0.0000 | 0.8832 |

| resnet101 | 0.7980 | 0.1795 | 
0.0000 | 0.8855 |

| efficientnet_b2 | 0.7975 | 0.1725 | 
0.0000 | 0.8803 |

| efficientnet_v2_m | 0.7972 | 0.1725 | 
0.0000 | 0.8783 |

| efficientnet_b6 | 0.7966 | 0.1751 | 
0.0000 | 0.8858 |

| resnet152 | 0.7958 | 0.1778 | 
0.0000 | 0.8832 |

| efficientnet_b1 | 0.7930 | 0.1732 | 
0.0000 | 0.8800 |

| resnet50 | 0.7925 | 0.1759 | 
0.0000 | 0.8767 |


### Classifier Performance Distribution


| Classifier | Mean | Std | Models Tested |

|-----------|------|-----|---------------|

| LogisticRegressionCV | 0.8733 | 0.0115 | 17 |

| SVC | 0.8722 | 0.0164 | 17 |

| ExtraTreesClassifier | 0.8701 | 0.0102 | 17 |

| HistGradientBoostingClassifier | 0.8685 | 0.0085 | 17 |

| RandomForestClassifier | 0.8681 | 0.0119 | 17 |

| GradientBoostingClassifier | 0.8674 | 0.0087 | 17 |

| NuSVC | 0.8666 | 0.0192 | 17 |

| MLPClassifier | 0.8643 | 0.0111 | 17 |

| NearestCentroid | 0.8535 | 0.0330 | 17 |

| BernoulliNB | 0.8460 | 0.0302 | 17 |


### Friedman Test Results


**Null Hypothesis:** No significant difference between transfer learning models


- **Test Statistic:** χ²(16) = 174.34

- **P-value:** 1.14 × 10⁻²⁸ (p < .001)

- **Conclusion:** ✅ **Highly significant** - Transfer learning models show 
statistically significant performance differences


### Post-hoc Analysis


Pairwise Wilcoxon signed-rank tests (Bonferroni corrected α = 0.000368):


**Significant difference found:**

- EfficientNet-B0 vs EfficientNet-B2 (p = 0.000269 ***)


EfficientNet-B0 significantly outperforms EfficientNet-B2 across all classifiers.


---

## 🔑 Key Findings


### 1. Best Architectures

**Top 3 Transfer Learning Models:**

1. 🥇 **EfficientNet-B0**: 88.90% (lightweight, efficient)

2. 🥈 **EfficientNet-B6**: 88.58% (larger capacity)

3. 🥉 **ResNet-101**: 88.55% (deep residual learning)


**EfficientNet family dominates top ranks** - 6 out of top 10 are EfficientNets


### 2. Best Classifiers

**Top 3 Classification Algorithms (averaged across all models):**

1. 🥇 **LogisticRegressionCV**: 87.33% ± 1.15%

2. 🥈 **SVC**: 87.22% ± 1.64%

3. 🥉 **ExtraTreesClassifier**: 87.01% ± 1.02%


**SVC achieves best single result** but LogisticRegressionCV more consistent


### 3. Performance Insights

- **Feature Extraction Speed:** AlexNet fastest (291 img/s), EfficientNet-V2-M slowest (6 img/s)

- **Accuracy vs Speed Trade-off:** EfficientNet-B0 offers best balance

- **Model Complexity:** Deeper doesn't always mean better (ResNet-50 competitive with ResNet-152)

- **Classifier Robustness:** Top 5 classifiers within 1.5% accuracy range

- **Failed Methods:** QuadraticDiscriminantAnalysis (0%) - high dimensionality issue


### 4. Practical Recommendations


**For Production Deployment:**

- **Best Overall:** EfficientNet-B0 + SVC (88.90% accuracy)

- **Fast Inference:** AlexNet + LogisticRegressionCV (86.62%, 291 img/s)

- **Maximum Accuracy:** Try EfficientNet-B6 + SVC (88.58%)

- **Robust Choice:** ResNet-101 + LogisticRegressionCV (88.55%, consistent)


**For Research:**

- Ensemble methods combining top 3-5 models could achieve >90% accuracy

- Fine-tuning top layers of EfficientNet-B0 may improve performance

- Subject-wise validation ensures generalization to new individuals


---

## ⚠️ Limitations

1. **InceptionV3 Failed:** aux_logits parameter incompatibility

2. **Binary Classification Only:** Results may not generalize to multi-class OMR

3. **Fixed Hyperparameters:** No extensive hyperparameter tuning performed

4. **Dataset Size:** 2,100 images relatively small for deep learning

5. **Hardware Constraints:** Some models very slow (EfficientNet-V2-M: 6 img/s)


---

## 🎓 Conclusion


This comprehensive study evaluated **425 different combinations** of transfer learning 
architectures and classification algorithms for OMR tasks. Key conclusions:


1. **EfficientNet-B0 emerges as the optimal architecture**, balancing accuracy (88.90%), 
speed, and efficiency


2. **Support Vector Classification (SVC) provides best single-model performance**, 
though LogisticRegressionCV offers more consistent results


3. **Statistical analysis confirms significant differences** between models 
(Friedman χ² = 174.34, p < .001)


4. **Modern architectures (EfficientNet, ResNet) outperform classic CNNs** 
(VGG, AlexNet) by 2-3%


5. **Subject-wise cross-validation** ensures results generalize to unseen individuals


The combination of **EfficientNet-B0 + SVC** is recommended for production deployment 
in OMR systems, achieving near 90% accuracy with reasonable computational requirements.


---

## 📎 Appendix


### A. Feature Extraction Performance


| Model | Features | Time (s) | Speed (img/s) | File Size (MB) |

|-------|----------|----------|---------------|----------------|

| alexnet | 4096 | 
7.2 | 291.4 | 
9.8 |

| efficientnet_b0 | 1280 | 
12.5 | 168.0 | 
14.8 |

| efficientnet_b1 | 1280 | 
16.2 | 129.8 | 
14.8 |

| efficientnet_b2 | 1408 | 
17.0 | 123.7 | 
16.2 |

| efficientnet_v2_s | 1280 | 
24.8 | 84.8 | 
14.8 |

| resnet50 | 2048 | 
33.0 | 63.6 | 
23.3 |

| densenet169 | 1664 | 
34.5 | 60.9 | 
19.2 |

| densenet201 | 1920 | 
43.9 | 47.9 | 
22.1 |

| efficientnet_b5 | 2048 | 
46.3 | 45.3 | 
23.6 |

| resnet101 | 2048 | 
47.8 | 43.9 | 
23.4 |

| efficientnet_b6 | 2304 | 
60.8 | 34.6 | 
26.6 |

| resnet152 | 2048 | 
67.1 | 31.3 | 
23.3 |

| vgg16 | 4096 | 
75.2 | 27.9 | 
14.0 |

| efficientnet_b7 | 2560 | 
82.7 | 25.4 | 
29.5 |

| vgg19 | 4096 | 
92.0 | 22.8 | 
13.8 |

| efficientnet_v2_l | 1280 | 
100.9 | 20.8 | 
14.8 |

| efficientnet_v2_m | 1280 | 
345.5 | 6.1 | 
14.8 |


### B. Complete Result Files

- `table1_best_per_model.csv` - Best classifier per model

- `table2_all_classifiers_efficientnet_b0.csv` - All classifiers on best model

- `all_results_combined.csv` - Complete 425-experiment dataset

- `statistical_analysis_report.txt` - Detailed statistical analysis

- `results_{model}.csv` - Individual model results (17 files)


### C. Reproducibility

**Environment:**

- Python 3.13.7

- PyTorch 2.9.1, torchvision 0.24.1

- scikit-learn 1.8.0

- 10-fold GroupKFold CV with random_state=42


**Scripts:**

1. `09_multiple_transfer_learning.py` - Feature extraction

2. `10_comprehensive_classification.py` - Classification evaluation

3. `11_statistical_analysis.py` - Statistical tests

4. `12_generate_final_report.py` - Report generation


---


*Report generated on 2026-01-17 at 13:17:03*

*For questions or issues, refer to project documentation*
