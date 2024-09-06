# Softmax Input on Models Trained with One-Hot Encoding

## Overview
This repository explores the effects of using softmax vectors on machine learning models that are typically trained with one-hot encoded vectors. Softmax encoding can offer a smooth, probabilistic alternative to one-hot encoding, but it introduces challenges when applied to certain types of models, particularly tree-based algorithms. This README covers observations and insights for linear regression, random forest, LightGBM, and XGBoost.

## Models and Observations
### 1. Linear Regression
#### Compatibility: Linear regression performs well with softmax vectors, as it treats the input values continuously and does not attempt to binarize them.
#### Outcome: Since there is no inherent thresholding or class separation, softmax encoding works without any major issues.
### 2. Random Forest
#### Inconsistencies: Random forest models treat values above 0.5 as 1, and anything below as 0.
#### Impact: This behavior can introduce inconsistencies for categorical features with high cardinality. For example, the softmax output for such features may not be handled correctly due to this binarization, potentially leading to reduced model performance.
### 3. XGBoost
#### Inconsistencies: XGBoost also follows a similar binarization pattern, treating values >= 0.5 as 1 and anything lower as 0.
#### Impact: This thresholding creates the same issue as random forest for softmax-encoded inputs, leading to a potential loss of information and accuracy, particularly for categorical features with many categories.
### 4. LightGBM
#### Unique Handling: LightGBM is especially problematic for softmax inputs because it treats any value not exactly equal to 0 or 1 as a separate class.
#### Impact: This makes LightGBM unsuitable for models with softmax input, as the intermediate softmax probabilities are seen as distinct, unintended classes rather than variations of 0 and 1.

## Summary of Challenges
When using softmax vectors with tree-based models, we encounter the following issues:

### Random Forest: Binarizes softmax values based on a 0.5 threshold, which can cause inaccuracies for features with many categories.
### XGBoost: Similarly, binarizes values >= 0.5, leading to loss of information.
### LightGBM: Treats non-binary softmax outputs as new classes, making it ineffective for softmax inputs.

## Solution Considerations
For linear models like linear regression, softmax input works seamlessly.
For tree-based models, consider using other strategies like alternative encoding schemes (e.g., target encoding or category embedding) to preserve information for high cardinality features.
