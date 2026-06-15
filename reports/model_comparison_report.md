# BrandPulse AI — Model Comparison Report

## Dataset
- Total tweets: 1,592,958
- Train set: 1,274,366
- Test set: 318,592
- Classes: Balanced (50% positive, 50% negative)

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 80.58% | 0.7968 | 0.8209 | 0.8087 |
| Naive Bayes | 78.60% | 0.7887 | 0.7812 | 0.7849 |
| LSTM (BiLSTM) | 81.04% | 0.8130 | 0.8062 | 0.8096 |

## Key Findings
- LSTM outperforms classical models by learning contextual patterns
- Logistic Regression is fastest and most interpretable
- Naive Bayes is lightest but least accurate
- All models struggle with sarcasm and negation

## Figures
- reports/figures/model_comparison.png
- reports/figures/all_confusion_matrices.png
- reports/figures/lstm_training_curves.png
