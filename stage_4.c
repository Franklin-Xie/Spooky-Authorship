from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# instantiates evaluators for different metrics
evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

# model eval: random forest
rf_accuracy = evaluator_accuracy.evaluate(predictions)
rf_precision = evaluator_precision.evaluate(predictions)
rf_recall = evaluator_recall.evaluate(predictions)
rf_f1 = evaluator_f1.evaluate(predictions)

print(f"Random Forest - Accuracy: {rf_accuracy:.2f}, Precision: {rf_precision:.2f}, Recall: {rf_recall:.2f}, F1 Score: {rf_f1:.2f}")

# model eval: logistic regression
lr_accuracy = evaluator_accuracy.evaluate(lr_predictions)
lr_precision = evaluator_precision.evaluate(lr_predictions)
lr_recall = evaluator_recall.evaluate(lr_predictions)
lr_f1 = evaluator_f1.evaluate(lr_predictions)

print(f"Logistic Regression - Accuracy: {lr_accuracy:.2f}, Precision: {lr_precision:.2f}, Recall: {lr_recall:.2f}, F1 Score: {lr_f1:.2f}")

# confusion matrix (logistic regression)
y_true = lr_predictions.select("label").rdd.flatMap(lambda x: x).collect()
y_pred = lr_predictions.select("prediction").rdd.flatMap(lambda x: x).collect()

# spark confusion matrix
confusion_matrix = MulticlassMetrics(lr_predictions.select("prediction", "label").rdd.map(tuple)).confusionMatrix().toArray()
confusion_df = pd.DataFrame(confusion_matrix, index=['Poe', 'Lovecraft', 'Shelley'], columns=['Poe', 'Lovecraft', 'Shelley'])

# confusion matrix visualization
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_df, annot=True, fmt="g", cmap="Blues", cbar=False)
plt.title("Confusion Matrix - Logistic Regression")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()
