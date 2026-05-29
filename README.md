# 02_telco_customer_churn

Учебный ML-проект по бинарной классификации оттока клиентов.

Dataset: Telco Customer Churn  
Target: Churn

## Learning goals

- binary classification
- categorical feature handling
- ColumnTransformer
- OneHotEncoder
- SimpleImputer
- Pipeline
- train/test split without leakage
- churn-specific metrics


Принято. Статус проекта 02_telco_customer_churn зафиксирован повторно.

Project status

Завершены и приняты:

Stage 0 — Dataset audit and task framing
Stage 1 — Initial EDA
Stage 2 — Train/test split and preprocessing plan
Stage 3 — Baseline classification models
Stage 4 — Controlled model comparison
Stage 5 — Controlled threshold and candidate refinement
Текущие кандидаты

Лучший при threshold 0.5:

LogisticRegression(class_weight="balanced")

Метрики:

F1        ≈ 0.6280
recall    ≈ 0.8013
precision ≈ 0.5164

Лучшая OOF-F1 threshold policy:

LogisticRegression(class_weight="balanced")
threshold ≈ 0.60

Метрики:

F1        ≈ 0.6367
recall    ≈ 0.7191
precision ≈ 0.5712

Лучший ranking candidate:

GradientBoostingClassifier
ROC-AUC ≈ 0.8471
PR-AUC  ≈ 0.6643
Пауза

Проект стоит на паузе после Stage 5.

Не делаем:

Stage 6
final model selection
hyperparameter tuning
final test evaluation
evaluation on X_test / y_test

X_test / y_test остаются нетронутыми до отдельного approval координатора.

Coordinator approved starting Round 5 for 02_telco_customer_churn.

Proceed to Stage 6 — Final evaluation on held-out test set.

Project framing:

task type: binary classification
target: Churn
positive class: Yes / 1
final frozen candidate: LogisticRegression(class_weight="balanced")
final frozen threshold policy: threshold ≈ 0.60
X_test has never been evaluated before
this is the first and only final test evaluation
no retuning after seeing test results
no threshold change after seeing test results

Required notebook:
notebooks/06_final_evaluation.ipynb

Stage 6 requirements:

Reuse the same data preparation logic:
y: Churn No -> 0, Yes -> 1
X: drop customerID and Churn
TotalCharges -> numeric with errors="coerce"
Reuse the same train/test split:
test_size=0.2
random_state=42
stratify=y

Build the final pipeline:
Numeric pipeline:

SimpleImputer(strategy="median")
StandardScaler()

Categorical pipeline:

SimpleImputer(strategy="most_frequent")
OneHotEncoder(handle_unknown="ignore")

Model:

LogisticRegression(class_weight="balanced")
Fit the final pipeline only on X_train / y_train.
Predict probabilities on X_test exactly once.
Apply the frozen threshold:
y_pred = 1 if probability >= 0.60 else 0
Report final test metrics:
accuracy
precision
recall
F1
ROC-AUC
PR-AUC / average precision
confusion matrix
classification report
Compare final test metrics with Stage 5 OOF policy:
Stage 5 OOF policy:
model: LogisticRegression(class_weight="balanced")
threshold ≈ 0.60
F1 ≈ 0.6367
recall ≈ 0.7191
precision ≈ 0.5712
Do not change threshold after seeing test results.
Do not change model parameters after seeing test results.
Do not try alternative models on X_test.
Do not save final model yet unless coordinator approves after reviewing test results.
Write final evaluation conclusions:
did test metrics align with OOF expectations?
did recall/precision trade-off behave as expected?
signs of overfitting or instability?
final honest model quality;
limitations of the threshold policy.

At the end, provide STAGE REPORT for coordinator with:

final candidate;
threshold policy;
train/test split confirmation;
test metrics;
confusion matrix;
comparison with Stage 5 OOF metrics;
leakage checks;
Git status;
latest commit.

Принято. Это Stage 6: первый и единственный выход на held-out test set. После метрик ничего не меняем — ни threshold, ни модель, ни параметры.

Думал на протяжении нескольких секунд
1. Что делаем на текущем шаге

Начинаем Stage 6 — Final evaluation on held-out test set.

Финальный кандидат уже заморожен:

LogisticRegression(class_weight="balanced")

Финальная threshold policy уже заморожена:

threshold = 0.60

На этом этапе мы:

создаём notebooks/06_final_evaluation.ipynb;
воспроизводим ту же подготовку данных;
воспроизводим тот же train/test split;
строим final pipeline;
fit только на X_train / y_train;
один раз получаем probabilities на X_test;
применяем threshold 0.60;
считаем финальные test metrics;
пишем выводы.
2. Зачем

До этого X_test / y_test были заморожены и не использовались. Теперь координатор разрешил финальную оценку.

Это значит:

X_test теперь используется как held-out final evaluation set.

Но правила строгие:

После просмотра test results нельзя менять threshold.
После просмотра test results нельзя менять model parameters.
После просмотра test results нельзя пробовать другие модели на X_test.

Иначе финальный test set превратится в validation set, а оценка перестанет быть честной.

3. Команды или код

В PowerShell из корня проекта:

cd C:\temp\python_learning\ml_projects\ml_projects_batch_01\02_telco_customer_churn
.venv\Scripts\Activate.ps1
New-Item notebooks\06_final_evaluation.ipynb

Открой notebook и выбери kernel:

Python (02_telco_customer_churn)
Ячейка 1 — environment check
import sys
from pathlib import Path

print(sys.executable)
print(Path.cwd())
Ячейка 2 — imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
Ячейка 3 — load raw dataset
PROJECT_ROOT = Path.cwd().parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

df.shape
Ячейка 4 — create target and features
target_col = "Churn"
id_cols = ["customerID"]

y = df[target_col].map({
    "No": 0,
    "Yes": 1,
})

X = df.drop(columns=id_cols + [target_col])
X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

X.shape, y.shape

Проверки:

assert y.isna().sum() == 0
assert set(y.unique()) == {0, 1}
assert "customerID" not in X.columns
assert "Churn" not in X.columns
Ячейка 5 — feature lists
numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

all_features = numeric_features + categorical_features

len(numeric_features), len(categorical_features), len(all_features), X.shape[1]

Проверка:

assert set(all_features) == set(X.columns)
Ячейка 6 — same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

Ожидаемо:

((5634, 19), (1409, 19), (5634,), (1409,))
Ячейка 7 — split distribution confirmation
split_distribution = pd.DataFrame({
    "full": y.value_counts(normalize=True).sort_index(),
    "train": y_train.value_counts(normalize=True).sort_index(),
    "test": y_test.value_counts(normalize=True).sort_index(),
})

split_distribution

Абсолютные значения:

split_counts = pd.DataFrame({
    "full": y.value_counts().sort_index(),
    "train": y_train.value_counts().sort_index(),
    "test": y_test.value_counts().sort_index(),
})

split_counts
Ячейка 8 — final pipeline
numeric_preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocessor, numeric_features),
        ("cat", categorical_preprocessor, categorical_features),
    ]
)

final_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ]
)

final_pipeline
Ячейка 9 — fit only on train
final_pipeline.fit(X_train, y_train)

Важно: это единственный fit, и он только на X_train / y_train.

Ячейка 10 — predict probabilities on X_test exactly once
y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]

y_test_proba[:10]
Ячейка 11 — apply frozen threshold
final_threshold = 0.60

y_test_pred = (y_test_proba >= final_threshold).astype(int)

pd.Series(y_test_pred).value_counts().sort_index()
Ячейка 12 — final test metrics
final_test_metrics = {
    "threshold": final_threshold,
    "accuracy": accuracy_score(y_test, y_test_pred),
    "precision": precision_score(y_test, y_test_pred, zero_division=0),
    "recall": recall_score(y_test, y_test_pred, zero_division=0),
    "f1": f1_score(y_test, y_test_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_test_proba),
    "pr_auc": average_precision_score(y_test, y_test_proba),
}

final_test_metrics_df = pd.DataFrame([final_test_metrics])

final_test_metrics_df.round(4)
Ячейка 13 — confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

cm

Более читаемо:

tn, fp, fn, tp = cm.ravel()

confusion_matrix_df = pd.DataFrame(
    {
        "predicted_0": [tn, fn],
        "predicted_1": [fp, tp],
    },
    index=["actual_0", "actual_1"]
)

confusion_matrix_df
Ячейка 14 — classification report
print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=["No churn", "Churn"],
        digits=4,
        zero_division=0,
    )
)
Ячейка 15 — compare with Stage 5 OOF policy
stage5_oof_metrics = {
    "source": "Stage 5 OOF",
    "model": "LogisticRegression(class_weight='balanced')",
    "threshold": 0.60,
    "precision": 0.5712,
    "recall": 0.7191,
    "f1": 0.6367,
}

stage6_test_metrics = {
    "source": "Stage 6 test",
    "model": "LogisticRegression(class_weight='balanced')",
    "threshold": final_threshold,
    "precision": final_test_metrics["precision"],
    "recall": final_test_metrics["recall"],
    "f1": final_test_metrics["f1"],
}

oof_vs_test = pd.DataFrame([
    stage5_oof_metrics,
    stage6_test_metrics,
])

oof_vs_test.round(4)
Ячейка 16 — final evaluation notes

Пока оставь markdown-шаблон. После получения test metrics мы его заполним честно по факту:

## Stage 6 final evaluation conclusions

### Final frozen candidate

- Model: `LogisticRegression(class_weight="balanced")`
- Threshold policy: predict `Churn = 1` if predicted probability is at least `0.60`.

### Evaluation protocol

- The final pipeline was fitted only on `X_train` / `y_train`.
- `X_test` / `y_test` were evaluated once for final held-out test evaluation.
- No model parameters were changed after seeing test results.
- The threshold was not changed after seeing test results.
- No alternative models were evaluated on `X_test`.

### Comparison with Stage 5 OOF expectations

To be filled after inspecting final test metrics.

### Final interpretation

To be filled after inspecting final test metrics.

### Limitations

- The threshold policy was selected using train-only OOF predictions, so it may not transfer perfectly to the held-out test set.
- This is a small tabular dataset, so metric variation is expected.
- The final threshold reflects an F1-oriented trade-off, not a fully cost-optimized business policy.
4. Что должно получиться

Ты должен получить:

финальные test metrics;
confusion matrix;
classification report;
сравнение Stage 5 OOF vs Stage 6 test.

Ожидание по смыслу:

test metrics должны быть близки к OOF, но не обязаны совпасть.

Если test F1 / recall / precision немного отличаются — это нормально.

5. Что мне прислать для проверки

Пришли:

final_test_metrics_df.round(4)
confusion_matrix_df
classification_report(...)
oof_vs_test.round(4)

И отдельно напиши:

Threshold changed after test results: no
Alternative models evaluated on X_test: no
Model parameters changed after test results: no

После этого я помогу сформулировать финальные conclusions и STAGE REPORT.

6. Контрольный вопрос

Почему после просмотра X_test-метрик нельзя сказать: “threshold 0.55 выглядит лучше, давай поменяем на него”?

threshold	accuracy	precision	recall	f1	roc_auc	pr_auc
0	0.6	0.7622	0.5399	0.7059	0.6118	0.8413	0.6326
	predicted_0	predicted_1 
actual_0	810	225
actual_1	110	264
precision    recall  f1-score   support 

    No churn     0.8804    0.7826    0.8286      1035
       Churn     0.5399    0.7059    0.6118       374

    accuracy                         0.7622      1409
   macro avg     0.7102    0.7442    0.7202      1409
weighted avg     0.7900    0.7622    0.7711      1409
	source	model	threshold	precision	recall	f1 
0	Stage 5 OOF	LogisticRegression(class_weight='balanced')	0.6	0.5712	0.7191	0.6367
1	Stage 6 test	LogisticRegression(class_weight='balanced')	0.6	0.5399	0.7059	0.6118
1. Что делаем на текущем шаге

Final evaluation выполнена корректно.

Финальный frozen candidate:

LogisticRegression(class_weight="balanced")

Frozen threshold policy:

threshold = 0.60

Test set был использован один раз для финальной оценки. Threshold и параметры после просмотра test results не меняем.

2. Финальные test metrics
accuracy  ≈ 0.7622
precision ≈ 0.5399
recall    ≈ 0.7059
F1        ≈ 0.6118
ROC-AUC   ≈ 0.8413
PR-AUC    ≈ 0.6326

Confusion matrix:

                 predicted_0   predicted_1
actual_0              810           225
actual_1              110           264

Интерпретация:

TN = 810  — правильно нашли No churn
FP = 225  — ошибочно пометили как churn
FN = 110  — пропустили churn
TP = 264  — правильно нашли churn

По churn-классу:

precision ≈ 0.5399
recall    ≈ 0.7059
F1        ≈ 0.6118
support   = 374

То есть модель поймала примерно:

264 из 374 churn-клиентов

И пропустила:

110 из 374 churn-клиентов
3. Сравнение с Stage 5 OOF

Stage 5 OOF policy:

precision ≈ 0.5712
recall    ≈ 0.7191
F1        ≈ 0.6367

Stage 6 test:

precision ≈ 0.5399
recall    ≈ 0.7059
F1        ≈ 0.6118

Разница:

precision: -0.0313
recall:    -0.0132
F1:        -0.0249

Вывод:

Test metrics немного ниже OOF-оценки, но поведение модели в целом совпало с ожиданиями.

Сильного провала нет. Есть умеренный optimistic bias Stage 5 OOF, что нормально для threshold policy, выбранной на train-only OOF.

4. Что добавить в notebook

Добавь в markdown-блок Stage 6 final evaluation conclusions финальную версию:

## Stage 6 final evaluation conclusions

### Final frozen candidate

- Model: `LogisticRegression(class_weight="balanced")`
- Threshold policy: predict `Churn = 1` if predicted probability is at least `0.60`.

### Evaluation protocol

- The final pipeline was fitted only on `X_train` / `y_train`.
- `X_test` / `y_test` were evaluated once for final held-out test evaluation.
- No model parameters were changed after seeing test results.
- The threshold was not changed after seeing test results.
- No alternative models were evaluated on `X_test`.

### Final test metrics

At threshold `0.60`, final held-out test metrics are:

- accuracy ≈ 0.7622
- precision ≈ 0.5399
- recall ≈ 0.7059
- F1 ≈ 0.6118
- ROC-AUC ≈ 0.8413
- PR-AUC ≈ 0.6326

Confusion matrix:

- TN = 810
- FP = 225
- FN = 110
- TP = 264

### Comparison with Stage 5 OOF expectations

Stage 5 OOF policy metrics:

- precision ≈ 0.5712
- recall ≈ 0.7191
- F1 ≈ 0.6367

Stage 6 test metrics:

- precision ≈ 0.5399
- recall ≈ 0.7059
- F1 ≈ 0.6118

The final test metrics are slightly lower than the Stage 5 OOF estimates:

- precision decreased by about 0.031
- recall decreased by about 0.013
- F1 decreased by about 0.025

This suggests mild optimistic bias in the OOF threshold estimate, but no major instability or collapse on the held-out test set.

### Recall / precision trade-off

The final model catches most churn customers:

- TP = 264
- FN = 110
- recall ≈ 0.706

The cost is a moderate number of false positives:

- FP = 225
- precision ≈ 0.540

This is consistent with the intended churn-detection trade-off: the model prioritizes detecting at-risk customers, while accepting some false retention actions.

### Final honest model quality

The model is a reasonable baseline-quality churn classifier:

- ranking quality is solid but not exceptional;
- recall is useful for churn detection;
- precision is moderate, so retention actions should not be too expensive or should be further prioritized;
- threshold policy is F1-oriented, not explicitly cost-optimized.

### Limitations

- The threshold policy was selected using train-only OOF predictions, so it does not transfer perfectly to the held-out test set.
- The final threshold is not business-cost-optimized.
- The model was not calibrated explicitly.
- The dataset is relatively small, so some metric variation is expected.
- No post-test retuning is allowed after this evaluation.
5. Сохраняем Stage 6

Сохрани notebook 06_final_evaluation.ipynb.

Потом выполни:

git status
git add notebooks/06_final_evaluation.ipynb
git commit -m "Evaluate final churn model on test set"
git push

После push:

git status
git log --oneline
6. Что мне прислать для STAGE REPORT

Пришли:

git status
git log --oneline

И отдельно:

Stage 6 notebook saved: yes/no
GitHub push successful: yes/no
Threshold changed after test results: no
Alternative models evaluated on X_test: no
Model parameters changed after test results: no

После этого подготовлю финальный STAGE REPORT для координатора.

(.venv) PS C:\temp\python_learning\ml_projects\ml_projects_batch_01\02_telco_customer_churn> git status
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
(.venv) PS C:\temp\python_learning\ml_projects\ml_projects_batch_01\02_telco_customer_churn> git log --oneline
97982c8 (HEAD -> main, origin/main) Evaluate final churn model on test set
3321065 Analyze threshold behavior for churn candidates
bd4d0a5 Compare classification model families
fcfa847 Add baseline classification models
1d0bd05 Complete train test split and preprocessing plan
39e80fb Complete initial EDA
bdfbe6a Complete dataset audit and task framing
6df6bac Initialize telco customer churn project
STAGE REPORT
Project

02_telco_customer_churn

Stage

Stage 6 — Final evaluation on held-out test set

Status

Stage 6 completed, notebook saved, committed, and pushed to GitHub.

Latest commit:

97982c8 Evaluate final churn model on test set

Git status:

nothing to commit, working tree clean

Git log:

97982c8 Evaluate final churn model on test set
3321065 Analyze threshold behavior for churn candidates
bd4d0a5 Compare classification model families
fcfa847 Add baseline classification models
1d0bd05 Complete train test split and preprocessing plan
39e80fb Complete initial EDA
bdfbe6a Complete dataset audit and task framing
6df6bac Initialize telco customer churn project
Final candidate

Frozen final candidate:

LogisticRegression(class_weight="balanced")

Frozen threshold policy:

Predict Churn = 1 if predicted probability >= 0.60

No post-test changes were made to:

model parameters
threshold
feature set
preprocessing logic

No alternative models were evaluated on X_test.

Train/test split confirmation

Same split logic was reused:

train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

Data preparation:

Churn: No -> 0, Yes -> 1
X: drop customerID and Churn
TotalCharges -> numeric with errors="coerce"

Split:

X_train: (5634, 19)
X_test:  (1409, 19)
y_train: (5634,)
y_test:  (1409,)

X_test / y_test were evaluated for the first and only final test evaluation.

Final pipeline

Numeric preprocessing:

SimpleImputer(strategy="median")
StandardScaler()

Categorical preprocessing:

SimpleImputer(strategy="most_frequent")
OneHotEncoder(handle_unknown="ignore")

Model:

LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

The final pipeline was fitted only on:

X_train / y_train
Final test metrics

Threshold:

0.60

Held-out test metrics:

accuracy  ≈ 0.7622
precision ≈ 0.5399
recall    ≈ 0.7059
F1        ≈ 0.6118
ROC-AUC   ≈ 0.8413
PR-AUC    ≈ 0.6326
Confusion matrix
                 predicted_0   predicted_1
actual_0              810           225
actual_1              110           264

Components:

TN = 810
FP = 225
FN = 110
TP = 264

Interpretation for churn class:

Detected churn customers: 264 / 374
Missed churn customers:   110 / 374
False churn alerts:       225
Classification report summary

For Churn = 1:

precision ≈ 0.5399
recall    ≈ 0.7059
F1        ≈ 0.6118
support   = 374

For No churn = 0:

precision ≈ 0.8804
recall    ≈ 0.7826
F1        ≈ 0.8286
support   = 1035
Comparison with Stage 5 OOF policy

Stage 5 OOF policy:

model: LogisticRegression(class_weight="balanced")
threshold ≈ 0.60
precision ≈ 0.5712
recall    ≈ 0.7191
F1        ≈ 0.6367

Stage 6 held-out test:

model: LogisticRegression(class_weight="balanced")
threshold = 0.60
precision ≈ 0.5399
recall    ≈ 0.7059
F1        ≈ 0.6118

Difference:

precision: -0.0313
recall:    -0.0132
F1:        -0.0249

Conclusion:

Final test metrics are slightly lower than Stage 5 OOF estimates, but broadly aligned.
There is mild optimistic bias in the train-only OOF threshold estimate, but no major collapse on the held-out test set.
Leakage checks

Confirmed:

customerID excluded from X
Churn excluded from X
TotalCharges converted to numeric
same split reused
final pipeline fitted only on X_train / y_train
X_test probabilities predicted once
frozen threshold 0.60 applied once
no threshold change after test metrics
no model parameter change after test metrics
no alternative models evaluated on X_test
final model not saved yet
Final evaluation conclusions

The final model is a reasonable churn-classification baseline.

Strengths:

recall is useful for churn detection;
model catches about 70.6% of churn customers;
pipeline is leakage-safe and reproducible;
test performance is close to OOF expectations.

Limitations:

precision is moderate, about 54%;
225 false positives may be costly if retention actions are expensive;
threshold is F1-oriented, not explicitly business-cost-optimized;
model is not calibrated explicitly;
no post-test retuning is allowed.

Stage 6 is complete. Final model artifact should not be saved until coordinator approves the test results and finalization step.

Coordinator approved Stage 6.

Stage 6 is accepted:
- final frozen candidate evaluated on X_test exactly once;
- frozen threshold 0.60 applied once;
- no threshold change after test evaluation;
- no model parameter changes after test evaluation;
- no alternative models evaluated on X_test;
- final model not saved yet.

Final test metrics:
accuracy ≈ 0.7622
precision ≈ 0.5399
recall ≈ 0.7059
F1 ≈ 0.6118
ROC-AUC ≈ 0.8413
PR-AUC ≈ 0.6326

Confusion matrix:
TN = 810
FP = 225
FN = 110
TP = 264

Main finding:
Final test metrics are slightly lower than Stage 5 OOF estimates but broadly aligned. The model catches about 70.6% of churn customers, but precision is moderate.

Important limitation:
The threshold policy is F1-oriented and may produce too many false positives if retention actions are expensive.

Proceed to Stage 7 — Save final pipeline and update README.

Requirements:
1. Fit the final pipeline on X_train / y_train using the frozen model and preprocessing.
2. Save a model bundle to models/, including:
   - fitted pipeline;
   - threshold = 0.60;
   - feature lists if needed;
   - target mapping if needed.
3. Verify artifact loading.
4. Verify sample predictions and probability-to-threshold logic.
5. Keep model artifact ignored by Git if models/ is ignored.
6. Update README.md with:
   - project goal;
   - dataset;
   - task framing;
   - final model;
   - final threshold;
   - final test metrics;
   - confusion matrix;
   - key limitations;
   - leakage controls;
   - reproduction steps;
   - project structure.
7. Commit code/notebook/README changes.
8. Provide final STAGE REPORT for coordinator.

Принято. Stage 7 — финализация артефакта: сохраняем pipeline + threshold bundle, проверяем загрузку и только потом обновляем README. Model artifact остаётся вне Git, если models/ уже в .gitignore.

Думал на протяжении 4s
1. Что делаем на текущем шаге

Начинаем Stage 7 — Save final pipeline and update README.

Сначала делаем техническую часть:

создаём notebook для финализации;
воспроизводим ту же подготовку данных и split;
строим frozen final pipeline;
fit только на X_train / y_train;
сохраняем bundle в models/;
загружаем artifact обратно;
проверяем sample probabilities и threshold logic.

README обновим следующим шагом после проверки artifact.

2. Зачем

На Stage 6 мы получили честную финальную test-оценку. Теперь нужно сохранить не просто модель, а воспроизводимый inference artifact.

Нам нужен bundle, где лежит:

fitted pipeline
threshold = 0.60
feature lists
target mapping
model metadata

Почему не только pipeline?

Потому что для классификации churn финальное решение зависит не только от вероятности, но и от frozen threshold:

predict Churn = 1 if proba >= 0.60

Если сохранить только pipeline, можно случайно использовать дефолтный threshold 0.5, и это будет уже другая модельная политика.

3. Команды или код

В PowerShell из корня проекта:

cd C:\temp\python_learning\ml_projects\ml_projects_batch_01\02_telco_customer_churn
.venv\Scripts\Activate.ps1
New-Item notebooks\07_save_final_pipeline.ipynb

Открой notebook и выбери kernel:

Python (02_telco_customer_churn)
Ячейка 1 — environment check
import sys
from pathlib import Path

print(sys.executable)
print(Path.cwd())
Ячейка 2 — imports
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
Ячейка 3 — paths
PROJECT_ROOT = Path.cwd().parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_BUNDLE_PATH = MODELS_DIR / "final_churn_pipeline_bundle.joblib"

MODELS_DIR.mkdir(exist_ok=True)

DATA_PATH, MODEL_BUNDLE_PATH
Ячейка 4 — load raw data
df = pd.read_csv(DATA_PATH)

df.shape
Ячейка 5 — create X/y
target_col = "Churn"
id_cols = ["customerID"]

target_mapping = {
    "No": 0,
    "Yes": 1,
}

inverse_target_mapping = {
    0: "No",
    1: "Yes",
}

y = df[target_col].map(target_mapping)

X = df.drop(columns=id_cols + [target_col])
X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

X.shape, y.shape

Проверки:

assert y.isna().sum() == 0
assert set(y.unique()) == {0, 1}
assert "customerID" not in X.columns
assert "Churn" not in X.columns
Ячейка 6 — feature lists
numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

all_features = numeric_features + categorical_features

len(numeric_features), len(categorical_features), len(all_features), X.shape[1]

Проверка:

assert set(all_features) == set(X.columns)
Ячейка 7 — same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

Важно: на Stage 7 мы используем X_train / y_train для fit artifact.
X_test здесь не нужен для retuning. Его не используем для новых решений.

Ячейка 8 — build final frozen pipeline
numeric_preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocessor, numeric_features),
        ("cat", categorical_preprocessor, categorical_features),
    ]
)

final_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ]
)

final_pipeline
Ячейка 9 — fit final pipeline on X_train only
final_pipeline.fit(X_train, y_train)
Ячейка 10 — create model bundle
final_threshold = 0.60

model_bundle = {
    "pipeline": final_pipeline,
    "threshold": final_threshold,
    "target_mapping": target_mapping,
    "inverse_target_mapping": inverse_target_mapping,
    "target_col": target_col,
    "id_cols": id_cols,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "all_features": all_features,
    "positive_class": 1,
    "positive_class_label": "Yes",
    "model_name": "LogisticRegression(class_weight='balanced')",
    "split": {
        "test_size": 0.2,
        "random_state": 42,
        "stratify": True,
    },
    "final_test_metrics": {
        "accuracy": 0.7622,
        "precision": 0.5399,
        "recall": 0.7059,
        "f1": 0.6118,
        "roc_auc": 0.8413,
        "pr_auc": 0.6326,
        "threshold": final_threshold,
    },
    "confusion_matrix": {
        "tn": 810,
        "fp": 225,
        "fn": 110,
        "tp": 264,
    },
}

model_bundle.keys()
Ячейка 11 — save artifact
joblib.dump(model_bundle, MODEL_BUNDLE_PATH)

MODEL_BUNDLE_PATH

Проверка, что файл появился:

MODEL_BUNDLE_PATH.exists(), MODEL_BUNDLE_PATH.stat().st_size
Ячейка 12 — load artifact back
loaded_bundle = joblib.load(MODEL_BUNDLE_PATH)

loaded_bundle.keys()

Проверки:

assert loaded_bundle["threshold"] == 0.60
assert loaded_bundle["target_mapping"] == {"No": 0, "Yes": 1}
assert loaded_bundle["numeric_features"] == numeric_features
assert loaded_bundle["categorical_features"] == categorical_features
Ячейка 13 — verify sample predictions

Берём первые 10 строк из X_train, чтобы проверить artifact mechanics. Это не новая оценка качества.

sample_X = X_train.head(10).copy()

sample_proba = loaded_bundle["pipeline"].predict_proba(sample_X)[:, 1]
sample_pred = (sample_proba >= loaded_bundle["threshold"]).astype(int)

sample_predictions = pd.DataFrame({
    "churn_probability": sample_proba,
    "predicted_class": sample_pred,
    "predicted_label": [loaded_bundle["inverse_target_mapping"][pred] for pred in sample_pred],
})

sample_predictions

Проверка threshold logic:

manual_pred = (sample_predictions["churn_probability"] >= 0.60).astype(int)

assert np.array_equal(
    manual_pred.to_numpy(),
    sample_predictions["predicted_class"].to_numpy(),
)

sample_predictions
Ячейка 14 — artifact summary markdown

Добавь markdown:

## Stage 7 artifact summary

### Saved artifact

Saved model bundle:

- `models/final_churn_pipeline_bundle.joblib`

### Bundle contents

The bundle includes:

- fitted sklearn pipeline;
- frozen threshold: `0.60`;
- target mapping:
  - `No` -> 0
  - `Yes` -> 1
- inverse target mapping;
- numeric feature list;
- categorical feature list;
- final test metrics;
- final confusion matrix;
- split metadata.

### Final pipeline

Numeric preprocessing:

- `SimpleImputer(strategy="median")`
- `StandardScaler()`

Categorical preprocessing:

- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore")`

Model:

- `LogisticRegression(class_weight="balanced")`

### Verification

- Artifact was saved with `joblib`.
- Artifact was loaded back successfully.
- Threshold value was verified.
- Feature lists were verified.
- Sample predictions were generated.
- Probability-to-threshold logic was verified.

### Git note

The `models/` directory is ignored by Git, so the binary model artifact is not committed to the repository.
4. Что должно получиться

Ожидаемый artifact:

models/final_churn_pipeline_bundle.joblib

Проверка файла:

exists: True
size: больше 0

Проверка bundle:

threshold = 0.60
target_mapping = {"No": 0, "Yes": 1}
pipeline загружается
predict_proba работает
threshold logic работает
5. Что мне прислать для проверки

Пришли:

MODEL_BUNDLE_PATH.exists(), MODEL_BUNDLE_PATH.stat().st_size
loaded_bundle.keys()
sample_predictions

И отдельно:

Artifact loading verified: yes/no
Sample predictions verified: yes/no
Model artifact committed to Git: no

Пока README не обновляй — сначала проверим artifact.

6. Контрольный вопрос

Почему в bundle нужно сохранить threshold 0.60, а не только fitted sklearn pipeline?


WindowsPath('c:/temp/python_learning/ml_projects/ml_projects_batch_01/02_telco_customer_churn/models/final_churn_pipeline_bundle.joblib')
dict_keys(['pipeline', 'threshold', 'target_mapping', 'inverse_target_mapping', 'target_col', 'id_cols', 'numeric_features', 'categorical_features', 'all_features', 'positive_class', 'positive_class_label', 'model_name', 'split', 'final_test_metrics', 'confusion_matrix'])
churn_probability	predicted_class	predicted_label
0	0.612081	1	Yes
1	0.526236	0	No
2	0.104638	0	No
3	0.217760	0	No
4	0.628100	1	Yes
5	0.045897	0	No
6	0.370852	0	No
7	0.275508	0	No
8	0.043155	0	No
9	0.131629	0	No
Думал на протяжении 4s
1. Что делаем на текущем шаге

Artifact verification выглядит корректно.

Загруженный bundle содержит нужные ключи:

pipeline
threshold
target_mapping
inverse_target_mapping
feature lists
final_test_metrics
confusion_matrix

Sample predictions тоже подтверждают threshold logic:

probability >= 0.60 → predicted_class = 1 / Yes
probability < 0.60  → predicted_class = 0 / No

Например:

0.612081 → 1 / Yes
0.526236 → 0 / No
0.628100 → 1 / Yes

Это именно та frozen policy, которую мы хотели сохранить.

2. Маленькая проверка перед README

Ты прислал путь к artifact, но не прислал размер файла. Выполни, пожалуйста, одну ячейку:

MODEL_BUNDLE_PATH.exists(), MODEL_BUNDLE_PATH.stat().st_size

Ожидаемо:

(True, какое-то число больше 0)

Если там True и размер больше нуля — artifact сохранён нормально.

3. Теперь обновляем README.md

Открой README.md и замени содержимое на это.

# 02_telco_customer_churn

Учебный ML-проект по бинарной классификации оттока клиентов на табличных данных.

Dataset: Telco Customer Churn  
Source: Kaggle — `blastchar/telco-customer-churn`

## Project goal

Цель проекта — построить воспроизводимый baseline-пайплайн для задачи churn prediction:

- корректно провести dataset audit и EDA;
- избежать data leakage;
- разделить данные на train/test;
- обрабатывать числовые и категориальные признаки через `ColumnTransformer`;
- сравнить baseline-модели;
- проанализировать threshold behavior;
- один раз оценить финальный frozen candidate на held-out test set;
- сохранить финальный inference artifact.

## Task framing

Task type:

```text
binary classification

Target:

Churn

Target mapping:

No  -> 0
Yes -> 1

Positive class:

1 = Churn Yes

Business interpretation:

False Positive: модель предсказывает churn, но клиент фактически не уходит.
Возможная цена: лишнее retention-действие, скидка, промо или контакт.
False Negative: модель предсказывает no churn, но клиент фактически уходит.
Возможная цена: потерянный клиент и будущая выручка.

Для churn-задачи важны не только accuracy, но и:

precision;
recall;
F1;
ROC-AUC;
PR-AUC;
confusion matrix.
Dataset summary

Original dataset shape:

7043 rows
21 columns

Target distribution:

No     5174 / 73.46%
Yes    1869 / 26.54%

Main preprocessing notes:

customerID is an identifier and is excluded from features.
TotalCharges is semantically numeric but loaded as string in the raw CSV.
TotalCharges contains 11 hidden missing values represented as blank strings.
TotalCharges is converted with:
pd.to_numeric(X["TotalCharges"], errors="coerce")
Feature groups

Numeric features:

SeniorCitizen
tenure
MonthlyCharges
TotalCharges

Categorical features:

gender
Partner
Dependents
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
Train/test split

The same split logic was used throughout the project:

train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

Split sizes:

X_train: (5634, 19)
X_test:  (1409, 19)
y_train: (5634,)
y_test:  (1409,)

The test set was kept untouched until Stage 6 final evaluation.

Preprocessing

All preprocessing is inside sklearn Pipeline / ColumnTransformer.

Numeric pipeline:

SimpleImputer(strategy="median")
StandardScaler()

Categorical pipeline:

SimpleImputer(strategy="most_frequent")
OneHotEncoder(handle_unknown="ignore")

Final model pipeline:

ColumnTransformer -> LogisticRegression(class_weight="balanced")
Model development summary
Stage 3 — Baseline models

Models compared with 5-fold StratifiedKFold CV on X_train only:

DummyClassifier(strategy="most_frequent")
LogisticRegression
LogisticRegression(class_weight="balanced")
DecisionTreeClassifier(max_depth=3)

Strongest simple baseline:

LogisticRegression(class_weight="balanced")

Reason:

best F1;
best recall;
better aligned with churn detection.
Stage 4 — Controlled model comparison

Additional model families compared:

RandomForestClassifier
ExtraTreesClassifier
GradientBoostingClassifier
HistGradientBoostingClassifier

Main finding:

LogisticRegression(class_weight="balanced") remained best by F1 and recall at default threshold.
GradientBoostingClassifier slightly improved ranking metrics.

Best ranking candidate:

GradientBoostingClassifier
Stage 5 — Threshold analysis

Threshold analysis was performed only with train-only out-of-fold probabilities.

Current best OOF-F1 policy:

Model: LogisticRegression(class_weight="balanced")
Threshold: 0.60

Stage 5 OOF metrics:

precision ≈ 0.5712
recall    ≈ 0.7191
F1        ≈ 0.6367

Important finding:

After threshold adjustment, these candidates were very close by F1:

LogisticRegression(class_weight="balanced")
LogisticRegression
GradientBoostingClassifier

The final choice should consider the business trade-off between precision and recall.

Final model

Frozen final candidate:

LogisticRegression(class_weight="balanced")

Frozen threshold policy:

Predict Churn = 1 if predicted probability >= 0.60

No threshold tuning was performed after test evaluation.
No model parameter changes were made after test evaluation.
No alternative models were evaluated on the test set.

Final held-out test metrics

Final evaluation was performed once on the held-out test set.

Threshold:

0.60

Final test metrics:

accuracy  ≈ 0.7622
precision ≈ 0.5399
recall    ≈ 0.7059
F1        ≈ 0.6118
ROC-AUC   ≈ 0.8413
PR-AUC    ≈ 0.6326

Confusion matrix:

TN = 810
FP = 225
FN = 110
TP = 264

Churn-class interpretation:

Detected churn customers: 264 / 374
Missed churn customers:   110 / 374
False churn alerts:       225
Comparison with Stage 5 OOF estimate

Stage 5 OOF policy:

precision ≈ 0.5712
recall    ≈ 0.7191
F1        ≈ 0.6367

Stage 6 held-out test:

precision ≈ 0.5399
recall    ≈ 0.7059
F1        ≈ 0.6118

Difference:

precision: -0.0313
recall:    -0.0132
F1:        -0.0249

Conclusion:

Final test metrics are slightly lower than Stage 5 OOF estimates but broadly aligned. There is mild optimistic bias in the train-only OOF threshold estimate, but no major collapse on the held-out test set.

Key limitations
Precision is moderate: about 54%.
The model produces 225 false positives on the held-out test set.
If retention actions are expensive, the threshold may produce too many unnecessary interventions.
The threshold policy is F1-oriented, not explicitly business-cost-optimized.
The model was not calibrated explicitly.
The dataset is relatively small, so metric variation is expected.
No post-test retuning is allowed.
Leakage controls

The project follows these leakage-control rules:

customerID excluded from X;
Churn excluded from X;
TotalCharges converted to numeric before modeling;
preprocessing placed inside Pipeline / ColumnTransformer;
imputers, scalers, and encoders fitted only on train folds during CV;
model selection used only X_train;
threshold analysis used only train-only OOF predictions;
X_test evaluated only once in Stage 6;
no model changes after final test evaluation.
Saved artifact

Final model bundle:

models/final_churn_pipeline_bundle.joblib

The bundle contains:

fitted sklearn pipeline;
frozen threshold 0.60;
target mapping;
inverse target mapping;
feature lists;
final test metrics;
final confusion matrix;
split metadata.

The models/ directory is ignored by Git, so the binary artifact is not committed.

Project structure
02_telco_customer_churn/
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
├── models/
│   └── final_churn_pipeline_bundle.joblib
├── notebooks/
│   ├── 00_dataset_audit.ipynb
│   ├── 01_initial_eda.ipynb
│   ├── 02_train_test_split.ipynb
│   ├── 03_baseline_classification.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_threshold_and_refinement.ipynb
│   ├── 06_final_evaluation.ipynb
│   └── 07_save_final_pipeline.ipynb
├── src/
├── .gitignore
├── README.md
└── requirements.txt
Reproduction steps

From the project root:

cd C:\temp\python_learning\ml_projects\ml_projects_batch_01\02_telco_customer_churn
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Run notebooks in order:

00_dataset_audit.ipynb
01_initial_eda.ipynb
02_train_test_split.ipynb
03_baseline_classification.ipynb
04_model_comparison.ipynb
05_threshold_and_refinement.ipynb
06_final_evaluation.ipynb
07_save_final_pipeline.ipynb

Use the project kernel:

Python (02_telco_customer_churn)
Current status

Project completed through final evaluation and artifact saving.

Final model artifact is saved locally but not committed to Git.


---

## 4. После README

Сохрани `README.md`, потом выполни:

```powershell
git status

Важно: models/final_churn_pipeline_bundle.joblib не должен попасть в tracked files, если models/ игнорируется.

Потом:

git add README.md notebooks/07_save_final_pipeline.ipynb
git commit -m "Save final pipeline and update README"
git push

После push:

git status
git log --oneline
5. Что мне прислать для финального STAGE REPORT

Пришли:

git status
git log --oneline

И отдельно:

Artifact saved: yes/no
Artifact loading verified: yes/no
Sample predictions verified: yes/no
Model artifact committed to Git: no
README updated: yes/no
GitHub push successful: yes/no
6. Контрольный вопрос

Почему для inference недостаточно знать только model.predict_proba(X), а нужно ещё хранить бизнес-правило threshold 0.60?

(True, 9117)

Отлично. Artifact сохранён нормально:

MODEL_BUNDLE_PATH.exists(): True
file size: 9117 bytes

То есть Stage 7 artifact part принят:

Artifact saved: yes
Artifact loading verified: yes
Sample predictions verified: yes
Threshold logic verified: yes

Теперь завершаем Stage 7 через README и Git.

1. Что делаем сейчас

Обнови README.md текстом из прошлого сообщения, сохрани файл и выполни:

git status

Проверь важный момент:

models/final_churn_pipeline_bundle.joblib

не должен быть в staged/tracked files, если models/ есть в .gitignore.

Затем:

git add README.md notebooks/07_save_final_pipeline.ipynb
git commit -m "Save final pipeline and update README"
git push

После push:

git status
git log --oneline
2. Что мне прислать

Пришли:

git status
git log --oneline

И отдельно:

Artifact saved: yes
Artifact loading verified: yes
Sample predictions verified: yes
Model artifact committed to Git: no
README updated: yes/no
GitHub push successful: yes/no
3. Контрольный вопрос

Почему для inference недостаточно знать только model.predict_proba(X), а нужно ещё хранить business rule:

threshold = 0.60
