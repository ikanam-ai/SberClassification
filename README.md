# SberClassification
**Прогнозирование оттока зарплатных клиентов ФЛ - Сбер 💼**

**Метрики AUC 📊:**
- 0.7613 - CatBoost с препроцессингом и выкрутасами
- 0.7601 - CatBoost с препроцессингом
- 0.7561 - XGBoost
- 0.6906 - Classifier_base с 4 линейными слоями
- 0.7010 - Classifier_dropout
- 0.7072 - Classifier_dropout2
- 0.7308 - Classifier_dropout2 с препроцессингом
- 0.7463 - Classifier_dropout2 с препроцессингом и CatBoost

**Модели и подходы 🛠️:**
- **CatBoostClassifier:** Использован с препроцессингом и выкрутасами, а также без препроцессинга (параметры: iterations=7500, learning_rate=0.01, depth=6, loss_function='MultiClass', eval_metric='AUC', random_seed=0, class_weights=[1, 12], task_type='GPU').
```python
from catboost import CatBoostClassifier

model_catboost = CatBoostClassifier(iterations=7500,
                                    learning_rate=0.01,
                                    depth=6,
                                    loss_function='MultiClass',
                                    eval_metric='AUC',
                                    random_seed=0,
                                    class_weights=[1, 12],  # примерное соотношение классов
                                    task_type='GPU')
```
- **XGBoost:** Применялся для прогнозирования с AUC метрикой 0.7561.
```python
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Создание модели XGBoost
model_xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=4, random_state=0)

# Обучение модели с выводом логов и графика
eval_set = [(X_train, y_train), (X_val, y_val)]  # Указываем обучающий и валидационный наборы данных
model_xgb.fit(X_train, y_train, eval_set=eval_set, eval_metric="auc", verbose=True)
```
- **Classifier_base, Classifier_dropout, Classifier_dropout2:** Линейные классификаторы с разными конфигурациями и слоями.
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Define your neural network model
class Classifier(nn.Module):
    def __init__(self, input_size):
        super(Classifier, self).__init__()
        self.fc0 = nn.Linear(input_size, 1028)
        self.fc1 = nn.Linear(1028, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)  # Adding dropout for regularization

    def forward(self, x):
        x = torch.relu(self.fc0(x))
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x
    
    def get_embed(self, x):
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x
```
- **Фиче инжиниринг:** Применен для улучшения производительности моделей.
```python
categorical_features = [col for col in df.columns[1:] if df[col].nunique() < 20]
numerical_features = [col for col in df.columns[1:] if df[col].nunique() >= 20]
# Метод Label Encoding
for feature in categorical_features:
    df[feature + '_label_encoded'] = df[feature].astype('category').cat.codes

# Метод One-Hot Encoding
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_features = pd.DataFrame(encoder.fit_transform(df[categorical_features]))
encoded_features.columns = encoder.get_feature_names_out(categorical_features)

df = pd.concat([df, encoded_features], axis=1)
```
- **Препроцессинг:** Использован для обработки данных с учетом разреженности данных, выбросов и улучшения точности предсказаний.
```python
# Применение всех методов масштабирования
scalers = {
    #"StandardScaler": StandardScaler(),
    #"MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
    #"Normalizer": Normalizer(),
    #"MaxAbsScaler": MaxAbsScaler(),
}

scaled_dfs = []

for scaler_name, scaler in tqdm(scalers.items()):
    scaled_data = scaler.fit_transform(df[numerical_features])
    scaled_df = pd.DataFrame(scaled_data, columns=[f"{feature}_{scaler_name}" for feature in numerical_features])
    scaled_dfs.append((scaler_name, scaled_df))
    
# Объединение результатов
df = pd.concat([df] + [scaled_df for _, scaled_df in scaled_dfs], axis=1)
```
- **Количество признаков:** Исходные данные содержали более 1000 признаков, в результате фиче инжиниринга получено более 4000 признаков.
- **Voting ансамбль:** Использован для объединения прогнозов различных моделей.
```python
  from sklearn.ensemble import VotingClassifier
voting_classifier = VotingClassifier(estimators=[('catboost', model_catboost), ('xgboost', model_xgboost)], voting='soft')
voting_classifier.fit(X_train, y_train)
```
- **CatBoost и XGBoost:** Популярные библиотеки градиентного бустинга, применяемые для построения моделей.

**Проблемы и решения 🛠️:**
- **Разреженность данных:** Обработка разреженных данных требует специального внимания и методов обработки.
- **Выбросы:** Необходимо принимать меры для обнаружения и обработки выбросов в данных.
- **Низкая точность наивных предсказаний:** Модели должны быть настроены и оптимизированы для достижения высокой точности предсказаний.


**Отбор признаков 🔍:**
- Код для отбора признаков будет предоставлен в репозитории проекта для обеспечения прозрачности и воспроизводимости результатов.
- **permutation importance:** Вычисления важности признаков в моделях машинного обучения.
```python
from sklearn.inspection import permutation_importance
scoring = 'roc_auc'
r = permutation_importance(model, X_test, y_test,
                           n_repeats=20,
                           random_state=0,scoring = scoring)
for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{X_test.columns[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")
```
- **RecursiveByShapValues**:
```python

clf.select_features(
                train_pool,
                eval_set=test_pool,
                features_for_select = list(X_train.columns),
                num_features_to_select=500,
                steps=10,
                algorithm='RecursiveByShapValues',
                shap_calc_type='Regular',
                train_final_model=True,
                plot=False,
                verbose = 1000)
```
- **shapley_feature_ranking**:
```python

def shapley_feature_ranking(explanation,func = np.mean):
    '''
    func по умолчанию np.mean, но можно заменить например на np.max
    '''
    feature_order = np.argsort(func(np.abs(explanation.values), axis=0))
    
    return pd.DataFrame(
        {
            "features": [explanation.feature_names[i] for i in feature_order][::-1],
            "importance": [
                func(np.abs(explanation.values), axis=0)[i] for i in feature_order
            ][::-1],
        }
    )


feature_names = X_train.columns

shap_values = explainer.shap_values(Pool(X_test, y_test, cat_features=cat_feature_names))

base_values = np.mean(shap_values)

explanation = shap.Explanation(values=shap_values, data=X_test, feature_names=feature_names, base_values=base_values)

important_feature = shapley_feature_ranking(explanation)
plt.show(important_feature.head(10))
```

**Результаты и обсуждение 💡:**
- Исследован ряд моделей и подходов к прогнозированию оттока зарплатных клиентов ФЛ в Сбере.
- Модели CatBoost показали лучшие результаты по метрике AUC.
- Применены различные методы препроцессинга данных, фиче инжиниринга и объединения моделей для повышения качества прогнозов.
- Решены проблемы с разреженностью данных, выбросами и низкой точностью предсказаний.
- Репозиторий содержит подробную документацию, включая код для отбора признаков и описания примененных методов.

Решение: sber.ipynb
