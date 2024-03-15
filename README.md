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
        return x```
- **Фиче инжиниринг:** Применен для улучшения производительности моделей.
- **Препроцессинг:** Использован для обработки данных с учетом разреженности данных, выбросов и улучшения точности предсказаний.
- **Количество признаков:** Исходные данные содержали более 1000 признаков, в результате фиче инжиниринга получено более 4000 признаков.
- **Voting ансамбль:** Использован для объединения прогнозов различных моделей.
- **CatBoost и XGBoost:** Популярные библиотеки градиентного бустинга, применяемые для построения моделей.

**Проблемы и решения 🛠️:**
- **Разреженность данных:** Обработка разреженных данных требует специального внимания и методов обработки.
- **Выбросы:** Необходимо принимать меры для обнаружения и обработки выбросов в данных.
- **Низкая точность наивных предсказаний:** Модели должны быть настроены и оптимизированы для достижения высокой точности предсказаний.

**Отбор признаков 🔍:**
- Код для отбора признаков будет предоставлен в репозитории проекта для обеспечения прозрачности и воспроизводимости результатов.


**Результаты и обсуждение 💡:**
- Исследован ряд моделей и подходов к прогнозированию оттока зарплатных клиентов ФЛ в Сбере.
- Модели CatBoost показали лучшие результаты по метрике AUC.
- Применены различные методы препроцессинга данных, фиче инжиниринга и объединения моделей для повышения качества прогнозов.
- Решены проблемы с разреженностью данных, выбросами и низкой точностью предсказаний.
- Репозиторий содержит подробную документацию, включая код для отбора признаков и описания примененных методов.

Решение: Sber.ipynb
