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
- **XGBoost:** Применялся для прогнозирования с AUC метрикой 0.7561.
- **Classifier_base, Classifier_dropout, Classifier_dropout2:** Линейные классификаторы с разными конфигурациями и слоями.
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
