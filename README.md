# Final_project
Прогнозирование стоимости жилья

Нужно искать способы точной оценки, как со стороны продавца, так и со стороны покупателя. Поэтому крайне важно независимое, быстрое и точное знание о ценах на рынке жилой недвижимости.

Проведен анализ данных и построены модели для прогнозирования стоимости жилья.
Стоит отметить что анализ метрики проводился на логарифмированной стоимости. в случае проверки моделей на других данных надо взять логарифм от стоимости.
Результаты
 нормализация: StandardScaler()
LinearRegression MSE = 0.84, RMSE = 0.92 После удаления признаков: MSE = 0.85, RMSE = 0.92
RandomForestRegressor MSE = 2.28, RMSE = 1.51 
нормализация MinMaxScaler()
LinearRegression MSE = 0.86, RMSE = 0.93
RandomForestRegression MSE = 0.52, RMSE = 0.72

Линейная регрессия с использованием методов оптимизации MSE = 0.90, RMSE = 0.95
RandomForestRegressor c get_dummies MSE = 0.40, RMSE = 0.63

Randomize(RandomForestRegressor) MSE = 0.17, RMSE = 0.41
GradientBoostingRegressor MSE = 0.38, RMSE = 0.62
