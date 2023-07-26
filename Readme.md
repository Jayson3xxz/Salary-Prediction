# Предсказывание заработной платы
## Постановка задачи
Предсказывание заработной платы человека на основе параметров из базы данных.

## Артхитектура модели
В качестве модели машинного обучения был выбран алгоритм GradientBoosting из библиотеки scikit-learn
## Этапы решения поставленной задачи
- Избавление от ненужных переменных (логически проанализировав)
- Проверка распределения ключевой переменной на нормальность (Проверка гипотезы: описывается ли искомая переменная формулой нормального распределения)
- Избавление от выбросов (В зависимости от решения основанного на предыдущем пункте)
- Построение таблицы корреляции для выбора  переменных
- Построение таблицы корреляции для выбранных переменных (поиск мультиколлениарности)
- Сделав соответствующий вывод, что корреляционный анализ не подходит, переходим к методу главных компонент
- Проанализировав базу данных без искомой величины определяем необходимое количество переменных для алгоритма регресии
- Разбиваем набор данных на тренировочную и тестовую выборки
- Обучаем модель,получаем результаты
## Итоги
Модель основанная на алгоритме GradientBoosting выдаёт результаты:
- На тренировочной выборке : 0.992798
- На тестовой : 94557664
В итоге получили достаточно высокую точность и усточивость модели(устойчивость с точностью до 3 знака)