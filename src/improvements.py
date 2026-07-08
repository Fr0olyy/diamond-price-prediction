import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor



MAX_REASONABLE_PRICE = 20_000


def remove_zero_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)].copy()

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df['y'] < 30) & (df['z'] < 30)].copy()

def add_volume_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['volume'] = df['x'] * df['y'] * df['z']
    return df

def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    
    df['cut_encoded'] = df['cut'].map({cat: i for i, cat in enumerate(cut_order)})
    df['color_encoded'] = df['color'].map({cat: i for i, cat in enumerate(color_order)})
    df['clarity_encoded'] = df['clarity'].map({cat: i for i, cat in enumerate(clarity_order)})
    return df

def evaluate_custom_model(X, y, model, is_target_logged=False, name='Model', evaluation_stage='val'):
    """
    Универсальная функция оценки.
    y — всегда передаем в РЕАЛЬНЫХ ДОЛЛАРАХ (так проще и чище).
    evaluation_stage: 'val' (для шагов 1-8) или 'test' (для финального замера).
    """
    # 1. Честный сплит на 3 выборки (Train 60% / Val 20% / Test 20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    # Выбираем, на каком этапе валидируемся
    if evaluation_stage == 'val':
        X_eval, y_eval = X_val, y_val
    else:
        X_eval, y_eval = X_test, y_test

    # 2. Модифицируем модель, если нужен лог-таргет 
    if is_target_logged:
        # Оборачиваем модель в TransformedTargetRegressor. 
        # Он сам сделает log1p перед fit() и сам сделает expm1 при predict() и кросс-валидации
        fit_model = TransformedTargetRegressor(
            regressor=model, 
            func=np.log1p, 
            inverse_func=np.expm1
        )
    else:
        fit_model = model

    # 3. Обучение и предсказание (предсказания теперь ВСЕГДА в долларах благодаря wrapper)
    fit_model.fit(X_train, y_train)
    y_pred = fit_model.predict(X_eval)
    
    # Защита от выбросов линейной регрессии
    y_pred = np.clip(y_pred, a_min=0, a_max=20000)
    
    # 4. Честная кросс-валидация в долларах на train_val выборке
    cv_scores = cross_val_score(fit_model, X_train_val, y_train_val, cv=5, scoring='r2')
    
    # 5. Расчет метрик (все в одной шкале — в долларах)
    rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
    mae = mean_absolute_error(y_eval, y_pred)
    r2 = r2_score(y_eval, y_pred)
    mape = np.mean(np.abs((y_eval.values - y_pred) / y_eval.values)) * 100

    # 6. Вывод результатов
    print(f"==========================================")
    print(f"📌 МОДЕЛЬ: {name} ({evaluation_stage.upper()} STAGE)")
    print(f"==========================================")
    print(f"R² (в долларах): {r2:.4f}")
    print(f"CV R² (в долларах): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"RMSE: {rmse:.2f} $")
    print(f"MAE: {mae:.2f} $")
    print(f"MAPE: {mape:.2f}%\n")

    # 7. Вывод коэффициентов (извлекаем их из обертки, если они есть)
    actual_model = fit_model.regressor if is_target_logged else fit_model
    if hasattr(actual_model, 'named_steps'):
        trained_linear_model = actual_model.named_steps['model']
    else:
        trained_linear_model = actual_model 

    if hasattr(trained_linear_model, 'coef_'):
        coef_df = pd.DataFrame({
            'Признак': X.columns,
            'Коэффициент': trained_linear_model.coef_
        }).sort_values('Коэффициент', ascending=False)

        print("Коэффициенты модели (после масштабирования):")
        print(coef_df.to_string(index=False, float_format=lambda x: f'{x:+.4f}'))
        print("==========================================\n")