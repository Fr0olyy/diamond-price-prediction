import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


MAX_REASONABLE_PRICE = 25_000


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

def evaluate_custom_model(X, y, model, is_target_logged=False):
    # 1. Сплит данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Обучение и предсказание
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 3. Кросс-валидация (считает ровно то, что ей передали в y_train)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # 4. Перевод в доллары ДЛЯ МЕТРИК (если обучались на логарифмах)
    if is_target_logged:
        # Защита от слишком больших предсказаний линейной регрессии
        y_pred = np.clip(y_pred, a_min=0, a_max=np.log1p(MAX_REASONABLE_PRICE))
        
        y_test_eval = np.expm1(y_test)
        y_pred_eval = np.expm1(y_pred)
    else:
        y_test_eval = y_test
        y_pred_eval = y_pred
        
    # 5. Расчет метрик (как в твоем базовом коде)
    rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred_eval))
    mae = mean_absolute_error(y_test_eval, y_pred_eval)
    r2 = r2_score(y_test_eval, y_pred_eval)
    mape = np.mean(np.abs((y_test_eval.values - y_pred_eval) / y_test_eval.values)) * 100

    print(f"R² (в долларах): {r2:.4f}")
    
    # Поясняем в принте, в какой шкале посчитан CV R²
    if is_target_logged:
        print(f"CV R² (в ЛОГАРИФМАХ): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    else:
        print(f"CV R² (в долларах): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # 6. Вывод коэффициентов
    if hasattr(model, 'named_steps'):
        trained_model = model.named_steps['model']
    else:
        trained_model = model 

    if hasattr(trained_model, 'coef_'):
        coef_df = pd.DataFrame({
            'Признак': X.columns,
            'Коэффициент': trained_model.coef_
        }).sort_values('Коэффициент', ascending=False)

        print("\nКоэффициенты модели (после масштабирования):")
        print(coef_df.to_string(index=False, float_format=lambda x: f'{x:+.4f}'))
