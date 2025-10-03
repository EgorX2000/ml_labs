"""
custom_models.py

Реализация простых моделей регрессии:
- LinearRegressionNormal: аналитическое решение (нормальное уравнение, с опцией L2-регуляризации)
- LinearRegressionGD: градиентный спуск (батч)
- LinearRegressionSGD: стохастический градиентный спуск
- RandomForestWrapper: обёртка над sklearn.ensemble.RandomForestRegressor (удобство использования)
- GradientBoostingWrapper: обёртка над xgboost.XGBRegressor или sklearn.ensemble.GradientBoostingRegressor

Интерфейс совместим с numpy- и pandas-матрицами: методы fit(X, y), predict(X).

Автор: сгенерировано ChatGPT
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def _ensure_numpy(X):
    if hasattr(X, "values"):
        X = X.values
    return np.asarray(X, dtype=float)


class LinearRegressionNormal:
    """Линейная регрессия — аналитическое решение (нормальное уравнение).

    Параметры:
        l2: float — коэффициент L2-регуляризации (Ridge). По умолчанию 0 (нет регуляризации).
    """

    def __init__(self, l2: float = 0.0):
        self.l2 = float(l2)
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = _ensure_numpy(X)
        y = _ensure_numpy(y).reshape(-1, 1)
        n_samples, n_features = X.shape

        # добавляем столбец единиц для свободного члена
        X_design = np.hstack([np.ones((n_samples, 1)), X])

        # матрица регуляризации: не штрафуем bias
        I = np.eye(n_features + 1)
        I[0, 0] = 0.0
        reg = self.l2 * I

        # нормальное уравнение с регуляризацией: w = (X^T X + reg)^(-1) X^T y
        A = X_design.T @ X_design + reg
        b = X_design.T @ y

        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(A) @ b

        self.intercept_ = float(w[0, 0])
        self.coef_ = w[1:, 0]
        return self

    def predict(self, X):
        X = _ensure_numpy(X)
        return (X @ self.coef_) + self.intercept_


class LinearRegressionGD:
    """Линейная регрессия, обучаемая батч-градиентным спуском.

    Параметры:
        lr: float — скорость обучения (eta)
        n_iter: int — число итераций
        l2: float — L2 регуляризация
        tol: float — критерий остановки по изменению loss
        verbose: bool — печатать прогресс
    """

    def __init__(self, lr=0.01, n_iter=1000, l2=0.0, tol=1e-6, verbose=False):
        self.lr = float(lr)
        self.n_iter = int(n_iter)
        self.l2 = float(l2)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self.coef_ = None
        self.intercept_ = None
        self.loss_history_ = []

    def fit(self, X, y):
        X = _ensure_numpy(X)
        y = _ensure_numpy(y).reshape(-1, 1)
        n_samples, n_features = X.shape

        # инициализация
        rng = np.random.RandomState(0)
        w = rng.normal(scale=0.01, size=(n_features, 1))
        b = 0.0

        for it in range(self.n_iter):
            # предсказание
            y_pred = X @ w + b
            error = y_pred - y

            # градиенты
            grad_w = (2.0 / n_samples) * (X.T @ error) + 2.0 * self.l2 * w
            grad_b = (2.0 / n_samples) * np.sum(error)

            # обновление параметров
            w = w - self.lr * grad_w
            b = b - self.lr * grad_b

            # loss (MSE)
            loss = float(np.mean(error ** 2) + self.l2 * np.sum(w ** 2))
            self.loss_history_.append(loss)

            if self.verbose and it % max(1, self.n_iter // 10) == 0:
                print(f"iter={it}, loss={loss:.6f}")

            # критерий остановки
            if it > 0 and abs(self.loss_history_[-2] - self.loss_history_[-1]) < self.tol:
                if self.verbose:
                    print(f"Stopped at iter {it}, delta loss < tol")
                break

        self.coef_ = w.flatten()
        self.intercept_ = float(b)
        return self

    def predict(self, X):
        X = _ensure_numpy(X)
        return (X @ self.coef_) + self.intercept_


class LinearRegressionSGD:
    """Линейная регрессия, обучаемая стохастическим градиентным спуском.

    Параметры:
        lr: float — начальная скорость обучения
        n_epochs: int — число проходов по данным
        batch_size: int — размер мини-батча (1 = классический SGD)
        l2: float — L2 регуляризация
        verbose: bool
    """

    def __init__(self, lr=0.01, n_epochs=10, batch_size=32, l2=0.0, shuffle=True, verbose=False, random_state: Optional[int] = None):
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.l2 = float(l2)
        self.shuffle = bool(shuffle)
        self.verbose = bool(verbose)
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.loss_history_ = []

    def fit(self, X, y):
        X = _ensure_numpy(X)
        y = _ensure_numpy(y).reshape(-1, 1)
        n_samples, n_features = X.shape

        rng = np.random.RandomState(self.random_state)
        w = rng.normal(scale=0.01, size=(n_features, 1))
        b = 0.0

        for epoch in range(self.n_epochs):
            # опционально перетасовываем
            if self.shuffle:
                idx = rng.permutation(n_samples)
            else:
                idx = np.arange(n_samples)

            for start in range(0, n_samples, self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                Xb = X[batch_idx]
                yb = y[batch_idx]

                y_pred_b = Xb @ w + b
                error_b = y_pred_b - yb

                grad_w = (2.0 / Xb.shape[0]) * (Xb.T @ error_b) + 2.0 * self.l2 * w
                grad_b = (2.0 / Xb.shape[0]) * np.sum(error_b)

                w = w - self.lr * grad_w
                b = b - self.lr * grad_b

            # loss по всей выборке
            y_pred = X @ w + b
            loss = float(np.mean((y - y_pred) ** 2) + self.l2 * np.sum(w ** 2))
            self.loss_history_.append(loss)

            if self.verbose:
                print(f"epoch={epoch+1}/{self.n_epochs}, loss={loss:.6f}")

        self.coef_ = w.flatten()
        self.intercept_ = float(b)
        return self

    def predict(self, X):
        X = _ensure_numpy(X)
        return (X @ self.coef_) + self.intercept_


# Обёртки над готовыми библиотечными моделями (для удобства)
class RandomForestWrapper:
    def __init__(self, **kwargs):
        try:
            from sklearn.ensemble import RandomForestRegressor
        except Exception as e:
            raise ImportError("scikit-learn is required for RandomForestWrapper") from e
        self.model = RandomForestRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class GradientBoostingWrapper:
    def __init__(self, use_xgboost: bool = True, **kwargs):
        self.use_xgboost = bool(use_xgboost)
        if self.use_xgboost:
            try:
                import xgboost as xgb
            except Exception as e:
                raise ImportError("xgboost is required but not installed") from e
            self.model = xgb.XGBRegressor(**kwargs)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


# Небольшие утилиты
__all__ = [
    "LinearRegressionNormal",
    "LinearRegressionGD",
    "LinearRegressionSGD",
    "RandomForestWrapper",
    "GradientBoostingWrapper",
]


# Пример использования (закомментированный):
# from custom_models import LinearRegressionNormal, LinearRegressionGD, LinearRegressionSGD
# model = LinearRegressionNormal(l2=0.01)
# model.fit(X_train_scaled, y_train)
# preds = model.predict(X_test_scaled)
""
