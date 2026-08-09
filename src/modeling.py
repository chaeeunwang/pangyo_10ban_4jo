"""응답자 정보로 income을 예측하는 모델 선택·평가 Pipeline을 담당한다.

작성 정보:
- 최초 작성자: 왕채은
- 공동 수정자는 이 파일을 변경할 때 아래 형식으로 이력을 추가한다.
- 수정 이력:
  - 2026-08-09 왕채은: 공동 작업용 작성자·수정 이력 형식 추가
  - YYYY-MM-DD 이름: 변경 내용

주요 기능:
- train/test 80/20 분할 후 train IQR 경계를 양쪽에 동일 적용
- 순서형·경력 파생변수와 명목형 원핫인코딩
- Employment·기술 스택 다중선택 문항의 train 기반 multi-hot 변환
- Ridge 기준선·Random Forest 5-Fold 교차검증
- Random Forest의 제한된 RandomizedSearchCV와 holdout 최종 평가
- 달러 단위 예측 Pipeline, 모델 비교표, permutation importance 저장

외부 test 세트는 IQR 경계 계산·모델 비교·튜닝에 사용하지 않는다. 모든 후보는 목표
income을 log1p로 학습하고 Pipeline.predict에서 달러로 복원하므로 저장된
joblib 사용자는 별도 역변환을 수행할 필요가 없다. 다중선택 어휘와 희귀
선택지 기준도 각 학습 fold 안에서만 학습해 검증 정보 누수를 막는다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict, cast

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import AnalysisError
from src.preprocessing import EXPERIENCE_DERIVED_COLUMNS, ORDINAL_SCORE_COLUMNS


class ModelMetrics(TypedDict):
    """holdout 평가 지표와 train 교차검증·모델 선택 정보."""

    selected_model: str
    cv_folds: int
    cv_log_rmse_mean: float
    cv_log_rmse_std: float
    cv_log_r2_mean: float
    cv_log_r2_std: float
    cv_mae_usd_mean: float
    tuning_candidate_count: int
    mae_usd: float
    rmse_usd: float
    r2: float
    log_rmse: float
    log_r2: float
    full_test_mae_usd: float
    full_test_rmse_usd: float
    full_test_r2: float
    full_test_log_rmse: float
    full_test_log_r2: float
    train_rows_before_outlier_filter: int
    train_rows: int
    test_rows_before_outlier_filter: int
    test_rows: int
    train_salary_lower_bound: float
    train_salary_upper_bound: float


# 명목형은 숫자 크기에 의미가 없으므로 원핫인코딩한다. Employment는
# 조합 문자열 하나로 취급하지 않도록 아래 다중선택 목록으로 이동했다.
CATEGORICAL_FEATURES = [
    "Age",
    "Country",
    "DevType",
    "RemoteWork",
    "EdLevel",
    "OrgSize",
    "Industry",
    "ICorPM",
]
MULTISELECT_FEATURES = [
    "Employment",
    "CodingActivities",
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith",
    "PlatformHaveWorkedWith",
    "ProfessionalTech",
]
MODEL_ADDITIONAL_NUMERIC_FEATURES = [
    *ORDINAL_SCORE_COLUMNS,
    *EXPERIENCE_DERIVED_COLUMNS,
]
CV_FOLDS = 5
RANDOM_STATE = 42


class MultiSelectEncoder(TransformerMixin, BaseEstimator):
    """세미콜론 다중선택 문항을 희소 multi-hot 행렬로 변환한다.

    각 열의 선택지 빈도는 ``fit`` 데이터에서만 계산한다. 최소 빈도보다
    적거나 예측 때 처음 등장한 선택지는 열별 ``__INFREQUENT__``로 묶고,
    문항 결측은 ``__MISSING__``으로 분리한다. 따라서 조합 문자열을 하나의
    범주로 보는 원핫 방식보다 개별 선택지의 신호를 보존하면서 차원 폭발과
    미지 범주 오류를 막는다.
    """

    def __init__(self, min_frequency: int = 50) -> None:
        self.min_frequency = min_frequency

    @staticmethod
    def _as_frame(values: Any, columns: list[str] | None = None) -> pd.DataFrame:
        """ColumnTransformer의 DataFrame·ndarray 입력을 동일 형태로 맞춘다."""
        if isinstance(values, pd.DataFrame):
            return values.copy()
        array = np.asarray(values, dtype=object)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        names = columns or [f"x{index}" for index in range(array.shape[1])]
        return pd.DataFrame(array, columns=names)

    @staticmethod
    def _tokens(value: Any) -> set[str]:
        """한 응답의 중복·공백 선택지를 제거하고 결측 표식을 반환한다."""
        if pd.isna(value) or not str(value).strip():
            return {"__MISSING__"}
        return {token.strip() for token in str(value).split(";") if token.strip()}

    def fit(self, x: Any, y: Any = None) -> MultiSelectEncoder:
        """학습 행에서 열별 공통 선택지 어휘와 출력 열 순서를 확정한다."""
        if self.min_frequency < 1:
            raise ValueError("min_frequency는 1 이상이어야 합니다.")
        frame = self._as_frame(x)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.vocabularies_: dict[str, tuple[str, ...]] = {}
        output_names: list[str] = []
        for column in frame.columns:
            counts: dict[str, int] = {}
            for value in cast(pd.Series, frame[column]):
                for token in self._tokens(value):
                    counts[token] = counts.get(token, 0) + 1
            common = sorted(
                token
                for token, count in counts.items()
                if count >= self.min_frequency and token != "__MISSING__"
            )
            vocabulary = tuple([*common, "__MISSING__", "__INFREQUENT__"])
            self.vocabularies_[str(column)] = vocabulary
            output_names.extend(f"{column}={token}" for token in vocabulary)
        self.output_feature_names_ = np.asarray(output_names, dtype=object)
        return self

    def transform(self, x: Any) -> sparse.csr_matrix:
        """학습된 열별 어휘에 맞춰 0/1 희소 multi-hot 행렬을 만든다."""
        if not hasattr(self, "vocabularies_"):
            raise ValueError("MultiSelectEncoder를 fit한 뒤 transform해야 합니다.")
        columns = [str(name) for name in self.feature_names_in_]
        frame = self._as_frame(x, columns)
        if list(map(str, frame.columns)) != columns:
            frame.columns = columns
        row_indexes: list[int] = []
        column_indexes: list[int] = []
        offset = 0
        for column in columns:
            vocabulary = self.vocabularies_[column]
            index_by_token = {token: index for index, token in enumerate(vocabulary)}
            for row_index, value in enumerate(cast(pd.Series, frame[column])):
                encoded: set[int] = set()
                for token in self._tokens(value):
                    encoded.add(
                        index_by_token.get(token, index_by_token["__INFREQUENT__"])
                    )
                for local_index in encoded:
                    row_indexes.append(row_index)
                    column_indexes.append(offset + local_index)
            offset += len(vocabulary)
        data = np.ones(len(row_indexes), dtype=np.float64)
        return sparse.csr_matrix(
            (data, (row_indexes, column_indexes)),
            shape=(len(frame), len(self.output_feature_names_)),
        )

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """ColumnTransformer와 산출물 해석에 사용할 multi-hot 열 이름을 반환한다."""
        if not hasattr(self, "output_feature_names_"):
            raise ValueError("MultiSelectEncoder를 fit한 뒤 열 이름을 조회해야 합니다.")
        return self.output_feature_names_.copy()


def _log_rmse(y_true: Any, y_pred: Any) -> float:
    """달러 예측을 log1p로 변환해 고소득 극단값에 덜 민감한 RMSE를 계산한다."""
    true_log = np.log1p(np.maximum(np.asarray(y_true, dtype=float), 0.0))
    pred_log = np.log1p(np.maximum(np.asarray(y_pred, dtype=float), 0.0))
    return float(mean_squared_error(true_log, pred_log) ** 0.5)


def _log_r2(y_true: Any, y_pred: Any) -> float:
    """달러 예측과 실제값을 log1p 척도에서 비교한 R2를 계산한다."""
    true_log = np.log1p(np.maximum(np.asarray(y_true, dtype=float), 0.0))
    pred_log = np.log1p(np.maximum(np.asarray(y_pred, dtype=float), 0.0))
    return float(r2_score(true_log, pred_log))


LOG_RMSE_SCORER = make_scorer(_log_rmse, greater_is_better=False)
LOG_R2_SCORER = make_scorer(_log_r2)
CV_SCORING = {
    "neg_log_rmse": LOG_RMSE_SCORER,
    "log_r2": LOG_R2_SCORER,
    "neg_mae_usd": "neg_mean_absolute_error",
    "neg_rmse_usd": "neg_root_mean_squared_error",
    "r2_usd": "r2",
}


def save_selected_model_columns(numeric_features: list[str], processed_dir: Path) -> Path:
    """최종 모델의 수치형·명목형·다중선택 입력 계약을 CSV로 저장한다."""
    if not numeric_features:
        raise AnalysisError("저장할 수치형 모델 컬럼이 없습니다.")
    processed_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.DataFrame({
        "role": (
            ["numeric"] * len(numeric_features)
            + ["categorical_onehot"] * len(CATEGORICAL_FEATURES)
            + ["multiselect_multihot"] * len(MULTISELECT_FEATURES)
        ),
        "column": numeric_features + CATEGORICAL_FEATURES + MULTISELECT_FEATURES,
    })
    path = processed_dir / "selected_model_columns.csv"
    try:
        selected.to_csv(path, index=False)
    except OSError as exc:
        raise AnalysisError("모델 선택 컬럼표를 저장하지 못했습니다.") from exc
    return path


def _build_preprocessor(
    numeric_features: list[str], force_dense: bool = False
) -> ColumnTransformer:
    """모든 후보에 동일한 결측·범주 처리 계약을 가진 전처리를 만든다."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        # 분기 문항의 결측 자체가 응답자 특성일 수 있어 별도 범주로 보존한다.
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=20,
            max_categories=40,
        )),
    ])
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("multiselect", MultiSelectEncoder(min_frequency=50), MULTISELECT_FEATURES),
        ],
        # dense 입력만 받는 후보를 다시 실험할 수 있도록 선택지를 유지한다.
        # Random Forest와 Ridge는 기본 희소 출력을 사용해 메모리를 아낀다.
        sparse_threshold=0.0 if force_dense else 0.3,
        verbose_feature_names_out=False,
    )


def _build_model_pipeline(
    numeric_features: list[str], regressor: Any, force_dense: bool = False
) -> Pipeline:
    """전처리와 log1p 목표 변환을 포함한 달러 예측 Pipeline을 만든다."""
    target_regressor = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )
    return Pipeline([
        ("preprocessor", _build_preprocessor(numeric_features, force_dense)),
        ("model", target_regressor),
    ])


def _build_candidates(numeric_features: list[str]) -> dict[str, Pipeline]:
    """선형 기준선과 요청된 Random Forest를 동일 Pipeline 계약으로 만든다."""
    return {
        "Ridge": _build_model_pipeline(numeric_features, Ridge(alpha=1.0)),
        "RandomForest": _build_model_pipeline(
            numeric_features,
            RandomForestRegressor(
                # 깊이와 leaf 크기를 제한해 희소 설문 특성의 과적합과
                # 5-Fold 탐색 시간을 함께 제어한다.
                n_estimators=120,
                max_depth=24,
                max_features=0.6,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
    }


def _target_bins(target: pd.Series, bin_count: int = 10) -> pd.Series:
    """회귀 target을 분위 구간으로 나눠 split마다 소득 분포를 안정화한다."""
    bins = cast(
        pd.Series,
        pd.qcut(target.rank(method="first"), q=min(bin_count, len(target)), labels=False),
    )
    if bins.nunique() < 2:
        raise AnalysisError("소득 분위 기반 분할을 만들 수 없습니다.")
    return bins.astype(int)


def _cv_record(name: str, scores: dict[str, Any], tuned: bool) -> dict[str, Any]:
    """cross_validate 또는 탐색 결과 배열을 비교 가능한 한 행으로 요약한다."""
    return {
        "model": name,
        "tuned": tuned,
        "cv_folds": CV_FOLDS,
        "cv_log_rmse_mean": -float(np.mean(scores["test_neg_log_rmse"])),
        "cv_log_rmse_std": float(np.std(scores["test_neg_log_rmse"], ddof=1)),
        "cv_log_r2_mean": float(np.mean(scores["test_log_r2"])),
        "cv_log_r2_std": float(np.std(scores["test_log_r2"], ddof=1)),
        "cv_mae_usd_mean": -float(np.mean(scores["test_neg_mae_usd"])),
        "cv_mae_usd_std": float(np.std(scores["test_neg_mae_usd"], ddof=1)),
        "cv_rmse_usd_mean": -float(np.mean(scores["test_neg_rmse_usd"])),
        "cv_r2_usd_mean": float(np.mean(scores["test_r2_usd"])),
    }


def _search_space(model_name: str) -> dict[str, list[Any]]:
    """과제 실행시간을 제한하면서 핵심 복잡도만 탐색하는 후보 공간을 반환한다."""
    if model_name == "Ridge":
        return {"model__regressor__alpha": [0.05, 0.2, 1.0, 5.0, 20.0, 80.0]}
    if model_name == "RandomForest":
        return {
            "model__regressor__n_estimators": [100, 160, 240],
            "model__regressor__max_features": [0.35, 0.55, 0.75],
            "model__regressor__min_samples_leaf": [2, 4, 8],
            "model__regressor__max_depth": [16, 24, 32],
            "model__regressor__max_samples": [0.75, 0.9, None],
        }
    raise AnalysisError(f"튜닝 공간이 정의되지 않은 모델입니다: {model_name}")


def _tuning_record(
    name: str, search: RandomizedSearchCV, best_index: int
) -> dict[str, Any]:
    """RandomizedSearchCV 최상위 조합의 fold별 지표를 비교표 형식으로 바꾼다."""
    results = search.cv_results_
    scores = {
        metric: np.asarray([
            results[f"split{fold}_{metric}"][best_index]
            for fold in range(CV_FOLDS)
        ])
        for metric in (
            "test_neg_log_rmse",
            "test_log_r2",
            "test_neg_mae_usd",
            "test_neg_rmse_usd",
            "test_r2_usd",
        )
    }
    return _cv_record(f"Tuned {name}", scores, tuned=True)


def _run_model_selection(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, str, int]:
    """학습 세트 안에서 후보 비교·튜닝 후 최종 Pipeline을 refit한다."""
    folds = list(
        StratifiedKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
        ).split(x_train, _target_bins(y_train))
    )
    candidates = _build_candidates(numeric_features)
    records: list[dict[str, Any]] = []
    for name, candidate in candidates.items():
        print(f"[모델 선택][START] {name} {CV_FOLDS}-Fold 교차검증", flush=True)
        try:
            scores = cross_validate(
                candidate,
                x_train,
                y_train,
                cv=folds,
                scoring=CV_SCORING,
                n_jobs=-1,
                pre_dispatch=2,
                error_score="raise",
            )
            records.append(_cv_record(name, scores, tuned=False))
        except (TypeError, ValueError) as exc:
            print(f"[모델 선택][FAIL] {name} 교차검증", flush=True)
            raise AnalysisError(f"{name} 5-Fold 교차검증에 실패했습니다.") from exc
        print(
            f"[모델 선택][SUCCESS] {name} CV log RMSE="
            f"{records[-1]['cv_log_rmse_mean']:.4f}",
            flush=True,
        )

    comparison = pd.DataFrame(records).sort_values(
        "cv_log_rmse_mean", ignore_index=True
    )
    # 이번 실험의 목적은 Random Forest 자체를 검증하는 것이다. Ridge가
    # 더 높은 CV 점수를 보이더라도 모델 종류를 바꾸지 않고 RF를 튜닝한다.
    tuning_target = "RandomForest"
    tuning_space = _search_space(tuning_target)
    candidate_count = min(8, int(np.prod([len(values) for values in tuning_space.values()])))
    search = RandomizedSearchCV(
        estimator=clone(candidates[tuning_target]),
        param_distributions=tuning_space,
        n_iter=candidate_count,
        scoring=CV_SCORING,
        refit="neg_log_rmse",
        cv=folds,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        pre_dispatch=2,
        error_score="raise",
        return_train_score=False,
    )
    print(
        f"[모델 튜닝][START] {tuning_target} {candidate_count}개 조합 x {CV_FOLDS}-Fold",
        flush=True,
    )
    try:
        search.fit(x_train, y_train)
    except (TypeError, ValueError) as exc:
        print(f"[모델 튜닝][FAIL] {tuning_target}", flush=True)
        raise AnalysisError(
            f"{tuning_target} 하이퍼파라미터 탐색에 실패했습니다."
        ) from exc

    best_index = int(search.best_index_)
    tuned_record = _tuning_record(tuning_target, search, best_index)
    print(
        f"[모델 튜닝][SUCCESS] {tuning_target} CV log RMSE="
        f"{tuned_record['cv_log_rmse_mean']:.4f}",
        flush=True,
    )
    comparison = pd.concat(
        [comparison, pd.DataFrame([tuned_record])], ignore_index=True
    ).sort_values("cv_log_rmse_mean", ignore_index=True)

    # 제한된 무작위 탐색이 기본값보다 나쁠 수도 있다. 같은 fold의 CV 결과를
    # 비교해 실제로 나아진 경우에만 튜닝 모델을 채택한다.
    baseline_score = float(
        comparison.loc[
            comparison["model"] == tuning_target, "cv_log_rmse_mean"
        ].iloc[0]
    )
    tuned_score = float(tuned_record["cv_log_rmse_mean"])
    if tuned_score <= baseline_score:
        final_pipeline = cast(Pipeline, search.best_estimator_)
        selected_name = f"Tuned {tuning_target}"
    else:
        final_pipeline = cast(Pipeline, clone(candidates[tuning_target]))
        final_pipeline.fit(x_train, y_train)
        selected_name = tuning_target

    result_columns = [
        "rank_test_neg_log_rmse",
        "mean_test_neg_log_rmse",
        "std_test_neg_log_rmse",
        "mean_test_log_r2",
        "mean_test_neg_mae_usd",
        "mean_test_neg_rmse_usd",
        "mean_test_r2_usd",
        "params",
    ]
    tuning_results = pd.DataFrame(search.cv_results_)[result_columns].copy()
    tuning_results["params"] = tuning_results["params"].map(
        lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
    )
    tuning_results = tuning_results.sort_values(
        "rank_test_neg_log_rmse", ignore_index=True
    )
    return final_pipeline, comparison, tuning_results, selected_name, candidate_count


def _raw_feature_importance(
    pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """holdout 일부에서 log R2 감소량 기반 raw 입력 permutation importance를 계산한다."""
    if len(x_test) > 2500:
        sampled_x = x_test.sample(n=2500, random_state=RANDOM_STATE)
        sampled_y = y_test.loc[sampled_x.index]
    else:
        sampled_x, sampled_y = x_test, y_test
    try:
        result = permutation_importance(
            pipeline,
            sampled_x,
            sampled_y,
            scoring=LOG_R2_SCORER,
            n_repeats=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except (TypeError, ValueError) as exc:
        raise AnalysisError("최종 모델의 permutation importance 계산에 실패했습니다.") from exc
    return pd.DataFrame({
        "feature": sampled_x.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
        "metric": "decrease_in_log_r2",
    }).sort_values("importance_mean", ascending=False, ignore_index=True)


def train_salary_model(
    df: pd.DataFrame,
    numeric_candidates: list[str],
    processed_dir: Path,
    models_dir: Path,
) -> tuple[
    Pipeline,
    ModelMetrics,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    pd.DataFrame,
]:
    """후보 비교·튜닝을 거쳐 holdout income 성능을 평가하고 저장한다.

    먼저 원본 유효 income 모집단을 80/20으로 나눈다. train target에서만
    IQR 경계를 계산해 train·test에 같은 범위를 적용하고, 남은 train 내부에서
    5-Fold Ridge 기준선 비교와 Random Forest 최대 8개 조합 튜닝을 수행한다.
    정제 test를 기본 평가 모집단으로 사용하되, 전체 test 지표도 비교용으로
    함께 저장해 평가 범위에 따른 성능 차이를 숨기지 않는다.

    반환값은 최종 Pipeline, holdout/CV 지표, raw 입력 중요도, train 수치형
    상관표, 최종 수치형 컬럼, 모델 비교표다.
    """
    if not numeric_candidates:
        raise AnalysisError("모델에 사용할 수치형 후보 변수가 없습니다.")
    numeric_features = list(dict.fromkeys([
        *numeric_candidates,
        *MODEL_ADDITIONAL_NUMERIC_FEATURES,
    ]))
    features = numeric_features + CATEGORICAL_FEATURES + MULTISELECT_FEATURES
    missing_columns = [column for column in features + ["ConvertedCompYearly"] if column not in df]
    if missing_columns:
        raise AnalysisError("income 모델에 필요한 열이 없습니다: " + ", ".join(missing_columns))
    source = cast(pd.DataFrame, df[features + ["ConvertedCompYearly"]]).copy()
    salary = cast(pd.Series, source["ConvertedCompYearly"])
    valid = cast(pd.Series, salary.notna() & (salary > 0))
    model_df = source.loc[valid].copy()
    if len(model_df) < 100:
        raise AnalysisError("income 모델에 사용할 유효 응답이 최소 100개 필요합니다.")

    x = cast(pd.DataFrame, model_df[features])
    y_usd = cast(pd.Series, model_df["ConvertedCompYearly"]).astype(float)
    try:
        x_train, x_test_full, y_train_usd, y_test_full_usd = train_test_split(
            x,
            y_usd,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
    except ValueError as exc:
        raise AnalysisError("income 모델의 train/test 데이터를 분할하지 못했습니다.") from exc

    # IQR 경계는 train target에서만 계산한다. 정제된 일반 급여 구간을
    # train·test에 동일하게 적용하되, 전체 test는 별도 진단 지표용으로 보존한다.
    first_quartile = float(y_train_usd.quantile(0.25))
    third_quartile = float(y_train_usd.quantile(0.75))
    interquartile_range = third_quartile - first_quartile
    lower_bound = first_quartile - 1.5 * interquartile_range
    upper_bound = third_quartile + 1.5 * interquartile_range
    train_inlier = cast(
        pd.Series, y_train_usd.between(lower_bound, upper_bound)
    )
    test_inlier = cast(
        pd.Series, y_test_full_usd.between(lower_bound, upper_bound)
    )
    train_rows_before_filter = len(x_train)
    test_rows_before_filter = len(x_test_full)
    x_train = x_train.loc[train_inlier].copy()
    y_train_usd = y_train_usd.loc[train_inlier].copy()
    x_test = x_test_full.loc[test_inlier].copy()
    y_test_usd = y_test_full_usd.loc[test_inlier].copy()
    if len(x_test) < 100:
        raise AnalysisError("IQR 정제 후 holdout test 응답이 최소 100개 필요합니다.")

    y_train_log = np.log1p(y_train_usd)
    correlations = (
        cast(pd.DataFrame, x_train[numeric_features])
        .apply(lambda series: pd.to_numeric(series, errors="coerce"))
        .corrwith(cast(pd.Series, y_train_log))
        .dropna()
        .rename("pearson_r")
        .to_frame()
        .assign(absolute_r=lambda frame: frame["pearson_r"].abs())
        .sort_values("absolute_r", ascending=False)
        .reset_index(names="feature")
    )
    if correlations.empty:
        raise AnalysisError("train 세트에서 income과 상관분석할 수치형 변수가 없습니다.")
    correlations["used_by_model"] = True

    pipeline, comparison, tuning_results, selected_name, candidate_count = (
        _run_model_selection(x_train, y_train_usd, numeric_features)
    )
    try:
        prediction_full_usd = np.maximum(pipeline.predict(x_test_full), 0.0)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("최종 income Pipeline의 holdout 예측에 실패했습니다.") from exc
    prediction_usd = prediction_full_usd[test_inlier.to_numpy()]
    prediction_log = np.log1p(prediction_usd)
    prediction_full_log = np.log1p(prediction_full_usd)
    selected_cv = comparison.loc[comparison["model"] == selected_name].iloc[0]
    metrics = ModelMetrics(
        selected_model=selected_name,
        cv_folds=CV_FOLDS,
        cv_log_rmse_mean=float(selected_cv["cv_log_rmse_mean"]),
        cv_log_rmse_std=float(selected_cv["cv_log_rmse_std"]),
        cv_log_r2_mean=float(selected_cv["cv_log_r2_mean"]),
        cv_log_r2_std=float(selected_cv["cv_log_r2_std"]),
        cv_mae_usd_mean=float(selected_cv["cv_mae_usd_mean"]),
        tuning_candidate_count=candidate_count,
        mae_usd=float(mean_absolute_error(y_test_usd, prediction_usd)),
        rmse_usd=float(mean_squared_error(y_test_usd, prediction_usd) ** 0.5),
        r2=float(r2_score(y_test_usd, prediction_usd)),
        log_rmse=float(mean_squared_error(np.log1p(y_test_usd), prediction_log) ** 0.5),
        log_r2=float(r2_score(np.log1p(y_test_usd), prediction_log)),
        full_test_mae_usd=float(
            mean_absolute_error(y_test_full_usd, prediction_full_usd)
        ),
        full_test_rmse_usd=float(
            mean_squared_error(y_test_full_usd, prediction_full_usd) ** 0.5
        ),
        full_test_r2=float(r2_score(y_test_full_usd, prediction_full_usd)),
        full_test_log_rmse=float(
            mean_squared_error(
                np.log1p(y_test_full_usd), prediction_full_log
            ) ** 0.5
        ),
        full_test_log_r2=float(
            r2_score(np.log1p(y_test_full_usd), prediction_full_log)
        ),
        train_rows_before_outlier_filter=train_rows_before_filter,
        train_rows=len(x_train),
        test_rows_before_outlier_filter=test_rows_before_filter,
        test_rows=len(x_test),
        train_salary_lower_bound=lower_bound,
        train_salary_upper_bound=upper_bound,
    )
    importance = _raw_feature_importance(pipeline, x_test, y_test_usd)

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    try:
        pd.DataFrame([metrics]).to_csv(processed_dir / "model_metrics.csv", index=False)
        correlations.to_csv(processed_dir / "model_numeric_correlations.csv", index=False)
        comparison.to_csv(processed_dir / "model_comparison_cv.csv", index=False)
        tuning_results.to_csv(processed_dir / "model_tuning_results.csv", index=False)
        importance.to_csv(processed_dir / "salary_feature_importance.csv", index=False)
        joblib.dump(pipeline, models_dir / "survey_income_prediction_pipeline.joblib")
    except (OSError, TypeError, ValueError) as exc:
        raise AnalysisError("모델 결과표 또는 joblib 파일을 저장하지 못했습니다.") from exc
    return pipeline, metrics, importance, correlations, numeric_features, comparison
