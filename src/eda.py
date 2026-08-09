"""원본·정제 EDA와 급여 관련 변수 탐색을 담당한다.

작성 정보:
- 최초 작성자: 왕채은
- 공동 수정자는 이 파일을 변경할 때 아래 형식으로 이력을 추가한다.
- 수정 이력:
  - 2026-08-09 왕채은: 공동 작업용 작성자·수정 이력 형식 추가
  - YYYY-MM-DD 이름: 변경 내용

주요 기능:
- 정제 전후의 shape·dtype·결측률·고유값·급여 분포 저장
- log_salary와 수치형 후보의 Pearson 상관계수 Top N 선정
- Country·DevType·RemoteWork·EdLevel별 중앙값·평균·분위수 비교
- 순서형 응답의 Spearman 상관과 명목형 응답의 η²·Kruskal-Wallis 검정
- Employment 다중선택을 multi-hot으로 분리한 income 관련성 비교

`preprocessing.clean_data`의 정제 결과를 받으며, 상관 Top N은 EDA 요약에만,
범주형 요약표는 Plotly 시각화의 입력으로 전달된다.
범주형 값에 임의 숫자 순서를 부여하지 않으며, 소표본 범주는 순위
왜곡을 줄이기 위해 제외한다.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import kruskal, pointbiserialr, spearmanr

from src.data_loader import AnalysisError
from src.preprocessing import (
    AGE_ORDER,
    EDUCATION_ORDER,
    KNOWLEDGE_SCORE_COLUMNS,
    ORDINAL_SCORE_COLUMNS,
    ORG_SIZE_ORDER,
)

NUMERIC_CANDIDATES = [
    "YearsCode_num",
    "YearsCodePro_num",
    "WorkExp_num",
    *KNOWLEDGE_SCORE_COLUMNS,
]
CATEGORICAL_CANDIDATES = ["Country", "DevType", "RemoteWork", "EdLevel"]
NOMINAL_ASSOCIATION_FEATURES = ["Country", "DevType", "RemoteWork", "Industry", "ICorPM"]
ORDINAL_TREND_SPECS = {
    "Age_order": ("Age", {value: label for label, value in AGE_ORDER.items()}),
    "EdLevel_order": (
        "Education level", {value: label for label, value in EDUCATION_ORDER.items()}
    ),
    "OrgSize_order": (
        "Organization size", {value: label for label, value in ORG_SIZE_ORDER.items()}
    ),
}

# JobSatPoints는 각 항목이 직무 만족도에 기여하는 정도를 총 100점으로
# 배분한 수치형 문항이다. 모델 선정 후보와는 분리하고, 탐색적 히트맵에만
# 포함해 경력·만족 요인·로그 급여의 관계를 한번에 비교한다.
JOB_SAT_POINT_COLUMNS = [
    "JobSatPoints_1",
    "JobSatPoints_4",
    "JobSatPoints_5",
    "JobSatPoints_6",
    "JobSatPoints_7",
    "JobSatPoints_8",
    "JobSatPoints_9",
    "JobSatPoints_10",
    "JobSatPoints_11",
]

EXTENDED_NUMERIC_COLUMNS = [
    "YearsCode_num",
    "YearsCodePro_num",
    "WorkExp_num",
    "JobSat",
    *JOB_SAT_POINT_COLUMNS,
    *KNOWLEDGE_SCORE_COLUMNS,
    "log_salary",
]

HEATMAP_LABELS = {
    "YearsCode_num": "Years coding",
    "YearsCodePro_num": "Professional years",
    "WorkExp_num": "Work experience",
    "JobSat": "Job satisfaction",
    "JobSatPoints_1": "Team strategy",
    "JobSatPoints_4": "Open source",
    "JobSatPoints_5": "Security",
    "JobSatPoints_6": "Code quality",
    "JobSatPoints_7": "New technology",
    "JobSatPoints_8": "Architecture",
    "JobSatPoints_9": "Tool expertise",
    "JobSatPoints_10": "Hardware",
    "JobSatPoints_11": "Network / observability",
    **{
        column: f"Knowledge {number}"
        for number, column in enumerate(KNOWLEDGE_SCORE_COLUMNS, start=1)
    },
    "log_salary": "Log annual income",
}


def _column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """열별 dtype·결측·고유값을 표로 만들어 원본과 정제 상태를 동일 기준으로 비교한다."""
    rows = len(df)
    records: list[dict[str, Any]] = []
    for column in df.columns:
        series = cast(pd.Series, df[column])
        missing = int(series.isna().sum())
        records.append({
            "column": column, "dtype": str(series.dtype), "missing_count": missing,
            "missing_rate_pct": (missing / rows * 100) if rows else 0.0,
            "unique_count": int(series.nunique(dropna=True)),
        })
    return pd.DataFrame(records).sort_values("missing_rate_pct", ascending=False, ignore_index=True)


def analyze_raw_data(df: pd.DataFrame, processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """정제 전 행·열, 타입, 결측률, 중복, 원본 연봉 분포를 저장한다."""
    # pandas-stubs는 to_numeric의 반환형을 광범위하게 정의하지만 입력은
    # 단일 Series이므로 외부 라이브러리 경계에서 타입을 명확히 좁힌다.
    salary = cast(
        pd.Series,
        pd.to_numeric(cast(pd.Series, df["ConvertedCompYearly"]), errors="coerce"),
    )
    valid_mask = cast(pd.Series, salary.notna() & (salary > 0))
    valid_salary = cast(pd.Series, salary.loc[valid_mask])
    overview = pd.DataFrame({
        "item": ["rows", "columns", "duplicate_response_ids", "valid_salary_rows", "salary_missing_rows"],
        "value": [len(df), df.shape[1], int(df.duplicated(subset=["ResponseId"]).sum()),
                  len(valid_salary), int(salary.isna().sum())],
    })
    profile = _column_profile(df)
    salary_summary = valid_salary.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).to_frame("value")
    try:
        overview.to_csv(processed_dir / "raw_dataset_overview.csv", index=False)
        profile.to_csv(processed_dir / "raw_column_profile.csv", index=False)
        salary_summary.to_csv(processed_dir / "raw_salary_summary.csv", index=True)
    except OSError as exc:
        raise AnalysisError("원본 EDA 결과를 저장하지 못했습니다.") from exc
    return overview, profile


def analyze_cleaned_data(df: pd.DataFrame, processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """정제 후 기술통계와 열별 결측 상태를 저장한다."""
    numeric = NUMERIC_CANDIDATES + ORDINAL_SCORE_COLUMNS + ["ConvertedCompYearly", "log_salary"]
    descriptive = cast(pd.DataFrame, df[numeric]).describe().T
    descriptive["median"] = cast(pd.DataFrame, df[numeric]).median()
    profile = _column_profile(cast(pd.DataFrame, df[numeric + CATEGORICAL_CANDIDATES]))
    try:
        descriptive.to_csv(processed_dir / "cleaned_descriptive_statistics.csv", index=True)
        profile.to_csv(processed_dir / "cleaned_column_profile.csv", index=False)
    except OSError as exc:
        raise AnalysisError("정제 EDA 결과를 저장하지 못했습니다.") from exc
    return descriptive, profile


def build_extended_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """경력·직무 만족도·로그 급여의 확장 Pearson 상관행렬을 만든다.

    수치형으로 파싱하지 못한 값은 NaN으로 두고, DataFrame.corr의 변수 쌍별
    완전 사례 계산을 사용한다. 축 라벨은 공식 설문 스키마의 의미를
    축약한 이름으로 변환해 그래프의 가독성을 높인다.
    """
    missing = [column for column in EXTENDED_NUMERIC_COLUMNS if column not in df.columns]
    if missing:
        raise AnalysisError("확장 상관행렬에 필요한 열이 없습니다: " + ", ".join(missing))
    numeric = cast(
        pd.DataFrame,
        cast(pd.DataFrame, df[EXTENDED_NUMERIC_COLUMNS]).apply(
            lambda series: pd.to_numeric(series, errors="coerce")
        ),
    )
    # 공식 문항은 만족 요인에 총 100점을 배분하도록 요구한다. 실제
    # 응답에는 합계가 0이거나 100을 크게 넘는 값도 있어, 이를 포함하면
    # 모든 항목이 함께 커지는 가짜 양의 상관이 생긴다. 따라서 모든 점수가
    # 존재하고 부동소수점 오차를 감안해 합계가 100인 행만 사용한다.
    point_total = cast(pd.Series, numeric[JOB_SAT_POINT_COLUMNS].sum(
        axis=1, min_count=len(JOB_SAT_POINT_COLUMNS)
    ))
    valid_allocation = cast(
        pd.Series,
        point_total.notna() & np.isclose(point_total, 100.0, atol=0.01),
    )
    numeric.loc[~valid_allocation, JOB_SAT_POINT_COLUMNS] = np.nan
    matrix = cast(pd.DataFrame, numeric.corr(method="pearson")).rename(
        index=HEATMAP_LABELS, columns=HEATMAP_LABELS
    )
    all_missing_by_column = cast(pd.Series, matrix.isna().all(axis=0))
    if bool(all_missing_by_column.all()):
        raise AnalysisError("확장 상관행렬을 계산할 유효한 숫자형 값이 없습니다.")
    return matrix


def analyze_numeric_correlations(df: pd.DataFrame, processed_dir: Path, top_n: int = 3) -> pd.DataFrame:
    """수치형 후보와 log_salary의 Pearson 상관계수 Top N을 선정한다.

    각 변수와 log_salary가 동시에 있는 행만 사용하며, 양·음의 방향보다
    선형 관련성의 크기를 기준으로 선정하기 위해 절대값으로 정렬한다.
    유효한 결과가 없거나 top_n이 잘못되면 AnalysisError를 발생시킨다.
    """
    if top_n < 1:
        raise AnalysisError("상관계수 Top N은 1 이상이어야 합니다.")
    records: list[dict[str, int | float | str]] = []
    for feature in NUMERIC_CANDIDATES:
        pairs = cast(pd.DataFrame, df[[feature, "log_salary"]]).dropna()
        if len(pairs) < 2:
            continue
        coefficient = float(cast(Any, pairs.corr(method="pearson").iloc[0, 1]))
        if not math.isfinite(coefficient):
            continue
        records.append({"feature": feature, "valid_rows": len(pairs), "pearson_r": coefficient,
                        "absolute_r": abs(coefficient)})
    correlations = pd.DataFrame(records).sort_values("absolute_r", ascending=False, ignore_index=True).head(top_n)
    if correlations.empty:
        raise AnalysisError("log_salary와 상관분석할 수치형 변수가 없습니다.")
    # 히트맵과 Top N이 서로 다른 계산 기준을 쓰지 않도록 동일한
    # Pearson 방법으로 전체 수치형 상관행렬도 함께 저장한다. Pandas corr는
    # 변수 쌍별로 두 값이 모두 있는 행만 사용한다.
    correlation_matrix = build_extended_correlation_matrix(df)
    try:
        correlations.to_csv(processed_dir / "numeric_correlation_top.csv", index=False)
        correlation_matrix.to_csv(processed_dir / "numeric_correlation_matrix.csv", index=True)
    except OSError as exc:
        raise AnalysisError("수치형 상관분석 결과를 저장하지 못했습니다.") from exc
    return correlations


def analyze_categorical_salary(df: pd.DataFrame, processed_dir: Path, min_group_size: int = 20) -> pd.DataFrame:
    """범주별 급여 중앙값·분포 지표를 계산하되 소표본은 제외한다.

    `min_group_size`는 우연한 고연봉 응답 몇 개로 범주 순위가 과대평가되는
    것을 줄이는 최소 표본 기준이다. 반환표는 범주별 표본 수, 중앙값,
    평균, 25·75분위수를 포함하고 Plotly 차트에서 재사용된다.
    """
    if min_group_size < 1:
        raise AnalysisError("범주형 최소 표본 수는 1 이상이어야 합니다.")
    tables: list[pd.DataFrame] = []
    for feature in CATEGORICAL_CANDIDATES:
        source = cast(pd.DataFrame, df[[feature, "ConvertedCompYearly"]]).dropna()
        grouped = source.groupby(feature, observed=True)["ConvertedCompYearly"].agg(
            respondents="size", median_salary="median", mean_salary="mean",
            q25=lambda values: values.quantile(0.25), q75=lambda values: values.quantile(0.75),
        ).reset_index().rename(columns={feature: "category"})
        grouped.insert(0, "feature", feature)
        tables.append(grouped.loc[grouped["respondents"] >= min_group_size])
    summary = pd.concat(tables, ignore_index=True).sort_values(
        ["feature", "median_salary"], ascending=[True, False], ignore_index=True
    )
    if summary.empty:
        raise AnalysisError("최소 표본 기준을 만족하는 범주형 응답이 없습니다.")
    try:
        summary.to_csv(processed_dir / "categorical_salary_summary.csv", index=False)
    except OSError as exc:
        raise AnalysisError("범주형 연봉 비교표를 저장하지 못했습니다.") from exc
    return summary


def analyze_ordinal_income(
    df: pd.DataFrame, processed_dir: Path, min_group_size: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """순서형 응답과 log income의 Spearman 상관·구간별 분포를 계산한다.

    순서 점수는 간격이 같은 연속형 값이 아니므로 Pearson 대신 순위만
    사용하는 Spearman을 적용한다. 추세선은 중앙값과 25·75분위수를
    함께 저장해 일부 소표본 구간이 추세를 왜곡하는지 확인할 수 있게 한다.
    """
    if min_group_size < 2:
        raise AnalysisError("순서형 그룹의 최소 표본 수는 2 이상이어야 합니다.")
    correlation_records: list[dict[str, int | float | str]] = []
    trend_tables: list[pd.DataFrame] = []
    for score_column, (label, order_labels) in ORDINAL_TREND_SPECS.items():
        pairs = cast(pd.DataFrame, df[[score_column, "log_salary"]]).dropna()
        if len(pairs) < 3:
            continue
        result = spearmanr(pairs[score_column], pairs["log_salary"], nan_policy="omit")
        rho = float(cast(Any, result.statistic))
        p_value = float(cast(Any, result.pvalue))
        if not math.isfinite(rho) or not math.isfinite(p_value):
            continue
        correlation_records.append(
            {
                "feature": label,
                "score_column": score_column,
                "valid_rows": len(pairs),
                "spearman_rho": rho,
                "p_value": p_value,
                "significant_at_0_05": p_value < 0.05,
            }
        )
        source = cast(
            pd.DataFrame,
            df[[score_column, "ConvertedCompYearly"]].dropna(),
        )
        grouped = (
            source.groupby(score_column, observed=True)["ConvertedCompYearly"]
            .agg(
                respondents="size",
                median_income="median",
                q25=lambda values: values.quantile(0.25),
                q75=lambda values: values.quantile(0.75),
            )
            .reset_index()
        )
        grouped = grouped.loc[grouped["respondents"] >= min_group_size].copy()
        grouped.insert(0, "feature", label)
        grouped["order"] = grouped[score_column].astype(int)
        grouped["category"] = grouped["order"].map(order_labels)
        trend_tables.append(
            cast(
                pd.DataFrame,
                grouped[
                    [
                        "feature",
                        "order",
                        "category",
                        "respondents",
                        "median_income",
                        "q25",
                        "q75",
                    ]
                ],
            )
        )
    correlations = pd.DataFrame(correlation_records).sort_values(
        "spearman_rho", key=lambda values: values.abs(), ascending=False, ignore_index=True
    )
    if correlations.empty or not trend_tables:
        raise AnalysisError("순서형 응답과 income의 관계를 계산할 수 없습니다.")
    correlations["p_value_bonferroni"] = (
        correlations["p_value"] * len(correlations)
    ).clip(upper=1.0)
    correlations["significant_bonferroni_0_05"] = (
        correlations["p_value_bonferroni"] < 0.05
    )
    trends = pd.concat(trend_tables, ignore_index=True).sort_values(
        ["feature", "order"], ignore_index=True
    )
    try:
        correlations.to_csv(processed_dir / "ordinal_income_spearman.csv", index=False)
        trends.to_csv(processed_dir / "ordinal_income_trends.csv", index=False)
    except OSError as exc:
        raise AnalysisError("순서형 income EDA 결과를 저장하지 못했습니다.") from exc
    return correlations, trends


def analyze_nominal_income_associations(
    df: pd.DataFrame, processed_dir: Path, min_group_size: int = 20
) -> pd.DataFrame:
    """명목형 응답이 log income 분산을 얼마나 구분하는지 계산한다.

    범주에 임의 정수를 부여하지 않고, 그룹 간 분산 비율인 eta-squared와
    분포 가정이 적은 Kruskal-Wallis p-value를 함께 제공한다. 소표본
    범주는 결과를 불안정하게 만들 수 있어 제외한다.
    """
    if min_group_size < 2:
        raise AnalysisError("명목형 그룹의 최소 표본 수는 2 이상이어야 합니다.")
    records: list[dict[str, int | float | str | bool]] = []
    for feature in NOMINAL_ASSOCIATION_FEATURES:
        source = cast(pd.DataFrame, df[[feature, "log_salary"]]).dropna()
        counts = source[feature].value_counts()
        categories = counts.loc[counts >= min_group_size].index
        source = source.loc[source[feature].isin(categories)].copy()
        groups = [
            group["log_salary"].to_numpy()
            for _, group in source.groupby(feature, observed=True)
        ]
        if len(groups) < 2:
            continue
        income = cast(pd.Series, source["log_salary"])
        overall_mean = float(income.mean())
        ss_total = float(((income - overall_mean) ** 2).sum())
        if ss_total <= 0:
            continue
        ss_between = 0.0
        for _, group in source.groupby(feature, observed=True):
            values = cast(pd.Series, group["log_salary"])
            ss_between += len(values) * (float(values.mean()) - overall_mean) ** 2
        eta_squared = ss_between / ss_total
        try:
            test = kruskal(*groups, nan_policy="omit")
        except ValueError as exc:
            raise AnalysisError(f"{feature} Kruskal-Wallis 검정에 실패했습니다.") from exc
        statistic = float(cast(Any, test.statistic))
        p_value = float(cast(Any, test.pvalue))
        records.append(
            {
                "feature": feature,
                "valid_rows": len(source),
                "group_count": len(groups),
                "eta_squared": eta_squared,
                "kruskal_h": statistic,
                "p_value": p_value,
                "significant_at_0_05": p_value < 0.05,
            }
        )
    associations = pd.DataFrame(records).sort_values(
        "eta_squared", ascending=False, ignore_index=True
    )
    if associations.empty:
        raise AnalysisError("명목형 응답과 income의 관련성을 계산할 수 없습니다.")
    associations["p_value_bonferroni"] = (
        associations["p_value"] * len(associations)
    ).clip(upper=1.0)
    associations["significant_bonferroni_0_05"] = (
        associations["p_value_bonferroni"] < 0.05
    )
    try:
        associations.to_csv(
            processed_dir / "categorical_income_associations.csv", index=False
        )
    except OSError as exc:
        raise AnalysisError("명목형 income 연관분석을 저장하지 못했습니다.") from exc
    return associations


def analyze_employment_multiselect(
    df: pd.DataFrame, processed_dir: Path, min_group_size: int = 20
) -> pd.DataFrame:
    """Employment 다중선택을 선택지별 0/1로 분리해 income과 비교한다.

    응답하지 않은 행을 '선택하지 않음'으로 오인하지 않도록 Employment와
    income이 모두 있는 행만 사용한다. 점이연 상관계수는 선택 여부와
    log income의 방향성을 보여주며 인과관계를 의미하지 않는다.
    """
    source = cast(
        pd.DataFrame,
        df[["Employment", "ConvertedCompYearly", "log_salary"]].dropna(),
    ).copy()
    if len(source) < min_group_size * 2:
        raise AnalysisError("Employment multi-hot 분석에 사용할 응답이 부족합니다.")
    selections = source["Employment"].astype(str).str.split(";")
    options = sorted(set(selections.explode().dropna()))
    records: list[dict[str, int | float | str | bool]] = []
    for option in options:
        selected = selections.apply(lambda values: option in values)
        selected_n = int(selected.sum())
        unselected_n = int((~selected).sum())
        if selected_n < min_group_size or unselected_n < min_group_size:
            continue
        result = pointbiserialr(selected.astype(int), source["log_salary"])
        coefficient = float(cast(Any, result.statistic))
        p_value = float(cast(Any, result.pvalue))
        records.append(
            {
                "employment_option": option,
                "selected_rows": selected_n,
                "unselected_rows": unselected_n,
                "selected_median_income": float(
                    source.loc[selected, "ConvertedCompYearly"].median()
                ),
                "unselected_median_income": float(
                    source.loc[~selected, "ConvertedCompYearly"].median()
                ),
                "point_biserial_r": coefficient,
                "absolute_r": abs(coefficient),
                "p_value": p_value,
                "significant_at_0_05": p_value < 0.05,
            }
        )
    effects = pd.DataFrame(records).sort_values(
        "absolute_r", ascending=False, ignore_index=True
    )
    if effects.empty:
        raise AnalysisError("Employment 선택지별 income 관계를 계산할 수 없습니다.")
    effects["p_value_bonferroni"] = (effects["p_value"] * len(effects)).clip(
        upper=1.0
    )
    effects["significant_bonferroni_0_05"] = effects["p_value_bonferroni"] < 0.05
    try:
        effects.to_csv(processed_dir / "employment_income_effects.csv", index=False)
    except OSError as exc:
        raise AnalysisError("Employment multi-hot 분석을 저장하지 못했습니다.") from exc
    return effects
