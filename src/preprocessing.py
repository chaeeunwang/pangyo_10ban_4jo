"""원본 응답을 급여 분석에 적합한 정제 DataFrame으로 변환한다.

주요 기능:
- ResponseId 중복 제거
- YearsCode·YearsCodePro·WorkExp의 수치형 파생열 생성
- 전체·실무 코딩 경력의 비율과 경력 차이 파생열 생성
- Knowledge_1~9 Likert 응답을 -2~2 점수 파생열로 변환
- Age·EdLevel·OrgSize 선택지를 순서를 보존한 EDA·모델 점수로 변환
- 급여 결측·0 이하와 유효 급여 양쪽 1% 이상치 제거
- log_salary 파생열과 정제 품질표 저장

`data_loader`의 Pandas DataFrame을 받아 파생열을 한 번만 만든다. EDA는
전체 유효 소득의 양쪽 1%를 제외한 모집단을 사용하고, 모델링은
평가 데이터 누수를 막기 위해 이상치 제거 전 모집단을 받아 train에서
IQR 경계를 학습한 뒤 train·test에 동일하게 적용한다. 원본 DataFrame은
직접 수정하지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.data_loader import AnalysisError

KNOWLEDGE_COLUMNS = [f"Knowledge_{number}" for number in range(1, 10)]
KNOWLEDGE_SCORE_COLUMNS = [f"{column}_score" for column in KNOWLEDGE_COLUMNS]
LIKERT_SCORE_MAP = {
    "Strongly disagree": -2,
    "Disagree": -1,
    "Neither agree nor disagree": 0,
    "Agree": 1,
    "Strongly agree": 2,
}

# 순서형 문항만 크기 순서를 가진 점수로 변환한다. 점수 간
# 간격이 같다고 가정하지 않는다. EDA에서는 Spearman 순위 상관으로
# 해석하고, 비선형 모델에서는 범주 순서 단서로 사용한다. 응답 거부·모호한
# 범주는 인위적 순서를 만들지 않도록 결측로 둔다.
AGE_ORDER = {
    "Under 18 years old": 0,
    "18-24 years old": 1,
    "25-34 years old": 2,
    "35-44 years old": 3,
    "45-54 years old": 4,
    "55-64 years old": 5,
    "65 years or older": 6,
}
EDUCATION_ORDER = {
    "Primary/elementary school": 0,
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 1,
    "Some college/university study without earning a degree": 2,
    "Associate degree (A.A., A.S., etc.)": 3,
    "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 4,
    "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 5,
    "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 6,
}
ORG_SIZE_ORDER = {
    "Just me - I am a freelancer, sole proprietor, etc.": 0,
    "2 to 9 employees": 1,
    "10 to 19 employees": 2,
    "20 to 99 employees": 3,
    "100 to 499 employees": 4,
    "500 to 999 employees": 5,
    "1,000 to 4,999 employees": 6,
    "5,000 to 9,999 employees": 7,
    "10,000 or more employees": 8,
}
ORDINAL_COLUMN_SPECS = {
    "Age": ("Age_order", AGE_ORDER, {"Prefer not to say"}),
    "EdLevel": ("EdLevel_order", EDUCATION_ORDER, {"Something else"}),
    "OrgSize": ("OrgSize_order", ORG_SIZE_ORDER, {"I don’t know"}),
}
ORDINAL_SCORE_COLUMNS = [spec[0] for spec in ORDINAL_COLUMN_SPECS.values()]
EXPERIENCE_DERIVED_COLUMNS = [
    "ProfessionalExperienceRatio",
    "PreProfessionalCodingYears",
    "WorkProfessionalGapYears",
]


def _years_to_number(series: pd.Series) -> pd.Series:
    """경계 범주를 대표값으로 바꾸고, 해석할 수 없는 값은 결측으로 보존한다."""
    replaced = series.replace({"Less than 1 year": 0.5, "More than 50 years": 51.0})
    return cast(pd.Series, pd.to_numeric(replaced, errors="coerce"))


def _likert_to_score(series: pd.Series, column: str) -> pd.Series:
    """5점 Likert 응답을 -2~2로 변환하고 원래 결측은 유지한다.

    알 수 없는 문자열을 결측으로 묵음 처리하면 설문 스키마 변경을
    놓칠 수 있으므로 변환 전에 허용된 응답인지 검증한다.
    """
    unknown = sorted(set(series.dropna().astype(str).unique()) - set(LIKERT_SCORE_MAP))
    if unknown:
        raise AnalysisError(
            f"{column}에 알 수 없는 Likert 응답이 있습니다: {', '.join(unknown)}"
        )
    # Likert 결측을 NaN으로 안전하게 유지하고 Pandas·scikit-learn이
    # nullable 정수 내부값을 잘못 축소하지 않도록 float64를 사용한다.
    return cast(pd.Series, series.map(LIKERT_SCORE_MAP).astype("float64"))


def _ordinal_to_score(
    series: pd.Series,
    column: str,
    mapping: dict[str, int],
    ignored_values: set[str],
) -> pd.Series:
    """순서형 선택지를 순위 점수로 변환하고 응답 거부는 결측로 둔다.

    정의하지 않은 실제 응답이 들어오면 설문 스키마 변경을 조용히
    결측로 처리하지 않고 즉시 알린다.
    """
    observed = set(series.dropna().astype(str).unique())
    unknown = sorted(observed - set(mapping) - ignored_values)
    if unknown:
        raise AnalysisError(
            f"{column}에 알 수 없는 순서형 응답이 있습니다: {', '.join(unknown)}"
        )
    return cast(pd.Series, series.map(mapping).astype("float64"))


def _derive_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """응답 ID를 중복 제거하고 설문 문항의 수치형 파생열을 만든다."""
    prepared = df.drop_duplicates(subset=["ResponseId"]).copy()
    prepared["YearsCode_num"] = _years_to_number(cast(pd.Series, prepared["YearsCode"]))
    prepared["YearsCodePro_num"] = _years_to_number(
        cast(pd.Series, prepared["YearsCodePro"])
    )
    prepared["WorkExp_num"] = pd.to_numeric(prepared["WorkExp"], errors="coerce")
    total_years = cast(pd.Series, prepared["YearsCode_num"])
    professional_years = cast(pd.Series, prepared["YearsCodePro_num"])
    # 전체 코딩 경력이 0보다 큰 응답만 비율을 정의한다. 자기보고 오차로
    # 실무 경력이 전체 경력보다 큰 경우는 1로 잘라 극단 비율이 모델을
    # 지배하지 않게 하되, 원래 두 경력 값은 별도 특성으로 그대로 보존한다.
    prepared["ProfessionalExperienceRatio"] = (
        professional_years.div(total_years.where(total_years > 0)).clip(0.0, 1.0)
    )
    prepared["PreProfessionalCodingYears"] = (
        total_years.sub(professional_years).clip(lower=0.0)
    )
    # 개발 외 직무 경력도 소득과 관련될 수 있어 음수를 임의로 지우지 않고
    # 원 응답의 차이를 보존한다. 결측 대치는 후속 Pipeline의 train에서만 한다.
    prepared["WorkProfessionalGapYears"] = cast(
        pd.Series, prepared["WorkExp_num"]
    ).sub(professional_years)
    for source_column, score_column in zip(KNOWLEDGE_COLUMNS, KNOWLEDGE_SCORE_COLUMNS):
        prepared[score_column] = _likert_to_score(
            cast(pd.Series, prepared[source_column]), source_column
        )
    for source_column, (score_column, mapping, ignored_values) in ORDINAL_COLUMN_SPECS.items():
        prepared[score_column] = _ordinal_to_score(
            cast(pd.Series, prepared[source_column]),
            source_column,
            mapping,
            ignored_values,
        )
    prepared["ConvertedCompYearly"] = pd.to_numeric(
        prepared["ConvertedCompYearly"], errors="coerce"
    )
    return prepared


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """모델 분할 전에 중복·무효 타겟만 제거한 모집단을 만든다.

    소득 분위수를 이 단계에서 사용하지 않아 후속 모델이 test 세트의
    타겟 분포를 보고 학습 행을 선택하는 누수를 방지한다.
    """
    prepared = _derive_analysis_columns(df)
    salary = cast(pd.Series, prepared["ConvertedCompYearly"])
    valid_salary = cast(pd.Series, salary.notna() & (salary > 0))
    model_data = prepared.loc[valid_salary].copy()
    if len(model_data) < 100:
        raise AnalysisError("소득 모델에 사용할 유효 응답이 부족합니다.")
    return model_data


def clean_data(df: pd.DataFrame, processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """중복, 무효 연봉, 상·하위 1% 연봉 이상치를 제거한다.

    탐구 목표가 연봉이므로 연봉 결측·0 이하 행은 정제 데이터에서 제외한다.
    환율·입력 오류일 가능성이 큰 양쪽 1%를 제거해 상관분석과 모델이
    극단값에 지나치게 좌우되지 않게 한다. 임계값은 품질표에 남긴다.

    반환값은 정제 DataFrame과 품질 지표 DataFrame이다. 정제 CSV와
    data_quality.csv를 저장하며, 유효 급여가 100개 미만이거나 저장에
    실패하면 AnalysisError를 발생시킨다.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    cleaned = _derive_analysis_columns(df)
    duplicate_count = len(df) - len(cleaned)
    salary = cast(pd.Series, cleaned["ConvertedCompYearly"])
    valid_salary_mask = cast(pd.Series, salary.notna() & (salary > 0))
    invalid_salary_count = int((~valid_salary_mask).sum())
    salary_valid = cast(pd.Series, salary.loc[valid_salary_mask])
    if len(salary_valid) < 100:
        raise AnalysisError("연봉 이상치 기준을 계산할 유효 응답이 부족합니다.")
    lower_bound = float(salary_valid.quantile(0.01))
    upper_bound = float(salary_valid.quantile(0.99))
    cleaned = cleaned.loc[valid_salary_mask].copy()
    salary = cast(pd.Series, cleaned["ConvertedCompYearly"])
    outlier_mask = cast(pd.Series, (salary < lower_bound) | (salary > upper_bound))
    outlier_count = int(outlier_mask.sum())
    cleaned = cleaned.loc[~outlier_mask].copy()
    cleaned["log_salary"] = np.log1p(cast(pd.Series, cleaned["ConvertedCompYearly"]))

    quality = pd.DataFrame({
        "item": [
            "rows_before", "duplicate_response_ids", "missing_or_nonpositive_salary_rows",
            "salary_lower_1pct_bound", "salary_upper_1pct_bound", "salary_outlier_rows",
            "rows_after", "total_missing_cells_after",
        ],
        "value": [
            len(df), duplicate_count, invalid_salary_count, lower_bound, upper_bound,
            outlier_count, len(cleaned), int(cleaned.isna().sum().sum()),
        ],
    })
    try:
        quality.to_csv(processed_dir / "data_quality.csv", index=False)
        analysis_columns = [
            "ResponseId", "ConvertedCompYearly", "log_salary", "YearsCode_num",
            "YearsCodePro_num", "WorkExp_num", *EXPERIENCE_DERIVED_COLUMNS,
            *KNOWLEDGE_COLUMNS,
            *KNOWLEDGE_SCORE_COLUMNS, *ORDINAL_SCORE_COLUMNS,
            "Country", "DevType", "RemoteWork", "EdLevel",
            "Employment", "OrgSize", "Age", "Industry", "ICorPM",
        ]
        cast(pd.DataFrame, cleaned[analysis_columns]).to_csv(
            processed_dir / "cleaned_salary_data.csv", index=False
        )
    except OSError as exc:
        raise AnalysisError("정제 데이터와 품질표를 저장하지 못했습니다.") from exc
    return cleaned, quality
