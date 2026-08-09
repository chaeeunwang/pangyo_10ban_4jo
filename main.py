"""Stack Overflow Survey 2024 응답자 정보로 income을 분석·예측한다.

역할:
- CLI 인수를 검증하고 분석 0~10단계를 순서대로 조정한다.
- 각 단계의 START·SUCCESS·FAIL과 소요 시간을 출력한다.
- 각 모듈의 실제 결과를 모아 outputs/report.md를 생성한다.

모듈 관계와 흐름:
data/raw/results.csv -> data_loader -> preprocessing -> eda/statistics/
visualization -> modeling -> outputs/report.md

세부 데이터 처리는 `src/`에 위임하며, 이 파일에서는 중간 결과를
재계산하지 않는다. 원본은 data/raw, 가공 표는 data/processed,
시각화·모델은 outputs에 저장한다.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, NoReturn, cast

try:
    import pandas as pd

    from src.data_loader import AnalysisError, LoadComparison, load_data, prepare_data_file
    from src.eda import (
        NUMERIC_CANDIDATES,
        analyze_categorical_salary,
        analyze_cleaned_data,
        analyze_employment_multiselect,
        analyze_nominal_income_associations,
        analyze_numeric_correlations,
        analyze_ordinal_income,
        analyze_raw_data,
    )
    from src.modeling import (
        CATEGORICAL_FEATURES,
        MULTISELECT_FEATURES,
        ModelMetrics,
        save_selected_model_columns,
        train_salary_model,
    )
    from src.preprocessing import clean_data, prepare_model_data
    from src.statistics import TTestResult, compare_remote_salary
    from src.visualization import create_visualizations
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"[오류] 필수 패키지 '{exc.name}'가 없습니다.\n"
        "python -m pip install -r requirements.txt 명령을 실행하세요."
    ) from exc

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHARTS_DIR = OUTPUT_DIR / "charts"
HTML_DIR = OUTPUT_DIR / "html"
MODELS_DIR = OUTPUT_DIR / "models"
DEFAULT_DATA_PATH = RAW_DIR / "results.csv"
LAST_STAGE = 10


@contextmanager
def stage_log(step: int, name: str) -> Iterator[dict[str, str]]:
    """단계별 시작·성공·실패와 소요 시간을 일관된 형식으로 로깅한다.

    하위 함수의 예외을 여기서 소비하지 않고 FAIL을 출력한 뒤 다시
    전달한다. 따라서 상위의 AnalysisError 처리와 종료 코드가 유지된다.
    context 내부에서 state["detail"]을 설정하면 성공 로그에 핵심 결과를
    포함할 수 있다.
    """
    if not 0 <= step <= LAST_STAGE:
        raise ValueError(f"단계는 0~{LAST_STAGE}여야 합니다: {step}")
    prefix = f"[{step}/{LAST_STAGE}]"
    state: dict[str, str] = {}
    started = time.perf_counter()
    print(f"{prefix}[START] {name}", flush=True)
    try:
        yield state
    except BaseException as exc:
        reason = " ".join(str(exc).splitlines()).strip() or type(exc).__name__
        print(f"{prefix}[FAIL] {name} ({time.perf_counter() - started:.2f}초) - {reason}",
              file=sys.stderr, flush=True)
        raise
    detail = state.get("detail", "").strip()
    print(f"{prefix}[SUCCESS] {name} ({time.perf_counter() - started:.2f}초)"
          f"{' - ' + detail if detail else ''}", flush=True)


def positive_integer(value: str) -> int:
    """`--sample-rows`에 1 이상의 정수만 허용한다."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("정수를 입력해야 합니다.") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수를 입력해야 합니다.")
    return number


def parse_args() -> argparse.Namespace:
    """원본 CSV 경로와 빠른 검증용 표본 행 수를 읽는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Survey 2024 results.csv 경로")
    parser.add_argument("--sample-rows", type=positive_integer, default=None, help="빠른 점검용 행 수")
    return parser.parse_args()


def ensure_directories() -> None:
    """분석 전에 필요한 모든 입력·가공·출력 폴더를 만든다."""
    try:
        for directory in (
            RAW_DIR,
            PROCESSED_DIR,
            CHARTS_DIR,
            HTML_DIR,
            MODELS_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AnalysisError(f"프로젝트 폴더를 만들 수 없습니다: {exc.filename}") from exc


def generate_report(
    comparison: LoadComparison,
    raw_overview: pd.DataFrame,
    quality: pd.DataFrame,
    descriptive: pd.DataFrame,
    correlations: pd.DataFrame,
    ordinal_correlations: pd.DataFrame,
    nominal_associations: pd.DataFrame,
    employment_effects: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    selected_numeric: list[str],
    test_result: TTestResult,
    metrics: ModelMetrics,
    model_comparison: pd.DataFrame,
    importance: pd.DataFrame,
    sample_rows: int | None,
) -> Path:
    """각 단계의 실제 결과·차트 경로·income 모델 평가값을 report.md로 생성한다.

    전달받은 DataFrame을 Markdown 표로 변환하고, 표본 실행 여부를 명시한
    보고서를 정식 산출물 경로인 outputs/report.md에 UTF-8로 저장한다.
    파일 쓰기 실패는 AnalysisError로 전환한다.
    """
    country = categorical_summary.loc[categorical_summary["feature"] == "Country"].head(10)
    dev_type = categorical_summary.loc[categorical_summary["feature"] == "DevType"].head(10)
    try:
        raw_md = raw_overview.to_markdown(index=False)
        quality_md = quality.round(2).to_markdown(index=False)
        descriptive_md = descriptive.round(2).to_markdown()
        correlations_md = correlations.round(4).to_markdown(index=False)
        ordinal_md = ordinal_correlations[
            [
                "feature",
                "valid_rows",
                "spearman_rho",
                "p_value",
                "p_value_bonferroni",
                "significant_bonferroni_0_05",
            ]
        ].to_markdown(index=False, floatfmt=".4g")
        nominal_md = nominal_associations[
            [
                "feature",
                "valid_rows",
                "group_count",
                "eta_squared",
                "kruskal_h",
                "p_value",
                "p_value_bonferroni",
                "significant_bonferroni_0_05",
            ]
        ].to_markdown(index=False, floatfmt=".4g")
        employment_md = employment_effects.head(9)[
            [
                "employment_option",
                "selected_rows",
                "selected_median_income",
                "unselected_median_income",
                "point_biserial_r",
                "p_value_bonferroni",
                "significant_bonferroni_0_05",
            ]
        ].to_markdown(index=False, floatfmt=".4g")
        country_md = country.round(2).to_markdown(index=False)
        dev_type_md = dev_type.round(2).to_markdown(index=False)
        model_comparison_md = model_comparison[
            [
                "model",
                "tuned",
                "cv_log_rmse_mean",
                "cv_log_rmse_std",
                "cv_log_r2_mean",
                "cv_mae_usd_mean",
            ]
        ].round(4).to_markdown(index=False)
        factors_md = importance.head(12).round(4).to_markdown(index=False)
    except ImportError as exc:
        raise AnalysisError("보고서 표에 필요한 tabulate가 없습니다.") from exc
    t = test_result
    quality_values = quality.set_index("item")["value"]
    # pandas 타입 스텁은 단일 셀 접근도 complex를 포함한 Scalar로
    # 넓게 추론한다. 이 표들은 숫자형 산출물임을 이미 검증했으므로
    # 보고서용 Python 스칼라로 변환하는 라이브러리 경계에서 타입을 좁힌다.
    raw_rows = int(cast(Any, quality_values.at["rows_before"]))
    valid_income_rows = comparison["pandas_valid_income_rows"]
    valid_income_rate = valid_income_rows / raw_rows * 100
    outlier_rows = int(cast(Any, quality_values.at["salary_outlier_rows"]))
    outlier_rate = outlier_rows / valid_income_rows * 100
    income_mean = float(cast(Any, descriptive.at["ConvertedCompYearly", "mean"]))
    income_median = float(cast(Any, descriptive.at["ConvertedCompYearly", "median"]))
    strongest = correlations.iloc[0]
    strongest_feature = str(strongest["feature"])
    strongest_r = float(cast(Any, strongest.at["pearson_r"]))
    strongest_ordinal = ordinal_correlations.iloc[0]
    strongest_ordinal_feature = str(strongest_ordinal["feature"])
    strongest_ordinal_rho = float(
        cast(Any, strongest_ordinal.at["spearman_rho"])
    )
    strongest_nominal = nominal_associations.iloc[0]
    strongest_nominal_feature = str(strongest_nominal["feature"])
    strongest_eta_squared = float(cast(Any, strongest_nominal.at["eta_squared"]))
    strongest_employment = employment_effects.iloc[0]
    strongest_employment_option = str(strongest_employment["employment_option"])
    strongest_employment_r = float(
        cast(Any, strongest_employment.at["point_biserial_r"])
    )
    top_country = country.iloc[0]
    top_dev_type = dev_type.iloc[0]
    top_country_median = float(cast(Any, top_country.at["median_salary"]))
    top_country_respondents = int(cast(Any, top_country.at["respondents"]))
    top_dev_type_median = float(cast(Any, top_dev_type.at["median_salary"]))
    top_dev_type_respondents = int(cast(Any, top_dev_type.at["respondents"]))
    remote_income_ratio = math.exp(t["remote_log_mean"] - t["in_person_log_mean"])
    sample_notice = (
        f"> 빠른 점검용 {sample_rows:,}행 결과입니다. 최종 제출 전 전체 실행하세요.\n"
        if sample_rows else "> 전체 데이터 실행 결과입니다.\n"
    )
    report = f"""# Stack Overflow Survey 2024 income 분석·예측

{sample_notice}
## 0. Pandas·Polars 로딩 비교

- Pandas shape: `{comparison['pandas_shape']}`
- Polars shape: `{comparison['polars_shape']}`
- shape·열 순서 일치: `{comparison['same_shape'] and comparison['same_columns']}`
- 결측 셀: Pandas `{comparison['pandas_missing_cells']:,}` / Polars `{comparison['polars_missing_cells']:,}`
- 중복 ResponseId: Pandas `{comparison['pandas_duplicate_response_ids']:,}` / Polars `{comparison['polars_duplicate_response_ids']:,}`
- 유효 소득 행: Pandas `{comparison['pandas_valid_income_rows']:,}` / Polars `{comparison['polars_valid_income_rows']:,}`
- 핵심 품질 요약 일치: `{comparison['same_quality_summary']}`

## 1. 원본 데이터 EDA

{raw_md}

열별 dtype·결측률·고유값은 `data/processed/raw_column_profile.csv`, 원본
급여 분포는 `raw_salary_summary.csv`에 저장했다.

> **해석:** 전체 {raw_rows:,}개 응답 중 income이 있고 0보다 큰 행은
> {valid_income_rows:,}개({valid_income_rate:.1f}%)다. income 문항은 선택 응답이므로,
> 모델의 학습 모집단은 전체 설문 응답자가 아니라 income을 제출한
> 응답자로 한정된다.

## 2. 결측치·중복·급여 이상치 처리

{quality_md}

ResponseId 중복, 급여 결측·0 이하, 유효 급여의 상·하위 1% 밖 응답을 제거했다.
입력 변수의 결측은 행을 지우지 않고 모델 Pipeline 내부에서 대치했다.

> **해석:** EDA에서는 유효 income의 {outlier_rows:,}개({outlier_rate:.1f}%)만
> 이상치로 제외했다. 이 제거는 그래프와 기술통계가 극단값에
> 지나치게 좌우되는 것을 줄이기 위한 것이다. 모델은 분할 후 train에서
> IQR 경계를 계산해 train·test에 동일하게 적용한다. 정제 test를 기본
> 평가 대상으로 삼되, 전체 test 결과도 별도 지표로 함께 보고한다.

## 3. 정제 데이터 EDA·기술통계

{descriptive_md}

> **해석:** 정제 income의 평균은 ${income_mean:,.0f}, 중앙값은
> ${income_median:,.0f}로 평균이 더 크다. 이는 고소득 응답이 오른쪽 긴 꼬리를
> 만든다는 뜻이며, 달러 단위 income과 `log1p` income을 함께 확인한 이유다.

## 4. 수치형·순서형 응답과 log income

### 4.1 연속형·Likert 점수 Pearson 상관 Top 3

{correlations_md}

Pearson 상관계수는 선형 관련성이며 인과관계를 의미하지 않는다.
이 Top 3는 EDA 요약이며 모델 변수 제거 기준으로 사용하지 않았다.

> **해석:** 가장 큰 단변량 상관은 `{strongest_feature}`의
> r={strongest_r:.3f}로, 경력이 늘수록 log income이 높아지는 양의 관계가 있다.
> 다만 계수가 1에 가깝지 않으므로 경력만으로 income을 정확히 예측할 수는
> 없고, 국가·직무·교육·조직 정보를 함께 고려해야 한다.

### 4.2 순서형 선택지 Spearman 순위 상관

{ordinal_md}

`Age`, `EdLevel`, `OrgSize`는 순서만 보존한 점수로 변환했다.
`Prefer not to say`, `Something else`, `I don’t know`에는 임의의 순서를
부여하지 않고 이 분석에서 제외했다.
순서형 3개 반복 검정의 우연한 유의성을 줄이기 위해 Bonferroni 보정 p-value를
함께 제시했다.

> **해석:** 순서형 응답 중 절대 순위 상관이 가장 큰 변수는
> `{strongest_ordinal_feature}`(Spearman rho={strongest_ordinal_rho:.3f})다.
> Spearman은 각 범주 간 거리가 같다고 가정하지 않고 income이 순서에 따라
> 일관되게 증가·감소하는지를 평가한다.

## 5. 범주형 변수별 급여 중앙값·분포

### 국가 Top 10

{country_md}

### 직무 Top 10

{dev_type_md}

각 범주의 중앙값·평균·25·75분위수 전체는 `categorical_salary_summary.csv`에 저장했다.

> **해석:** 최소 20개 응답 기준을 만족한 범주 중 income 중앙값이
> 가장 높은 국가는 `{top_country['category']}`
> (${top_country_median:,.0f}, n={top_country_respondents:,}),
> 직무는 `{top_dev_type['category']}`
> (${top_dev_type_median:,.0f}, n={top_dev_type_respondents:,})다.
> 국가 간 물가·환율·직무 구성이 다르므로 이 순위를 개인의 인과적
> 소득 효과로 해석하면 안 된다.

### 5.1 명목형 응답 효과크기·Kruskal-Wallis 검정

{nominal_md}

eta-squared는 각 범주형 변수가 log income 분산을 구분하는 비율이다.
Kruskal-Wallis p-value는 적어도 한 범주의 분포가 다른지를 검정하며,
어느 범주가 다른지나 인과관계를 설명하지는 않는다.
p-value `0` 표시는 정확한 0이 아니라 부동소수점 표현 범위보다 작음을 뜻한다.
명목형 5개 검정에 대해서도 Bonferroni 보정 p-value를 함께 제시했다.

> **해석:** 비교한 명목형 응답 중 `{strongest_nominal_feature}`의
> eta-squared가 {strongest_eta_squared:.3f}로 가장 크다. 범주 수가 많은 변수는
> eta-squared가 커질 수 있으므로 표본 수와 그룹별 income 분포를 함께
> 확인해야 한다.

### 5.2 Employment 다중선택 multi-hot 분석

{employment_md}

Employment 문자열을 세미콜론으로 분리해 각 선택지를 0/1로 표현했다.
문항 결측은 '선택하지 않음'으로 취급하지 않았다.
선택지별 반복 검정에도 Bonferroni 보정을 적용했다.

> **해석:** income과 가장 큰 절대 점이연 상관을 보인 선택지는
> `{strongest_employment_option}`(r={strongest_employment_r:.3f})다.
> 양수는 해당 선택지를 고른 응답자의 log income이 더 높은 경향,
> 음수는 더 낮은 경향을 뜻한다.

## 6. income 모델 응답자 정보

- 수치형 응답 {len(selected_numeric)}개: `{', '.join(selected_numeric)}`
- 명목형 원핫 응답 {len(CATEGORICAL_FEATURES)}개: `{', '.join(CATEGORICAL_FEATURES)}`
- 다중선택 multi-hot 응답 {len(MULTISELECT_FEATURES)}개: `{', '.join(MULTISELECT_FEATURES)}`

임의의 상관계수 Top N으로 변수를 제거하지 않고, 목표 income과
직접 파생 변수를 제외한 응답자·경력·근무·기술 정보를 사용했다.
순서가 없는 범주는 임의 숫자로 바꾸지 않았으며, 세미콜론 다중선택은
개별 선택지로 분리했다. 희귀 선택지 통합 기준과 결측 대치는 각 학습
fold 안에서만 학습했다.

## 7. Seaborn·Plotly 시각화

![정제 급여와 로그 급여 히스토그램](charts/salary_distribution.png)

![전문 코딩 경력별 급여 분포](charts/salary_by_experience.png)

![설문 수치형 변수와 로그 income의 Pearson 상관관계](charts/salary_correlation_heatmap.png)

![연령·교육·조직 규모 순서별 income 중앙값 추세](charts/ordinal_income_trends.png)

![명목형 효과크기와 Employment multi-hot income 관련성](charts/categorical_income_associations.png)

히트맵의 `JobSatPoints` 항목은 응답자가 직무 만족에 기여하는 요인에
총 100점을 배분한 결과다. 점수 합계가 100인 응답만 해당 요인의
상관계수에 사용했다. 이 확장 히트맵은 탐색용이며, 모델의
수치형 상관계수 Top 3는 EDA 요약에만 사용했고, income 모델은
사전에 정의한 수치형 응답을 모두 사용했다.
`Knowledge_1`~`Knowledge_9`은 Strongly disagree부터 Strongly agree까지를
`-2`~`2`로 변환한 파생열을 사용했다.

> **차트 해석:** income 분포는 오른쪽으로 길어 log 변환 후에 더 안정적이다.
> 경력-income 산점도는 전반적인 양의 추세와 함께 같은 경력에서도 income
> 편차가 크다는 점을 보여준다. 히트맵의 경력 변수 간 높은 상관은
> 서로 유사한 경력 개념을 측정한 결과이므로 단일 상관계수를 인과적으로
> 해석하면 안 된다.
> 순서형 추세선은 중앙값과 25·75분위 범위를 함께 보여줘 단순한
> 순위 상관계수만으로 보이지 않는 비선형 변화를 확인할 수 있다.

- [국가별 중앙 급여](html/median_salary_by_country.html)
- [직무별 중앙 급여](html/median_salary_by_devtype.html)

## 8. 원격·대면 로그 급여 Welch t-test

- 원격: n={t['remote_n']:,}, log_salary 평균={t['remote_log_mean']:.4f}
- 대면: n={t['in_person_n']:,}, log_salary 평균={t['in_person_log_mean']:.4f}
- t={t['t_statistic']:.3f}, p={t['p_value']:.6g}, alpha={t['alpha']:.2f}
- 해석: **{t['interpretation']}**

관찰자료의 집단 차이이므로 근무 형태가 급여 차이를 일으킨다고 단정할 수 없다.

> **해석:** p-value가 0.05보다 훨씬 작아 두 집단의 평균 log income이
> 같다는 귀무가설을 기각한다. log 평균 차이를 비율로 환산하면 원격
> 집단의 기하평균 income 수준이 대면 집단의 약 {remote_income_ratio:.2f}배다.
> 그러나 국가·직무·경력 구성이 다른 관찰자료이므로 원격근무의 인과효과로
> 해석하지 않는다.

## 9. income prediction Pipeline

- 목표변수: `ConvertedCompYearly` (USD 환산 income)
- 문제 유형: 연속값 회귀이므로 분류용 정확도·F1 대신 MAE·RMSE·R-squared 사용
- 외부 평가 분할: 80/20 무작위 holdout, `random_state=42`
- 전처리: 수치형 중앙값 대치·표준화, 명목형 원핫, 다중선택 multi-hot
- 비교 모델: Ridge 기준선, Random Forest
- 선택 모델: **{metrics['selected_model']}**
- 검증: train 내부 {metrics['cv_folds']}-Fold 교차검증 + Random Forest 최대 {metrics['tuning_candidate_count']}개 조합 RandomizedSearchCV
- CV log RMSE: **{metrics['cv_log_rmse_mean']:.3f} ± {metrics['cv_log_rmse_std']:.3f}**
- CV log R-squared: **{metrics['cv_log_r2_mean']:.3f} ± {metrics['cv_log_r2_std']:.3f}**
- 누수 방지: holdout test를 모델 선택·튜닝에서 격리하고 train에서만 IQR 경계·전처리·어휘 학습
- train 기반 IQR 범위: `${max(0.0, metrics['train_salary_lower_bound']):,.2f}` ~ `${metrics['train_salary_upper_bound']:,.2f}`
- 학습 행: `{metrics['train_rows_before_outlier_filter']:,}` -> `{metrics['train_rows']:,}`
- test 행: `{metrics['test_rows_before_outlier_filter']:,}` -> `{metrics['test_rows']:,}`
- IQR test MAE: **${metrics['mae_usd']:,.2f}**
- IQR test RMSE: **${metrics['rmse_usd']:,.2f}**
- IQR test R-squared: **{metrics['r2']:.3f}**
- IQR test log RMSE: **{metrics['log_rmse']:.3f}**
- IQR test log R-squared: **{metrics['log_r2']:.3f}**
- 전체 test MAE: **${metrics['full_test_mae_usd']:,.2f}**
- 전체 test RMSE: **${metrics['full_test_rmse_usd']:,.2f}**
- 전체 test R-squared: **{metrics['full_test_r2']:.3f}**
- 전체 test log R-squared: **{metrics['full_test_log_r2']:.3f}**
- 저장: `outputs/models/survey_income_prediction_pipeline.joblib`

### 5-Fold 모델 비교

{model_comparison_md}

Ridge는 선형 기준선으로 함께 표시하고, 이번 실험에서는 요청한 Random Forest를
튜닝·저장한다. 튜닝 결과가 기본 Random Forest보다 나쁠 경우에는 기본
Random Forest를 유지한다.

> **성능 해석:** train에서 정한 IQR 범위의 일반 income 응답에서는 분산의
> {metrics['r2'] * 100:.1f}%를 설명하고 평균 절대오차는
> ${metrics['mae_usd']:,.0f}다. 극단값을 포함한 전체 test R-squared는
> {metrics['full_test_r2']:.3f}로, 평가 모집단을 제한하면 성능이 얼마나
> 달라지는지 함께 보여준다. 따라서 IQR 지표를 전체 응답자 성능으로
> 일반화하지 않고 두 결과를 구분해 해석해야 한다.

### 모델 입력 중요도 후보

{factors_md}

표의 중요도는 holdout 일부에서 각 원 입력 열을 섞었을 때 감소한 log R-squared다.
양수 값이 클수록 현재 모델이 그 입력에 더 의존하지만, 인과관계 순위는 아니다.

## 10. 한계·재현

- 국가별 물가·세금·환율을 직접 통제하지 못했다.
- 설문은 비확률·자기보고 표본이므로 전체 개발자로 일반화할 때 주의해야 한다.
- 실행: `python main.py`
"""
    report_path = OUTPUT_DIR / "report.md"
    try:
        report_path.write_text(report, encoding="utf-8")
    except OSError as exc:
        raise AnalysisError(f"보고서를 저장하지 못했습니다: {report_path}") from exc
    return report_path


def fail_unexpectedly(exc: Exception) -> NoReturn:
    """예상하지 못한 오류의 유형과 메시지를 남기고 종료한다."""
    print(f"[예상하지 못한 오류] {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc


def main() -> int:
    """분석을 0~10단계로 실행하고 프로세스 종료 코드를 반환한다.

    성공은 0, 예상 가능한 AnalysisError는 1, 사용자 중단은 130을 반환한다.
    예상하지 못한 예외은 유형과 메시지를 stderr에 남기고 1로 종료한다.
    """
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    try:
        with stage_log(0, "Pandas·Polars 양쪽 로딩·비교") as stage:
            ensure_directories()
            data_path = prepare_data_file(data_path)
            pandas_df, polars_df, comparison = load_data(data_path, args.sample_rows)
            stage["detail"] = f"{len(pandas_df):,}행 x {pandas_df.shape[1]}열, 구조 일치"
        with stage_log(1, "원본 데이터 EDA") as stage:
            raw_overview, _ = analyze_raw_data(pandas_df, PROCESSED_DIR)
            stage["detail"] = "dtype·결측률·원본 급여 분포 저장"
        with stage_log(2, "결측치·중복·급여 이상치 처리") as stage:
            cleaned, quality = clean_data(pandas_df, PROCESSED_DIR)
            model_data = prepare_model_data(pandas_df)
            stage["detail"] = f"정제 후 {len(cleaned):,}행"
        with stage_log(3, "정제 데이터 EDA·기술통계") as stage:
            descriptive, _ = analyze_cleaned_data(cleaned, PROCESSED_DIR)
            stage["detail"] = "정제 후 분포·결측률 저장"
        with stage_log(4, "로그 급여-수치형 상관관계 Top N") as stage:
            correlations = analyze_numeric_correlations(cleaned, PROCESSED_DIR)
            ordinal_correlations, ordinal_trends = analyze_ordinal_income(
                cleaned, PROCESSED_DIR
            )
            eda_top_numeric = correlations["feature"].astype(str).tolist()
            stage["detail"] = (
                f"Pearson={','.join(eda_top_numeric)}, "
                f"순서형 Spearman {len(ordinal_correlations)}개"
            )
        with stage_log(5, "범주형 변수별 급여 중앙값·분포") as stage:
            categorical_summary = analyze_categorical_salary(cleaned, PROCESSED_DIR)
            nominal_associations = analyze_nominal_income_associations(
                cleaned, PROCESSED_DIR
            )
            employment_effects = analyze_employment_multiselect(
                cleaned, PROCESSED_DIR
            )
            stage["detail"] = (
                f"{len(categorical_summary):,}개 유효 범주, "
                f"명목형 {len(nominal_associations)}개, "
                f"Employment 선택지 {len(employment_effects)}개"
            )
        with stage_log(6, "모델 입력 후보 컬럼 정의") as stage:
            stage["detail"] = (
                f"수치형 후보 {len(NUMERIC_CANDIDATES)} + "
                f"명목형 {len(CATEGORICAL_FEATURES)} + "
                f"다중선택 {len(MULTISELECT_FEATURES)}"
            )
        with stage_log(7, "Seaborn·Plotly 시각화") as stage:
            chart_paths = create_visualizations(
                cleaned,
                categorical_summary,
                ordinal_trends,
                nominal_associations,
                employment_effects,
                CHARTS_DIR,
                HTML_DIR,
            )
            stage["detail"] = f"차트 {len(chart_paths)}개 저장"
        with stage_log(8, "Welch t-test·p-value 해석") as stage:
            test_result = compare_remote_salary(cleaned)
            stage["detail"] = f"p-value={test_result['p_value']:.3g}, {test_result['interpretation']}"
        with stage_log(9, "Pipeline 회귀 모델 학습·평가·저장") as stage:
            _, metrics, importance, _, selected_numeric, model_comparison = train_salary_model(
                model_data, NUMERIC_CANDIDATES, PROCESSED_DIR, MODELS_DIR
            )
            save_selected_model_columns(selected_numeric, PROCESSED_DIR)
            stage["detail"] = (
                f"{metrics['selected_model']}, "
                f"R2={metrics['r2']:.3f}, MAE=${metrics['mae_usd']:,.0f}"
            )
        with stage_log(10, "report.md 자동 생성") as stage:
            report_path = generate_report(
                comparison, raw_overview, quality, descriptive, correlations,
                ordinal_correlations, nominal_associations, employment_effects,
                categorical_summary, selected_numeric, test_result, metrics,
                model_comparison, importance, args.sample_rows,
            )
            stage["detail"] = str(report_path)
    except AnalysisError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 실행을 취소했습니다.", file=sys.stderr)
        return 130
    except Exception as exc:
        fail_unexpectedly(exc)
    print(f"[완료] 보고서: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
