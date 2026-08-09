"""정제 급여의 분포·경력 관계·범주별 중앙값을 시각화한다.

주요 산출물:
- Seaborn: 원 급여·로그 급여 히스토그램, 전문 경력-급여 산점도·추세선,
  수치형 변수·로그 급여 Pearson 상관행렬 히트맵
- 순서형: 연령·교육·조직 규모별 income 중앙값과 사분위 추세선
- 범주형: 명목형 eta-squared와 Employment multi-hot 상관 막대그래프
- Plotly: 국가별·직무별 중앙 급여 인터랙티브 막대그래프

`eda.analyze_categorical_salary`의 요약표를 재사용해 시각화와 CSV의
모집단 정의를 일치시킨다. GUI가 없는 채점 환경에서도 PNG를 저장하도록
pyplot import 전에 Agg 백엔드를 설정한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from src.data_loader import AnalysisError
from src.eda import build_extended_correlation_matrix

TREND_SHORT_LABELS = {
    "Age": ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    "Education level": [
        "Primary",
        "Secondary",
        "Some college",
        "Associate",
        "Bachelor's",
        "Master's",
        "Professional",
    ],
    "Organization size": [
        "Solo",
        "2-9",
        "10-19",
        "20-99",
        "100-499",
        "500-999",
        "1,000-4,999",
        "5,000-9,999",
        "10,000+",
    ],
}


def create_visualizations(
    df: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    ordinal_trends: pd.DataFrame,
    nominal_associations: pd.DataFrame,
    employment_effects: pd.DataFrame,
    charts_dir: Path,
    html_dir: Path,
) -> list[Path]:
    """income 분포·상관·순서형 추세·범주형 효과 차트를 만든다.

    정제 단계에서 급여 양쪽 1%가 제거됐으므로 히스토그램과 산점도는
    동일한 분석 모집단을 표현한다. 산점도는 중첩을 줄이기 위해 최대 10,000행을
    고정 시드로 추출하지만 회귀모델은 전체 정제 데이터를 사용한다.

    반환값은 실제로 저장한 5개 PNG와 2개 HTML 경로다. 필요한
    요약표가 비어 있거나 생성·저장에 실패하면 AnalysisError를 발생시킨다.
    """
    charts_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    created: list[Path] = []

    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    try:
        sns.histplot(data=df, x="ConvertedCompYearly", bins=50, kde=True, ax=axes[0], color="#4472C4")
        axes[0].set(title="Cleaned annual salary distribution", xlabel="Annual salary (USD)", ylabel="Respondents")
        sns.histplot(data=df, x="log_salary", bins=50, kde=True, ax=axes[1], color="#ED7D31")
        axes[1].set(title="Log annual salary distribution", xlabel="log1p(annual salary)", ylabel="Respondents")
        figure.tight_layout()
        path = charts_dir / "salary_distribution.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        created.append(path)
    except (OSError, ValueError) as exc:
        raise AnalysisError("연봉 히스토그램을 생성하지 못했습니다.") from exc
    finally:
        plt.close(figure)

    if ordinal_trends.empty:
        raise AnalysisError("순서형 income 추세선에 사용할 데이터가 없습니다.")
    figure, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
    try:
        for axis, feature in zip(axes, TREND_SHORT_LABELS):
            subset = ordinal_trends.loc[ordinal_trends["feature"] == feature].sort_values(
                "order"
            )
            if subset.empty:
                raise AnalysisError(f"{feature} income 추세선 데이터가 없습니다.")
            axis.plot(
                subset["order"],
                subset["median_income"],
                marker="o",
                linewidth=2.5,
                color="#4472C4",
                label="Median income",
            )
            axis.fill_between(
                subset["order"].to_numpy(dtype=float),
                subset["q25"].to_numpy(dtype=float),
                subset["q75"].to_numpy(dtype=float),
                color="#4472C4",
                alpha=0.18,
                label="25th-75th percentile",
            )
            orders = subset["order"].astype(int).tolist()
            labels = TREND_SHORT_LABELS[feature]
            axis.set_xticks(orders, [labels[order] for order in orders], rotation=35, ha="right")
            axis.set(title=feature, xlabel="Ordered category")
            axis.legend(fontsize=10)
        axes[0].set_ylabel("Annual income (USD)")
        figure.suptitle("Median Annual Income across Ordered Survey Categories", y=1.02)
        figure.tight_layout()
        path = charts_dir / "ordinal_income_trends.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        created.append(path)
    except (OSError, ValueError, IndexError) as exc:
        raise AnalysisError("순서형 income 추세선을 생성하지 못했습니다.") from exc
    finally:
        plt.close(figure)

    if nominal_associations.empty or employment_effects.empty:
        raise AnalysisError("범주형 income 막대그래프에 사용할 데이터가 없습니다.")
    figure, axes = plt.subplots(1, 2, figsize=(22, 8))
    try:
        nominal_plot = nominal_associations.sort_values("eta_squared")
        sns.barplot(
            data=nominal_plot,
            x="eta_squared",
            y="feature",
            color="#70AD47",
            ax=axes[0],
        )
        axes[0].set(
            title="Nominal Features: Explained Log-Income Variance",
            xlabel="Eta-squared",
            ylabel="Survey feature",
        )
        employment_plot = employment_effects.sort_values("point_biserial_r")
        sns.barplot(
            data=employment_plot,
            x="point_biserial_r",
            y="employment_option",
            color="#ED7D31",
            ax=axes[1],
        )
        axes[1].axvline(0, color="black", linewidth=1)
        axes[1].set(
            title="Employment Choices and Log Annual Income",
            xlabel="Point-biserial correlation",
            ylabel="Employment choice",
        )
        figure.tight_layout()
        path = charts_dir / "categorical_income_associations.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        created.append(path)
    except (OSError, ValueError) as exc:
        raise AnalysisError("범주형 income 관련성 막대그래프를 생성하지 못했습니다.") from exc
    finally:
        plt.close(figure)

    # EDA가 저장하는 행렬과 동일한 함수를 사용해 CSV와 히트맵의 값이
    # 달라지지 않게 한다. 만족도 문항의 결측은 변수 쌍별로 제외된다.
    correlation_matrix = build_extended_correlation_matrix(df)
    matrix_size = len(correlation_matrix.columns)
    figure_side = max(18, matrix_size * 0.9)
    figure, axis = plt.subplots(figsize=(figure_side, figure_side))
    try:
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            annot_kws={"size": 8},
            cbar_kws={"label": "Pearson correlation"},
            ax=axis,
        )
        axis.set_title("Pearson Correlation: Survey Features and Log Annual Income")
        axis.tick_params(axis="x", labelrotation=55)
        axis.tick_params(axis="y", labelrotation=0)
        figure.tight_layout()
        path = charts_dir / "salary_correlation_heatmap.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        created.append(path)
    except (OSError, ValueError) as exc:
        raise AnalysisError("급여 상관관계 히트맵을 생성하지 못했습니다.") from exc
    finally:
        plt.close(figure)

    experience = cast(pd.DataFrame, df[["YearsCodePro_num", "ConvertedCompYearly"]]).dropna()
    if len(experience) > 10_000:
        experience = experience.sample(n=10_000, random_state=42)
    figure, axis = plt.subplots(figsize=(12, 7))
    try:
        sns.regplot(data=experience, x="YearsCodePro_num", y="ConvertedCompYearly",
                    scatter_kws={"alpha": 0.18, "s": 18}, line_kws={"color": "crimson"}, ax=axis)
        axis.set(title="Professional coding experience and annual salary",
                 xlabel="Professional coding experience (years)", ylabel="Annual salary (USD)")
        figure.tight_layout()
        path = charts_dir / "salary_by_experience.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        created.append(path)
    except (OSError, ValueError) as exc:
        raise AnalysisError("경력-연봉 Seaborn 차트를 생성하지 못했습니다.") from exc
    finally:
        plt.close(figure)

    for feature, filename, title in (
        ("Country", "median_salary_by_country.html", "Median annual salary by country"),
        ("DevType", "median_salary_by_devtype.html", "Median annual salary by developer type"),
    ):
        subset = categorical_summary.loc[categorical_summary["feature"] == feature].head(20).copy()
        if subset.empty:
            raise AnalysisError(f"{feature} Plotly 차트에 사용할 범주가 없습니다.")
        try:
            chart = px.bar(subset.sort_values("median_salary"), x="median_salary", y="category",
                           orientation="h", color="respondents", title=title,
                           labels={"median_salary": "Median salary (USD)", "category": feature,
                                   "respondents": "Respondents"})
            path = html_dir / filename
            chart.write_html(path, include_plotlyjs="cdn")
            created.append(path)
        except (OSError, ValueError) as exc:
            raise AnalysisError(f"{feature} Plotly 차트를 생성하지 못했습니다.") from exc
    return created
