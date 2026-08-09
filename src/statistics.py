"""원격·대면 근무자의 평균 로그 급여 차이를 Welch t-test로 검정한다.

작성 정보:
- 최초 작성자: 왕채은
- 공동 수정자는 이 파일을 변경할 때 아래 형식으로 이력을 추가한다.
- 수정 이력:
  - 2026-08-09 왕채은: 공동 작업용 작성자·수정 이력 형식 추가
  - YYYY-MM-DD 이름: 변경 내용

`preprocessing`이 생성한 log_salary를 사용해 우측 긴 꼬리와 이상치에
대한 민감도를 줄인다. 집단별 표본 수·평균·t 통계량·p-value·해석을
`TTestResult`로 반환해 main.py의 8단계 로그와 report.md에 제공한다.
결측치 대치로 분산을 인위적으로 줄이지 않도록 집단별 완전 사례만 사용한다.
"""

from __future__ import annotations

import math
from typing import Any, TypedDict, cast

import pandas as pd
from scipy.stats import ttest_ind

from src.data_loader import AnalysisError


class TTestResult(TypedDict):
    """Welch t-test의 표본 요약, 검정통계량, 유의수준 해석."""
    remote_n: int
    in_person_n: int
    remote_log_mean: float
    in_person_log_mean: float
    t_statistic: float
    p_value: float
    alpha: float
    interpretation: str


def compare_remote_salary(df: pd.DataFrame) -> TTestResult:
    """집단별 log_salary 결측은 대치하지 않고 Welch t-test를 수행한다.

    결측 연봉을 평균으로 대치하면 집단 분산이 인위적으로 작아져 p-value가
    왜곡될 수 있어 완전 사례 분석을 선택했다.
    """
    remote = df.loc[df["RemoteWork"] == "Remote", "log_salary"].dropna()
    in_person = df.loc[df["RemoteWork"] == "In-person", "log_salary"].dropna()
    if remote.size < 2 or in_person.size < 2:
        raise AnalysisError("t-test에는 원격·대면 집단별 연봉이 각각 2개 이상 필요합니다.")
    statistic_raw, p_value_raw = ttest_ind(remote, in_person, equal_var=False, nan_policy="omit")
    statistic, p_value = float(cast(Any, statistic_raw)), float(cast(Any, p_value_raw))
    if not math.isfinite(statistic) or not math.isfinite(p_value):
        raise AnalysisError("t-test 결과가 유효한 숫자가 아닙니다.")
    interpretation = "통계적으로 유의한 평균 차이가 있다" if p_value < 0.05 else "통계적으로 유의한 평균 차이를 확인하지 못했다"
    return {"remote_n": int(remote.size), "in_person_n": int(in_person.size),
            "remote_log_mean": float(remote.mean()), "in_person_log_mean": float(in_person.mean()),
            "t_statistic": statistic, "p_value": p_value, "alpha": 0.05,
            "interpretation": interpretation}
