"""Stack Overflow Survey 2024 CSV의 준비·검증·로드를 담당한다.

주요 기능:
- 로컬 results.csv 존재 및 Git LFS 포인터 여부 확인
- 필요 시 임시 파일로 다운로드한 뒤 크기·헤더 검증 후 원자적 교체
- Pandas·Polars 양쪽 로딩 후 shape와 열 순서 교차 검증

`main.py` 0단계에서 호출되며, 이 모듈은 원본 값을 변경하지 않고
DataFrame을 다음 단계에 전달한다. 파일 작성 중 실패해도 기존 CSV를
훼손하지 않도록 `.part` 파일을 사용한다.
"""

from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import TypedDict

import pandas as pd
import polars as pl

DATA_URL = (
    "https://media.githubusercontent.com/media/StackExchange/Survey/"
    "main/packages/archive/2024/results.csv"
)
MIN_DOWNLOAD_BYTES = 1_000_000
DOWNLOAD_TIMEOUT_SECONDS = 60
REQUIRED_COLUMNS = {
    "ResponseId", "YearsCode", "YearsCodePro", "ConvertedCompYearly",
    "LanguageHaveWorkedWith", "RemoteWork", "Age", "EdLevel", "DevType",
    "Country", "Employment", "OrgSize", "WorkExp",
    "Industry", "ICorPM",
    "JobSat", "JobSatPoints_1", "JobSatPoints_4", "JobSatPoints_5",
    "JobSatPoints_6", "JobSatPoints_7", "JobSatPoints_8", "JobSatPoints_9",
    "JobSatPoints_10", "JobSatPoints_11",
    "Knowledge_1", "Knowledge_2", "Knowledge_3", "Knowledge_4", "Knowledge_5",
    "Knowledge_6", "Knowledge_7", "Knowledge_8", "Knowledge_9",
}


class AnalysisError(RuntimeError):
    """예상 가능한 입력·분석·저장 오류를 사용자에게 설명하기 위한 예외."""


class LoadComparison(TypedDict):
    """Pandas와 Polars의 로드 구조·핵심 품질 지표 비교 결과."""

    pandas_shape: list[int]
    polars_shape: list[int]
    same_shape: bool
    same_columns: bool
    pandas_missing_cells: int
    polars_missing_cells: int
    pandas_duplicate_response_ids: int
    polars_duplicate_response_ids: int
    pandas_valid_income_rows: int
    polars_valid_income_rows: int
    same_quality_summary: bool


def _is_git_lfs_pointer(path: Path) -> bool:
    """작은 파일만 읽어 실제 CSV 대신 Git LFS 포인터가 받아졌는지 확인한다."""
    try:
        if not path.exists() or path.stat().st_size > 1_024:
            return False
        return path.read_text(encoding="utf-8", errors="ignore").startswith(
            "version https://git-lfs.github.com/spec/v1"
        )
    except OSError as exc:
        raise AnalysisError(f"데이터 파일 상태를 확인할 수 없습니다: {path}") from exc


def _validate_download(path: Path) -> None:
    """크기와 헤더를 검사해 HTML 오류 페이지를 CSV로 오인하지 않게 한다."""
    try:
        size = path.stat().st_size
        with path.open("rb") as file:
            header = file.readline(512).decode("utf-8-sig", errors="replace")
    except OSError as exc:
        raise AnalysisError(f"다운로드 파일을 검증할 수 없습니다: {path}") from exc
    if size < MIN_DOWNLOAD_BYTES or not header.startswith("ResponseId,"):
        raise AnalysisError("다운로드 결과가 정상적인 Survey 2024 CSV가 아닙니다.")


def _download(path: Path) -> None:
    """불완전한 파일이 원본을 덮지 않도록 `.part`에 받고 검증 후 교체한다."""
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "so-survey-project/1.0"})
    print(f"[다운로드] {DATA_URL}")
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        _validate_download(temporary)
        temporary.replace(path)
        print(f"[다운로드][SUCCESS] {path} ({path.stat().st_size / 1024**2:.1f} MB)")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise AnalysisError("데이터를 다운로드하지 못했습니다. 네트워크를 확인하거나 --data를 사용하세요.") from exc
    except OSError as exc:
        raise AnalysisError(f"데이터 파일을 저장하지 못했습니다: {path}") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            # 임시 파일 정리 실패는 기존 CSV 사용을 막지 않으므로 WARN으로 남긴다.
            print(f"[다운로드][WARN] 임시 파일 정리 실패: {exc}")


def prepare_data_file(path: Path) -> Path:
    """로컬 CSV를 우선 사용하고, 없거나 LFS 포인터이면 공식 CSV를 받는다.

    반환값은 비어 있지 않은 CSV 후보 경로다. 폴더 경로, 저장 실패,
    네트워크 오류, 비정상 다운로드는 AnalysisError로 전환한다.
    """
    if path.exists() and path.is_dir():
        raise AnalysisError(f"--data에는 CSV 파일을 지정해야 합니다: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AnalysisError(f"데이터 폴더를 만들 수 없습니다: {path.parent}") from exc
    if not path.exists() or _is_git_lfs_pointer(path):
        _download(path)
    if path.stat().st_size == 0:
        raise AnalysisError(f"데이터 파일이 비어 있습니다: {path}")
    return path


def load_data(path: Path, sample_rows: int | None) -> tuple[pd.DataFrame, pl.DataFrame, LoadComparison]:
    """동일한 CSV를 두 라이브러리로 로드하고 행·열 구조를 교차 검증한다.

    `sample_rows`가 있으면 두 라이브러리에 동일하게 적용한다. 필수 열, shape,
    열 순서 중 하나라도 다르면 이후 분석 기준이 모호해지므로 즉시 중단한다.
    """
    try:
        pandas_df = pd.read_csv(path, nrows=sample_rows, low_memory=False)
        polars_df = pl.read_csv(path, n_rows=sample_rows, null_values=["NA"], infer_schema_length=10_000)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, pl.exceptions.PolarsError, OSError, UnicodeDecodeError) as exc:
        raise AnalysisError(f"CSV를 로드하지 못했습니다: {path}") from exc
    if pandas_df.empty:
        raise AnalysisError("CSV에 분석할 행이 없습니다.")
    missing = sorted(REQUIRED_COLUMNS.difference(pandas_df.columns))
    if missing:
        raise AnalysisError("필수 열이 없습니다: " + ", ".join(missing))
    # shape만 같으면 두 로더가 NA나 숫자 타입을 다르게 해석해도
    # 놓칠 수 있다. 후속 분석에 직접 영향을 주는 핵심 품질 지표를
    # 각 엔진에서 독립적으로 계산해 로딩 결과를 교차 검증한다.
    pandas_salary = pd.to_numeric(pandas_df["ConvertedCompYearly"], errors="coerce")
    polars_salary = pl.col("ConvertedCompYearly").cast(pl.Float64, strict=False)
    pandas_missing = int(pandas_df.isna().sum().sum())
    polars_missing = int(polars_df.null_count().sum_horizontal().item())
    pandas_duplicates = int(pandas_df["ResponseId"].duplicated().sum())
    polars_duplicates = int(
        polars_df.height - polars_df.select(pl.col("ResponseId").n_unique()).item()
    )
    pandas_valid_income = int((pandas_salary.notna() & (pandas_salary > 0)).sum())
    polars_valid_income = int(
        polars_df.select((polars_salary.is_not_null() & (polars_salary > 0)).sum()).item()
    )
    comparison: LoadComparison = {
        "pandas_shape": list(pandas_df.shape), "polars_shape": list(polars_df.shape),
        "same_shape": pandas_df.shape == polars_df.shape,
        "same_columns": pandas_df.columns.tolist() == polars_df.columns,
        "pandas_missing_cells": pandas_missing,
        "polars_missing_cells": polars_missing,
        "pandas_duplicate_response_ids": pandas_duplicates,
        "polars_duplicate_response_ids": polars_duplicates,
        "pandas_valid_income_rows": pandas_valid_income,
        "polars_valid_income_rows": polars_valid_income,
        "same_quality_summary": (
            pandas_missing == polars_missing
            and pandas_duplicates == polars_duplicates
            and pandas_valid_income == polars_valid_income
        ),
    }
    print("[로드 비교]", json.dumps(comparison, ensure_ascii=False))
    if (
        not comparison["same_shape"]
        or not comparison["same_columns"]
        or not comparison["same_quality_summary"]
    ):
        raise AnalysisError("Pandas와 Polars의 로딩·품질 요약 결과가 다릅니다.")
    return pandas_df, polars_df, comparison
