# 판교 10반 4조 — Stack Overflow Survey 2024 income 분석·예측

Stack Overflow Developer Survey 2024로 income(총 연간 보상)과 관련된 요인을 탐색하고,
응답자의 배경 정보로 환산 소득(`ConvertedCompYearly`)을 추론하는
회귀 모델을 학습하는 종합 실습 프로젝트입니다.

최종 분석 결과는 [`outputs/report.md`](outputs/report.md)에서 확인할 수 있습니다.

## 작성자와 공동 작업

- 최초 작성자: **왕채은**
- 팀원이 코드를 수정하면 해당 Python 파일의 머리말 `수정 이력`에
  `YYYY-MM-DD 이름: 변경 내용` 형식으로 본인의 작업을 추가합니다.
- 공동 작업 방법과 커밋 권장 형식은 [`CONTRIBUTING.md`](CONTRIBUTING.md)를 따릅니다.

`survey.pdf` 10페이지의 문항은 income을 급여·보너스·복리후생을
포함한 세전 총 연간 보상으로 정의합니다. 원본 `CompTotal`과 `Currency`를
국가 간 비교 가능한 달러로 환산한 `ConvertedCompYearly`를 목표값으로 사용합니다.

## 탐구 질문

1. 전체·전문 코딩 경력은 연봉과 어느 정도의 선형 상관관계가 있는가?
2. 원격 근무자와 대면 근무자의 평균 연봉은 통계적으로 다른가?
3. 경력·국가·교육·직무·고용형태·조직규모·근무형태로 연봉을 얼마나 잘 추론할 수 있는가?
4. 모델에서 연봉 예측에 큰 영향을 주는 후보 요인은 무엇인가?

## 프로젝트 구조

```text
DAY2/
├─ main.py
├─ README.md
├─ data/
│  ├─ raw/                 # 원본 results.csv
│  └─ processed/           # 품질·기술통계·상관·모델 결과 CSV
├─ src/
│  ├─ data_loader.py       # Pandas·Polars 로드·품질 지표 교차 검증
│  ├─ preprocessing.py     # 중복·이상치, Likert·순서형 파생열
│  ├─ eda.py               # Pearson·Spearman·eta-squared·multi-hot EDA
│  ├─ visualization.py     # Seaborn PNG, Plotly HTML
│  ├─ statistics.py        # Welch t-test와 p-value 해석
│  └─ modeling.py          # multi-hot, 5-Fold 모델 비교·튜닝, 평가
└─ outputs/
   ├─ charts/
   ├─ html/
   └─ models/
```

## 실행

원본 설문 데이터는 용량과 재배포 범위를 고려해 저장소에 포함하지 않았습니다.
[Stack Overflow Survey 2024 공식 아카이브](https://github.com/StackExchange/Survey/tree/main/packages/archive/2024)에서
데이터를 내려받아 `data/raw/results.csv`로 저장한 뒤 실행합니다.

```bash
git clone https://github.com/chaeeunwang/pangyo_10ban_4jo.git
cd pangyo_10ban_4jo
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

빠른 검증에는 표본 행 수를 지정할 수 있습니다. 최종 제출용 결과는 이 옵션 없이
전체 데이터로 다시 생성해야 합니다.

```bash
python main.py --sample-rows 10000
python main.py --data /absolute/path/results.csv
```

## 결측치·이상치 처리

- 원본 열은 보존하고 `YearsCode_num`, `YearsCodePro_num`, `WorkExp_num`, `log_salary` 파생열을 만듭니다.
- 전체·실무 경력 비율, 실무 전 코딩 기간, 전체 직장 경력과 개발 실무
  경력의 차이를 파생해 비선형 경력 패턴도 모델이 학습할 수 있게 합니다.
- `Knowledge_1`~`Knowledge_9`의 5점 Likert 응답을 `Strongly disagree=-2`,
  `Disagree=-1`, `Neither agree nor disagree=0`, `Agree=1`,
  `Strongly agree=2`로 변환한 `*_score` 파생열을 만듭니다.
- `Age`, `EdLevel`, `OrgSize`는 선택지 순서만 보존한 `*_order`
  파생열로 만들어 Spearman EDA에 사용합니다. 응답 거부·모호한
  선택지에는 임의의 순서를 부여하지 않습니다.
- 급여 탐구에 사용할 수 없는 `ConvertedCompYearly` 결측·0 이하 행을 제외합니다.
- EDA는 유효 income의 하위 1% 미만과 상위 1% 초과를 이상치로 제거하고, 임계값을 `data_quality.csv`에 남깁니다.
- 입력 컬럼의 결측은 행을 제거하지 않습니다. Pipeline이 train 세트에서 수치형은 중앙값으로 대치하고 결측 표시열을 추가하며, 명목형은 `Missing` 범주로 보존합니다.
- 모델은 원본 유효 income을 train/test로 먼저 나눈 뒤 train에서만 IQR
  경계를 계산합니다. 같은 경계를 train·test에 적용한 일반 income 구간을
  기본 평가 대상으로 사용하고, 전체 test 지표도 비교용으로 함께 저장합니다.
- 상관계수 Top 3는 EDA 요약에만 사용하며, income 모델은 수치형 18개,
  명목형 8개, 다중선택형 6개 원 문항을 사용합니다.
- `Employment`, 코딩 활동, 사용 언어·DB·플랫폼·전문 기술은 세미콜론
  조합 전체가 아니라 개별 선택지를 multi-hot으로 변환합니다. 희귀·미지
  선택지는 train fold에서만 학습한 별도 범주로 묶습니다.
- 모델은 연봉을 `log1p`로 변환해 고연봉 극단값의 영향을 완화한 뒤 예측을 달러로 복원합니다.
- 목표가 연속형 income인 회귀 문제이므로 분류용 정확도·F1 대신
  MAE, RMSE, R-squared와 log 척도 지표를 출력합니다.
- Ridge를 기준선으로 두고 Random Forest를 동일한 5-Fold로 비교하며,
  Random Forest를 최대 8개 조합으로 튜닝합니다. 모델 선택과 튜닝이
  끝날 때까지 holdout test는 사용하지 않습니다.

## 분석 단계

0. Pandas·Polars 양쪽 로딩 및 행·열·결측·중복·유효 income 비교
1. 원본 데이터 EDA
2. 결측치·중복·급여 양쪽 1% 이상치 처리
3. 정제 데이터 EDA·기술통계
4. 연속형·Likert Pearson Top N과 순서형 선택지 Spearman 상관
5. 범주별 income 분포, 명목형 eta-squared·Kruskal-Wallis,
   Employment 다중선택 multi-hot 분석
6. 수치형·명목형·다중선택형 모델 입력 후보 정의
7. Seaborn 분포·산점도·히트맵·순서형 추세선·범주형 효과 막대그래프와
   Plotly 범주별 income 시각화
   (`JobSatPoints` 상관계수는 점수 합계가 100인 유효 배분 응답만 사용)
8. 원격·대면 `log_salary` Welch t-test와 p-value 해석
9. train 내 이상치 처리, 5-Fold 후보 비교·RandomizedSearchCV,
   최종 Pipeline holdout 평가·저장
10. 실제 결과를 반영한 `report.md` 자동 생성

## 생성 결과

- `data/processed/data_quality.csv`
- `data/processed/raw_dataset_overview.csv`
- `data/processed/raw_column_profile.csv`
- `data/processed/cleaned_salary_data.csv`
- `data/processed/cleaned_descriptive_statistics.csv`
- `data/processed/numeric_correlation_top.csv`
- `data/processed/numeric_correlation_matrix.csv`
- `data/processed/ordinal_income_spearman.csv`
- `data/processed/ordinal_income_trends.csv`
- `data/processed/categorical_income_associations.csv`
- `data/processed/employment_income_effects.csv`
- `data/processed/categorical_salary_summary.csv`
- `data/processed/selected_model_columns.csv`
- `data/processed/model_metrics.csv`
- `data/processed/model_comparison_cv.csv`
- `data/processed/model_tuning_results.csv`
- `data/processed/model_numeric_correlations.csv`
- `data/processed/salary_feature_importance.csv`
- `outputs/charts/salary_distribution.png`
- `outputs/charts/salary_by_experience.png`
- `outputs/charts/salary_correlation_heatmap.png`
- `outputs/charts/ordinal_income_trends.png`
- `outputs/charts/categorical_income_associations.png`
- `outputs/html/median_salary_by_country.html`
- `outputs/html/median_salary_by_devtype.html`
- `outputs/models/survey_income_prediction_pipeline.joblib`
- `outputs/report.md`

각 단계는 `[START]`, `[SUCCESS]`, `[FAIL]`로 소요 시간과 핵심 결과를 출력합니다.
실패하면 중단된 단계와 원인을 출력하고 종료 코드 `1`을 반환합니다.

대용량 `cleaned_salary_data.csv`와 학습 모델 `.joblib`은 실행 시 재생성되며,
공개 저장소에는 요약 CSV·차트·HTML·최종 보고서만 포함합니다.

공식 자료:

- 아카이브: <https://github.com/StackExchange/Survey/tree/main/packages/archive/2024>
- 안내 페이지: <https://survey.stackoverflow.co/2024/>
