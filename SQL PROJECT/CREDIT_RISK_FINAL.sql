-- ============================================================================
--         CREDIT RISK SEGMENTATION & LENDING STRATEGY OPTIMIZATION
-- ============================================================================
-- Database : PostgreSQL / MySQL compatible
-- Author   : KUSHAGRA YADAV — Data Analyst
-- Sections :
--   1. Data Preparation & Feature Engineering
--   2. Portfolio Overview & Risk-Adjusted Performance
--   3. Risk Distribution (Window Functions — NTILE, PERCENT_RANK)
--   4. Risk Ranking — Top Contributors (RANK, DENSE_RANK)
--   5. Risk Trend Analysis (LAG / LEAD)
--   6. Cohort & Partition Analysis
--   7. Risk Concentration — Cumulative Pareto (80/20)
--   8. Anomaly Detection
--   9. Strategy Simulation & Business Impact
--  10. Final Decision Framework
-- ============================================================================


-- ============================================================================
-- TABLE DEFINITION
-- ============================================================================

CREATE TABLE loan_default (
    LoanID          VARCHAR(50) PRIMARY KEY,
    Age             INT,
    Income          INT,
    LoanAmount      INT,
    CreditScore     INT,
    MonthsEmployed  INT,
    NumCreditLines  INT,
    InterestRate    DECIMAL(10,2),
    LoanTerm        INT,
    DTIRatio        FLOAT,
    Education       VARCHAR(50),
    EmploymentType  VARCHAR(50),
    MaritalStatus   VARCHAR(50),
    HasMortgage     VARCHAR(10),
    HasDependents   VARCHAR(10),
    LoanPurpose     VARCHAR(50),
    HasCoSigner     VARCHAR(10),
    DefaultFlag     INT
);

SELECT * FROM loan_default LIMIT 10;


-- ============================================================================
-- SECTION 1 — DATA PREPARATION & FEATURE ENGINEERING
-- ============================================================================

/* BUSINESS PROBLEM
 Raw loan data is not directly usable for risk analysis. Financial institutions
 require structured, categorized data to identify risk patterns, compare customer
 groups, and support consistent loan approval decisions.

 OBJECTIVE
 Transform raw data into an analysis-ready foundation by:
   * Removing invalid/inconsistent records
   * Applying CIBIL-standard credit score bands
   * Creating derived financial ratios (LTI, risk score)
   * Classifying DTI, income, employment stability */

-- NOTE: This VIEW is created once and reused across all sections below,
-- which is cleaner SQL engineering than repeating CTEs in every query.
drop view feature_data
CREATE VIEW feature_data AS
WITH base_data AS (
    SELECT *
    FROM loan_default
    WHERE
        Income      > 0
        AND LoanAmount  > 0
        AND CreditScore BETWEEN 300 AND 850
        AND DTIRatio    >= 0
)
SELECT
    *,

    -- CIBIL / FICO Standard Credit Score Bands
    -- (300-579 = Poor, 580-669 = Fair, 670-739 = Good, 740-799 = Very Good, 800+ = Excellent)
    CASE
        WHEN CreditScore < 580             THEN 'Poor'
        WHEN CreditScore BETWEEN 580 AND 669 THEN 'Fair'
        WHEN CreditScore BETWEEN 670 AND 739 THEN 'Good'
        WHEN CreditScore BETWEEN 740 AND 799 THEN 'Very Good'
        ELSE                                    'Excellent'
    END AS credit_score_band,

    -- Income Segmentation (derived from actual dataset)
    CASE 
        WHEN Income < 30000 THEN 'Low Income'
        WHEN Income BETWEEN 30000 AND 80000 THEN 'Middle Income'
        ELSE 'High Income'
    END AS income_band,

    -- DTI Risk Band
    CASE
        WHEN DTIRatio < 0.20 THEN 'Low DTI'
        WHEN DTIRatio BETWEEN 0.20 AND 0.40 THEN 'Medium DTI'
        ELSE 'High DTI'
    END AS dti_band,

    -- Employment Stability (quantile-based: P25=30, P50=60, P75=90 months)
    CASE
        WHEN MonthsEmployed < 30 THEN 'Low Stability'
        WHEN MonthsEmployed BETWEEN 30 AND 60 THEN 'Lower-Mid Stability'
        WHEN MonthsEmployed BETWEEN 60 AND 90 THEN 'Upper-Mid Stability'
        ELSE 'High Stability'
    END AS employment_stability,

    -- Loan-to-Income Ratio (derived financial ratio — key BFSI metric)
    ROUND(LoanAmount * 1.0 / NULLIF(Income, 0), 2) AS lti_ratio,

    -- LTI Risk Band
    CASE
        WHEN LoanAmount * 1.0 / NULLIF(Income, 0) < 2   THEN 'Low Burden'
        WHEN LoanAmount * 1.0 / NULLIF(Income, 0) <= 4  THEN 'Moderate Burden'
        ELSE 'High Burden'
    END AS loan_burden_category,

    -- Multi-factor Weighted Risk Score 
    (
        CASE WHEN CreditScore < 500 THEN 3
             WHEN CreditScore BETWEEN 500 AND 650 THEN 1
             ELSE 0 END
        +
        CASE WHEN DTIRatio > 0.40 THEN 2
             WHEN DTIRatio BETWEEN 0.20 AND 0.40 THEN 1
             ELSE 0 END
        +
        CASE WHEN Income < 50000 THEN 2
             WHEN Income BETWEEN 50000 AND 80000 THEN 1
             ELSE 0 END
        +
        CASE WHEN MonthsEmployed < 30 THEN 2
             WHEN MonthsEmployed BETWEEN 30 AND 60 THEN 1
             ELSE 0 END
    ) AS risk_score

FROM base_data;


-- DATA VALIDATION — confirm distributions look correct before analysis

SELECT credit_score_band,    COUNT(*) AS n FROM feature_data GROUP BY credit_score_band;
SELECT income_band,          COUNT(*) AS n FROM feature_data GROUP BY income_band;
SELECT dti_band,             COUNT(*) AS n FROM feature_data GROUP BY dti_band;
SELECT employment_stability, COUNT(*) AS n FROM feature_data GROUP BY employment_stability ORDER BY n DESC;
SELECT loan_burden_category, COUNT(*) AS n FROM feature_data GROUP BY loan_burden_category;

-- =========
-- INSIGHTS
-- =========
/* Credit Score
 Poor segment is highest → portfolio is high-risk heavy
 Excellent segment is low → limited premium customers

 Income
 High income dominates → affordability is strong overall
 Risk likely driven by other factors (not income alone)

 DTI
 High DTI segment is highest → majority customers are over-leveraged
 Primary risk driver in portfolio

 Employment Stability
 High stability dominates → employment is not main risk factor
 Risk comes from financial behavior, not job stability

 Overall Takeaway
 Portfolio risk driven by high DTI + low credit score combination
 Indicates weak credit filtering in approval process */

-- ============================================================================
-- SECTION 2 — PORTFOLIO OVERVIEW & RISK-ADJUSTED PERFORMANCE
-- ============================================================================

/* BUSINESS PROBLEM
 The bank needs a high-level health check of the full loan portfolio:
 Is the portfolio profitable? How much risk exists in absolute terms?
 Are we earning enough interest to cover default losses?

 OBJECTIVE
 Measure total exposure, default rate, expected vs realized return,
 and flag whether the portfolio is risk-optimized or volume-driven. */

-- PORTFOLIO SNAPSHOT
SELECT DISTINCT

    COUNT(*)  OVER () AS total_loans,
    SUM(LoanAmount) OVER () AS total_exposure,
    ROUND(AVG(DefaultFlag) OVER (), 4) AS default_rate,
    ROUND(AVG(InterestRate) OVER (), 2) AS avg_interest_rate,

    -- Total interest the bank EXPECTS to earn (assuming zero defaults)
    SUM(LoanAmount * InterestRate / 100) OVER () AS total_expected_interest,

    -- Total capital at risk from defaulted loans
    SUM(LoanAmount * DefaultFlag) OVER () AS total_loss_exposure,

    -- Risk-adjusted return: what the bank actually earns after absorbing default losses
    ROUND(
        SUM(LoanAmount * InterestRate / 100 * (1 - DefaultFlag)) OVER ()
        / SUM(LoanAmount) OVER ()
    , 4) AS risk_adjusted_return

FROM feature_data;

-- PORTFOLIO EFFICIENCY — loan-level expected vs realized return (sample)

SELECT
    LoanID,
    LoanAmount,
    InterestRate,
    DefaultFlag,
    ROUND(LoanAmount * InterestRate / 100, 2) AS expected_interest,
    ROUND(LoanAmount * InterestRate / 100 * (1 - DefaultFlag), 2) AS realized_return
FROM feature_data
LIMIT 50;

-- =========
-- INSIGHTS
-- =========
/* ₹32.5B portfolio with 11.6% default rate (~29.6K defaults) → high absolute loss exposure
 Avg interest 13.49% vs realized ~11.41% → ~2.08% yield loss due to defaults
 ~29.6K defaults → interest income drops to 0 on these loans → major revenue leakage
 Default rate >10% → indicates moderate-high risk portfolio (not prime lending)
 High scale + high default → risk not aligned with pricing strategy
 Portfolio driven by volume (₹32.5B) but losing efficiency via ~2% return erosion
 Key issue → borrower selection not optimized for risk-adjusted returns */


-- ============================================================================
-- SECTION 3 — RISK DISTRIBUTION (WINDOW FUNCTIONS: NTILE, PERCENT_RANK)
-- ============================================================================

/* BUSINESS PROBLEM
 The bank knows the overall default rate is ~12%, but does not understand
 HOW risk is distributed. Is it concentrated in a few customers, or spread evenly?

 OBJECTIVE
 Segment customers based on risk (DTI + Credit Score indirectly)
 Identify which segments carry highest default risk
 Understand risk concentration across the portfolio */

WITH risk_buckets AS (
    SELECT
        LoanID,
        DTIRatio,
        CreditScore,
        LoanAmount,
        DefaultFlag,

        CASE 
            WHEN DTIRatio < 0.3 THEN 'Low DTI'
            WHEN DTIRatio BETWEEN 0.3 AND 0.6 THEN 'Moderate DTI'
            ELSE 'High DTI'
        END AS dti_segment

    FROM feature_data
)

SELECT
    dti_segment,
    COUNT(*) AS total_loans,
    ROUND(AVG(DTIRatio)::numeric, 2) AS avg_dti,
    SUM(DefaultFlag) AS total_defaults,
    ROUND(AVG(DefaultFlag), 3) AS default_rate,

    ROUND(
        SUM(DefaultFlag) * 1.0 / SUM(SUM(DefaultFlag)) OVER (),
    2) AS pct_of_all_defaults

FROM risk_buckets
GROUP BY dti_segment
ORDER BY avg_dti DESC;


-- PERCENT_RANK — where does each customer stand in the risk distribution?
-- Useful to identify the top 10% riskiest customers
SELECT
    LoanID,
    CreditScore,
    DTIRatio,
    Income,
    DefaultFlag,
    ROUND(PERCENT_RANK() OVER (ORDER BY DTIRatio DESC)::numeric, 4)  AS dti_risk_percentile,
    ROUND(PERCENT_RANK() OVER (ORDER BY CreditScore ASC)::NUMERIC,4) AS credit_risk_percentile
FROM feature_data
ORDER BY dti_risk_percentile DESC
LIMIT 100;


-- COMBINED CREDIT × DTI RISK MATRIX
-- (this is where risk actually concentrates — single-variable analysis misses this)
SELECT
    credit_score_band,
    dti_band,
    COUNT(*) AS total_loans,
    SUM(DefaultFlag) AS total_defaults,
    ROUND(AVG(DefaultFlag), 3) AS default_rate,
    ROUND(
        SUM(DefaultFlag) * 1.0 / SUM(SUM(DefaultFlag)) OVER ()
    , 3) AS contribution_to_defaults
FROM feature_data
GROUP BY credit_score_band, dti_band
ORDER BY default_rate DESC;

-- =========
-- INSIGHTS
-- =========
/* High DTI -> 12.2% default rate, Low DTI -> 10.6% -> only ~1.6% gap -> weak standalone predictor
 Default contribution: High DTI 39%, Moderate DTI 39% -> ~78% defaults from mid–high DTI
 Low DTI -> 10.6% default rate + 22% contribution -> relatively safer segment
 No sharp risk jump across DTI bands -> no clear cutoff for decisioning
 Insight -> DTI shows directional risk increase, not strong separation
 Key limitation -> DTI alone insufficient -> must combine with credit score / income */


-- =====================================================================
-- SECTION 4 — HIGH-RISK SEGMENT IDENTIFICATION & CONTRIBUTION ANALYSIS
-- =====================================================================

/* BUSINESS PROBLEM
 The bank needs to know WHICH specific customer profiles and loan purposes
 contribute the most to defaults. Knowing "Poor credit is risky" is not enough —
 we need ranked prioritization to decide where to act first.

 OBJECTIVE
 FIND
   * Which segments are driving maximum losses
   * Segments that are dangerous due to scale, not just risk
   * Specific real-world borrower profiles causing defaults
   * Clear priority list for risk control and lending decisions */

-- SEGMENT RANKING — which risk profiles should the bank address first?
WITH segment_summary AS (
    SELECT
        credit_score_band,
        dti_band,
        income_band,
        COUNT(*) AS total_loans,
        SUM(DefaultFlag) AS total_defaults,
        ROUND(AVG(DefaultFlag), 4) AS default_rate,
        SUM(LoanAmount * DefaultFlag) AS total_loss_exposure
    FROM feature_data
    GROUP BY credit_score_band, dti_band, income_band
)
SELECT
    credit_score_band,
    dti_band,
    income_band,
    total_loans,
    total_defaults,
    default_rate,
    total_loss_exposure,
 
    RANK() OVER (ORDER BY total_loss_exposure DESC) AS loss_rank,
    DENSE_RANK() OVER (ORDER BY default_rate DESC) AS default_rate_rank,
   
    RANK() OVER ( PARTITION BY credit_score_band ORDER BY total_defaults DESC) 
	AS rank_within_credit_band
FROM segment_summary
ORDER BY loss_rank



-- LOAN PURPOSE RANKING — where is the bank losing most money?
WITH purpose_loss AS (
    SELECT
        LoanPurpose,
        COUNT(*) AS total_loans,
        SUM(DefaultFlag) AS total_defaults,
        ROUND(AVG(DefaultFlag) * 100, 2) AS default_rate_pct,
        SUM(LoanAmount * DefaultFlag) AS total_loss,
        ROUND(AVG(InterestRate), 2) AS avg_interest_rate
    FROM feature_data
    GROUP BY LoanPurpose
)
SELECT
    *,
    RANK() OVER (ORDER BY total_loss DESC) AS loss_rank,
    RANK() OVER (ORDER BY default_rate_pct DESC) AS default_rate_rank
FROM purpose_loss
ORDER BY loss_rank;


-- HIGH-RISK CUSTOMER IDENTIFICATION — individual customers for manual review
-- Rank customers within each risk_score bucket by loan amount (biggest exposure first)
SELECT
    LoanID,
    CreditScore,
    Income,
    LoanAmount,
    DTIRatio,
    risk_score,
    DefaultFlag,
    RANK() OVER (
        PARTITION BY risk_score
        ORDER BY LoanAmount DESC
    ) AS rank_within_risk_tier
FROM feature_data
WHERE risk_score >= 6      -- High risk threshold from scoring model
ORDER BY risk_score DESC, rank_within_risk_tier
;

-- =========
-- INSIGHTS
-- =========
/* * Poor credit + High DTI -> top default driver -> highest risk concentration across segments
 * Low income -> **20–23% default rate (high intensity)** but low volume -> limited total impact
 * High income -> **~9–10% default rate (low risk)** but high volume -> major default contribution
 * Risk driven by **Volume × Default Rate**, not % alone -> large segments dominate losses
 * Good/Very Good/Excellent + High DTI -> still **high defaults** -> DTI overrides credit quality
 * Loan purpose (Business/Auto/Education/Other) -> **~11–12% default, ~13–14% contribution each** -> no major differentiation
 * Excellent/Very Good + Low DTI -> **~6–8% default rate** -> lowest risk, ideal target segment */

-- =====================================================
-- SECTION 5 — DEFAULT TREND ANALYSIS ACROSS LOAN TERMS
-- =====================================================

/* BUSINESS PROBLEM
 The bank has been analyzing risk as a static snapshot.
 It does not know: Is the default rate improving or worsening over time?
 Are certain loan terms creating deferred risk (customers defaulting later)?

 OBJECTIVE
 Understand whether longer or shorter loan durations carry higher risk
 Detect if risk is increasing or decreasing step-by-step
 Identify whether upcoming loan terms show higher default tendency
 Understand if risk is delayed rather than immediate
 LOAN TERM TREND — default rate by loan term (proxy for time cohort) */

WITH term_summary AS (
    SELECT
        LoanTerm,
        COUNT(*) AS total_loans,
        SUM(DefaultFlag) AS total_defaults,
        ROUND(AVG(DefaultFlag) * 100, 2) AS default_rate_pct,
        ROUND(AVG(InterestRate), 2) AS avg_interest_rate,
        SUM(LoanAmount) AS total_exposure
    FROM feature_data
    GROUP BY LoanTerm
)
SELECT
    LoanTerm,
    total_loans,
    total_defaults,
    default_rate_pct,
    avg_interest_rate,
    total_exposure,

    -- LAG: what was the default rate in the previous loan term cohort?
    LAG(default_rate_pct)  OVER (ORDER BY LoanTerm) AS prev_term_default_rate,
	
    -- Change from previous period (positive = worsening, negative = improving)
    ROUND(
        default_rate_pct
        - LAG(default_rate_pct) OVER (ORDER BY LoanTerm)
    , 2) AS default_rate_change,

    -- LEAD: what is the default rate in the next loan term cohort?
    LEAD(default_rate_pct) OVER (ORDER BY LoanTerm) AS next_term_default_rate,

    -- Flag: is the next period riskier than the current?
    CASE
        WHEN LEAD(default_rate_pct) OVER (ORDER BY LoanTerm)
             > default_rate_pct
        THEN 'Risk Increasing'
        WHEN LEAD(default_rate_pct) OVER (ORDER BY LoanTerm)
             < default_rate_pct
        THEN 'Risk Decreasing'
        ELSE 'Stable'
    END AS risk_trend_flag

FROM term_summary
ORDER BY LoanTerm;


-- CREDIT SCORE BAND TREND — how does default rate shift across score bands?
WITH band_summary AS (
    SELECT
        credit_score_band,
        COUNT(*) AS total_loans,
        ROUND(AVG(DefaultFlag) * 100, 2) AS default_rate_pct,
        ROUND(AVG(LoanAmount), 0) AS avg_loan_amount,
        -- Order by risk (Poor is highest risk → order by descending credit score effectively)
        CASE credit_score_band
            WHEN 'Poor'       THEN 1
            WHEN 'Fair'       THEN 2
            WHEN 'Good'       THEN 3
            WHEN 'Very Good'  THEN 4
            WHEN 'Excellent'  THEN 5
        END AS band_order
    FROM feature_data
    GROUP BY credit_score_band
)
SELECT
    credit_score_band,
    total_loans,
    default_rate_pct,
    avg_loan_amount,

    LAG(default_rate_pct)  OVER (ORDER BY band_order) AS prev_band_default_rate,
    ROUND(
        default_rate_pct
        - LAG(default_rate_pct) OVER (ORDER BY band_order)
    , 2) AS improvement_vs_prev_band,

    LEAD(default_rate_pct) OVER (ORDER BY band_order) AS next_band_default_rate

FROM band_summary
ORDER BY band_order;

-- =========
-- INSIGHTS
-- =========
/* * Loan term default rate **~11.5%–11.7%** -> variation only **±0.1%** -> negligible impact
 * No trend across terms -> **risk remains stable (no increase/decrease pattern)**
 * Credit score impact: Poor **12.47% -> Excellent 9.81%** -> **~2.66% risk gap**
 * Credit segments show **consistent decline in default (step-wise improvement)**
 * Loan term ≠ strong predictor -> **borrower quality drives risk, not duration* */

-- ============================================================================
-- SECTION 6 — COHORT & PARTITION ANALYSIS
-- ============================================================================

/* BUSINESS PROBLEM
 Aggregate stats hide important sub-group behavior. A 12% default rate
 looks uniform until you partition by employment type, education, and
 marital status — then dramatically different risk profiles emerge.
 Banks use cohort analysis to set differentiated approval policies.

 OBJECTIVE
 Calculate cohort-level default rates using PARTITION BY (employment, education, marital status)
 Compare individual customer risk vs their cohort average
 Identify high-risk vs low-risk cohorts hidden within overall portfolio
 Detect within-cohort variation (customers better/worse than their group)
 Enable cohort-based risk differentiation for lending decisions */

-- EMPLOYMENT TYPE COHORT — default rate and rank within cohort
SELECT
    EmploymentType,
    LoanPurpose,
    COUNT(*) AS total_loans,
    SUM(DefaultFlag) AS total_defaults,
    ROUND(AVG(DefaultFlag) * 100, 2) AS default_rate_pct,

    -- How does this purpose rank within its employment type cohort?
    RANK() OVER (
        PARTITION BY EmploymentType
        ORDER BY AVG(DefaultFlag) DESC
    ) AS purpose_risk_rank_in_employment,

    -- Average default rate for this employment type (across all purposes)
    ROUND(AVG(AVG(DefaultFlag)) OVER (
        PARTITION BY EmploymentType
    ) * 100, 2) AS employment_avg_default_rate,

    -- How much riskier is this sub-group vs its employment type average?
    ROUND(
        (AVG(DefaultFlag) - AVG(AVG(DefaultFlag)) OVER (
            PARTITION BY EmploymentType
        )) * 100
    , 2) AS deviation_from_employment_avg

FROM feature_data
GROUP BY EmploymentType, LoanPurpose
ORDER BY EmploymentType, default_rate_pct DESC;


-- EDUCATION × MARITAL STATUS COHORT — risk profile matrix
SELECT
    Education,
    MaritalStatus,
    COUNT(*) AS total_loans,
    ROUND(AVG(DefaultFlag) * 100, 2) AS default_rate_pct,
    ROUND(AVG(Income), 0) AS avg_income,
    ROUND(AVG(LoanAmount), 0) AS avg_loan_amount,

    -- Rank within Education cohort
    RANK() OVER (
        PARTITION BY Education
        ORDER BY AVG(DefaultFlag) DESC
    ) AS marital_risk_rank_in_education,

    -- Portfolio-wide default rate for comparison (no partition = full window)
    ROUND(AVG(AVG(DefaultFlag)) OVER () * 100, 2) AS portfolio_avg_default_rate

FROM feature_data
GROUP BY Education, MaritalStatus
ORDER BY Education, default_rate_pct DESC;


-- INCOME BAND COHORT — running average default rate within income segment
WITH customer_level AS (
    SELECT
        LoanID,
        income_band,
        CreditScore,
        LoanAmount,
        DefaultFlag,
        -- Running average of default flag within each income band, ordered by credit score
        ROUND(
            AVG(DefaultFlag) OVER (
                PARTITION BY income_band
                ORDER BY CreditScore ASC
                ROWS BETWEEN 499 PRECEDING AND CURRENT ROW
            ) * 100
        , 2) AS rolling_default_rate_in_band
    FROM feature_data
)
SELECT
    income_band,
    COUNT(*) AS total_customers,
    ROUND(AVG(DefaultFlag) * 100, 2) AS overall_default_rate,
    ROUND(AVG(rolling_default_rate_in_band), 2) AS avg_rolling_default_rate
FROM customer_level
GROUP BY income_band
ORDER BY overall_default_rate DESC;

-- =========
-- INSIGHTS
-- =========
/* **Income is the strongest risk driver (clear separation)**

  * Low Income -> 21.96 % default (≈2.3x vs High Income)
  * Middle Income -> 11.76 %
  * High Income -> 9.29 %

* **Employment stability directly impacts risk**

  * Unemployed -> 13.55 % (highest)
  * Part-time -> 11.96 %
  * Self-employed -> 11.46 %
  * Full-time -> 9.46 % (lowest)

* **Business loans consistently overperform in risk (+ vs cohort avg)**

  * Full-time: 10.00 % vs 9.46 % (+0.53)
  * Part-time: 12.57 % vs 11.96 % (+0.60)
  * Self-employed: 12.34 % vs 11.46 % (+0.87)
  * Unemployed: 14.37% vs 13.55 % (+0.82)

* **Home loans consistently underperform in risk (safest segment)**

  * Full-time: 8.05% (-1.42 vs avg)
  * Part-time: 10.62% (-1.34)
  * Self-employed: 10.41% (-1.06)
  * Unemployed: 11.85% (-1.70)

* **Marital status shows strong behavioral signal**

  * Divorced -> highest risk (~13–14%)
  * Single -> moderate (~11–13%)
  * Married -> lowest (~9–11%)

* **Education reduces risk progressively**

  * High School -> ~13–14 %
  * Bachelor’s -> ~10–13 %
  * Master’s / PhD → ~9–11 %

* **Within-cohort variation proves segmentation is necessary**

  * Example (Full-time avg = 9.46 %)

    * Business -> 10.00 % (+0.53)
    * Home -> 8.05 % (-1.42)

* **Key business insight**

 Risk is **not uniform within groups** -> depends on **loan purpose + income + employment stability together**/


-- ============================================================================
-- SECTION 7 —DEFAULT CONCENTRATION ANALYSIS (Pareto-Based Risk Segmentation)
-- ============================================================================

/* BUSINESS PROBLEM
 The bank suspects a small portion of customers drives a disproportionately
 large share of losses (Pareto principle: 20% of customers = 80% of defaults).
 If true, targeted intervention on this group is far more efficient than
 broad policy changes.

 OBJECTIVE
 Rank customers based on risk score / default contribution
 Calculate cumulative share of total defaults across ranked customers
 Identify the smallest customer segment contributing ~80% of defaults
 Quantify risk concentration vs total portfolio size
 Enable targeted risk control on high-impact customers */

-- CUMULATIVE DEFAULT CONCENTRATION BY RISK SCORE
WITH risk_summary AS (
    SELECT
        risk_score,
        COUNT(*) AS total_customers,
        SUM(DefaultFlag) AS total_defaults,
        ROUND(AVG(DefaultFlag) * 100, 2) AS default_rate_pct,
        SUM(LoanAmount * DefaultFlag) AS total_loss
    FROM feature_data
    GROUP BY risk_score
),
cumulative AS (
    SELECT
        risk_score,
        total_customers,
        total_defaults,
        default_rate_pct,
        total_loss,

        -- Cumulative defaults as risk score increases (highest risk first)
        SUM(total_defaults) OVER (
            ORDER BY risk_score DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_defaults,

        -- Cumulative customers
        SUM(total_customers) OVER (
            ORDER BY risk_score DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_customers,

        -- Cumulative loss exposure
        SUM(total_loss) OVER (
            ORDER BY risk_score DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_loss

    FROM risk_summary
)
SELECT
    risk_score,
    total_customers,
    total_defaults,
    default_rate_pct,
    total_loss,
    cumulative_defaults,
    cumulative_customers,
    cumulative_loss,

    -- What % of ALL defaults have we captured at this risk score threshold?
    ROUND(
        cumulative_defaults * 100.0
        / SUM(total_defaults) OVER ()
    , 1) AS pct_defaults_captured,

    -- What % of ALL customers does this represent?
    ROUND(
        cumulative_customers * 100.0
        / SUM(total_customers) OVER ()
    , 1) AS pct_customers_captured,

    -- Pareto flag: at what point do we hit 80% of defaults?
    CASE
        WHEN SUM(total_defaults) OVER (
            ORDER BY risk_score DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) * 100.0 / SUM(total_defaults) OVER () >= 80
        THEN 'Captures 80%+ of Defaults'
        ELSE '—'
    END AS pareto_flag

FROM cumulative
ORDER BY risk_score DESC;


-- LOAN PURPOSE PARETO — which purposes drive 80% of losses?
WITH purpose_loss AS (
    SELECT
        LoanPurpose,
        SUM(LoanAmount * DefaultFlag) AS total_loss
    FROM feature_data
    GROUP BY LoanPurpose
)
SELECT
    LoanPurpose,
    total_loss,
    ROUND(total_loss * 100.0 / SUM(total_loss) OVER (), 2) AS pct_of_total_loss,
    ROUND(
        SUM(total_loss) OVER (
            ORDER BY total_loss DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) * 100.0 / SUM(total_loss) OVER ()
    , 1) AS cumulative_loss_pct
FROM purpose_loss
ORDER BY total_loss DESC;

-- =========
-- INSIGHTS
-- =========
/* Top risk buckets (7–9) contribute 22.5% of defaults from a small segment
 Risk buckets (5–9) contribute 58.3% of defaults → majority concentration
 Risk buckets (3–9) capture 89.7% of total defaults (~80% threshold crossed)
 Bottom segments (0–2) contribute only ~10% of defaults → low impact group
 Default contribution is highly concentrated in mid-to-high risk segments
 Small portion of customers drives disproportionate default volume (Pareto effect)
 Loan purpose contribution:
 Business → 21.18%
 Auto → 20.54%
 Other → 20.28%
 Education → 20.22%
 Home → 17.78%
 Top 4 loan purposes together contribute ~82% of total default amount
 Default contribution across loan purposes is evenly distributed (no extreme outlier)
 * Default risk is customer-driven (concentrated in high-risk segments), not loan-purpose-driven
 * Targeting top ~30–40% high-risk customers can control ~80% of total defaults */


-- =============================================================
-- SECTION 8 — OUTLIER DETETION & ABNORMAL RISK IDENTIFICATION
-- =============================================================

/* BUSINESS PROBLEM
 Standard segmentation analyzes average behavior. But anomalies —
 customers whose financials are statistically unusual — often represent
 either fraud risk, data errors, or extreme credit risk that averages hide.

 OBJECTIVE
 Identify customers with extreme deviation from cohort behavior using Z-score (> ±2 SD)
 Detect outliers within segments using IQR (upper/lower extreme ranges)
 Flag customers with unusual financial patterns vs their peer group
 Identify high-risk anomalies hidden behind normal averages
 Detect inconsistent or contradictory data patterns using business rules
 Isolate small high-impact outlier groups for risk monitoring */

-- Z-SCORE ANOMALY DETECTION — customers statistically far from their income band average
WITH band_stats AS (
    SELECT
        income_band,
        AVG(LoanAmount) AS mean_loan,
        STDDEV(LoanAmount) AS sd_loan,
        AVG(DTIRatio) AS mean_dti,
        STDDEV(DTIRatio) AS sd_dti
    FROM feature_data
    GROUP BY income_band
),
scored AS (
    SELECT
        f.LoanID,
        f.income_band,
        f.LoanAmount,
        f.DTIRatio,
        f.CreditScore,
        f.DefaultFlag,
        ROUND(
            ((f.LoanAmount - b.mean_loan) / NULLIF(b.sd_loan, 0))::NUMERIC
        , 2) AS loan_zscore,
        ROUND(
            ((f.DTIRatio - b.mean_dti) / NULLIF(b.sd_dti, 0))::NUMERIC
        , 2) AS dti_zscore
    FROM feature_data f
    JOIN band_stats b ON f.income_band = b.income_band
)
SELECT
    *,
  CASE
    WHEN ABS(loan_zscore) > 2 AND ABS(dti_zscore) > 2 
        THEN 'Extreme Anomaly'
    WHEN ABS(loan_zscore) > 1.5 AND ABS(dti_zscore) > 1.5 
        THEN 'High Anomaly'
    WHEN ABS(loan_zscore) > 1.5 
        THEN 'Loan Anomaly'
    WHEN ABS(dti_zscore)  > 1.5 
        THEN 'DTI Anomaly'
    ELSE 'Normal'
END AS anomaly_flag
FROM scored
WHERE ABS(loan_zscore) > 1.5 
   OR ABS(dti_zscore) > 1.5
ORDER BY ABS(loan_zscore) + ABS(dti_zscore) DESC
LIMIT 100;


-- IQR-BASED EXTREME RISK DETECTION — customers in bottom quartile of credit score
-- AND top quartile of DTI simultaneously
WITH thresholds AS (
    SELECT
        PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY CreditScore) AS low_credit_cutoff,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY DTIRatio)    AS high_dti_cutoff
    FROM feature_data
)
SELECT
    f.LoanID,
    f.CreditScore,
    f.DTIRatio,
    f.Income,
    f.LoanAmount,
    f.EmploymentType,
    f.LoanPurpose,
    f.DefaultFlag,
    'Extreme Risk Profile' AS anomaly_type
FROM feature_data f
JOIN thresholds t ON 1=1
WHERE 
    f.CreditScore <= t.low_credit_cutoff   -- Bottom 10% credit
    AND
    f.DTIRatio    >= t.high_dti_cutoff     -- Top 10% DTI
ORDER BY f.CreditScore ASC, f.DTIRatio DESC;


-- BUSINESS LOGIC ANOMALIES — contradictory or impossible data combinations
SELECT
    LoanID,
    Income,
    LoanAmount,
    DTIRatio,
    MonthsEmployed,
    EmploymentType,
    DefaultFlag,
    CASE
        WHEN EmploymentType = 'Unemployed' AND Income > 100000
            THEN 'Unemployed but Very High Income — verify source'
        WHEN DTIRatio > 0.8
            THEN 'DTI > 80% — technically insolvent at origination'
        WHEN LoanAmount > Income * 10
            THEN 'Loan > 10x Annual Income — extreme leverage'
        WHEN CreditScore > 800 AND DefaultFlag = 1
            THEN 'Excellent Credit but Defaulted — potential fraud/external shock'
    END AS anomaly_reason
FROM feature_data
WHERE
    (EmploymentType = 'Unemployed' AND Income > 100000)
    OR DTIRatio > 0.8
    OR LoanAmount > Income * 10
    OR (CreditScore > 800 AND DefaultFlag = 1)
ORDER BY anomaly_reason;

-- =========
-- INSIGHTS
-- =========
/* ~1% customers flagged as anomalies -> high-impact risk group
 Dual anomalies (high DTI + low credit / high loan) -> highest default risk
 High income customers also show anomalies -> income ≠ safety
 Risk exists beyond segments -> needs anomaly-level screening
 Small group can drive losses -> focus = targeted control
 Business logic anomalies reveal data quality issues:
   -> Unemployed customers with very high income (unusual income source?)
   -> DTI > 80% means customer was technically already over-leveraged at origination
   -> Excellent credit + default = possible sudden external shock (medical, job loss) */


-- ============================================================================
-- SECTION 9 — CREDIT POLICY SIMULATION & PROFIT-RISK TRADEOFF ANALYSIS
-- ============================================================================

/* BUSINESS PROBLEM
 The bank has identified risk segments, but leadership needs to know:
 What happens to defaults and revenue if we apply stricter approval rules?
 What is the financial trade-off between rejecting high-risk customers
 (loss prevention) vs. losing their interest income (revenue cost)?

 OBJECTIVE
 Measure impact of approval rules on defaults (risk reduction)
 Measure impact on interest revenue (opportunity loss)
 Compare 3 strategies: Full, Moderate, Conservative
 Identify optimal cutoff balancing risk vs revenue
 Evaluate portfolio quality improvement (default rate change)
 Support data-driven credit policy decision */

-- STRATEGY SIMULATION — 3-scenario comparison
WITH decision_model AS (
    SELECT
        LoanID,
        LoanAmount,
        InterestRate,
        DefaultFlag,
        risk_score,
        LoanAmount * DefaultFlag AS actual_loss,
        LoanAmount * InterestRate / 100 * (1 - DefaultFlag) AS realized_interest,

        -- Strategy 1: Current (approve everyone)
        'Approved' AS strategy_current,

        -- Strategy 2: Conservative (reject risk score >= 6)
        CASE WHEN risk_score >= 6 THEN 'Rejected' ELSE 'Approved' END
        AS strategy_conservative,

        -- Strategy 3: Moderate (reject risk score >= 7 only)
        CASE WHEN risk_score >= 7 THEN 'Rejected' ELSE 'Approved' END
        AS strategy_moderate
    FROM feature_data
)

-- CURRENT STATE
SELECT
    'CURRENT — Approve All' AS strategy,
    COUNT(*) AS total_loans,
    SUM(DefaultFlag) AS total_defaults,
    ROUND(AVG(DefaultFlag) * 100, 2) AS default_rate_pct,
    SUM(actual_loss) AS total_loss,
    ROUND(SUM(realized_interest), 0) AS total_interest_earned
FROM decision_model

UNION ALL

-- CONSERVATIVE STRATEGY
SELECT
    'CONSERVATIVE — Reject Score ≥ 6',
    COUNT(*),
    SUM(CASE WHEN strategy_conservative = 'Approved' THEN DefaultFlag ELSE 0 END),
    ROUND(
        SUM(CASE WHEN strategy_conservative = 'Approved' THEN DefaultFlag ELSE 0 END)
        * 100.0
        / NULLIF(SUM(CASE WHEN strategy_conservative = 'Approved' THEN 1 ELSE 0 END), 0)
    , 2),
    SUM(CASE WHEN strategy_conservative = 'Approved' THEN actual_loss ELSE 0 END),
    ROUND(SUM(CASE WHEN strategy_conservative = 'Approved' THEN realized_interest ELSE 0 END), 0)
FROM decision_model

UNION ALL

-- MODERATE STRATEGY
SELECT
    'MODERATE — Reject Score ≥ 7',
    COUNT(*),
    SUM(CASE WHEN strategy_moderate = 'Approved' THEN DefaultFlag ELSE 0 END),
    ROUND(
        SUM(CASE WHEN strategy_moderate = 'Approved' THEN DefaultFlag ELSE 0 END)
        * 100.0
        / NULLIF(SUM(CASE WHEN strategy_moderate = 'Approved' THEN 1 ELSE 0 END), 0)
    , 2),
    SUM(CASE WHEN strategy_moderate = 'Approved' THEN actual_loss ELSE 0 END),
    ROUND(SUM(CASE WHEN strategy_moderate = 'Approved' THEN realized_interest ELSE 0 END), 0)
FROM decision_model;


-- REJECTED CUSTOMER PROFILE — what are we giving up?
-- Understanding who gets rejected is as important as who gets approved
WITH decision_model AS (
    SELECT
        *,
        CASE WHEN risk_score >= 6 THEN 'Rejected' ELSE 'Approved' END AS decision
    FROM feature_data
)
SELECT
    decision,
    COUNT(*) AS total_customers,
    ROUND(AVG(Income), 0) AS avg_income,
    ROUND(AVG(LoanAmount), 0) AS avg_loan_amount,
    ROUND(AVG(CreditScore), 0) AS avg_credit_score,
    ROUND(AVG(DTIRatio)::NUMERIC, 3) AS avg_dti,
    ROUND(AVG(DefaultFlag) * 100, 2) AS actual_default_rate_pct,
    -- Interest revenue foregone by rejecting this group
    ROUND(SUM(LoanAmount * InterestRate / 100 * (1 - DefaultFlag)), 0) AS foregone_interest
FROM decision_model
GROUP BY decision;

-- =========
-- INSIGHTS
-- =========
/* Conservative strategy reduces defaults from 11.61% → 9.82% → strong risk reduction
 But revenue drops significantly (~4.28B → ~2.59B) → high opportunity cost
 Moderate strategy gives balanced outcome → defaults 10.51% with revenue ~3.29B
 -> Better risk vs revenue trade-off than conservative
 Rejecting high-risk customers removes disproportionately risky loans
 Rejected segment default rate = 16.33% vs approved = 9.82%
 Approved portfolio becomes higher quality (lower default, stable revenue)
 Key insight → Moderate cutoff is optimal
 -> avoids excessive revenue loss while still improving risk
 Business decision → Do not fully tighten (conservative)
 -> use controlled filtering (moderate strategy)*/

-- ============================================================================
-- SECTION 10 — FINAL DECISION FRAMEWORK
-- ============================================================================

/* BUSINESS PROBLEM
 All analysis culminates in a single need: a structured, data-driven
 loan approval system that can be applied consistently at scale.
 Human judgment alone is inconsistent and does not scale to 255K+ loans.

 OBJECTIVE
 Build a final 3-tier decision framework using the risk score model.
 Validate the framework against actual default data.
 Provide strategic recommendations per tier.*/

-- FINAL LOAN DECISION FRAMEWORK
CREATE TEMP TABLE scored AS
SELECT
    LoanID,
    CreditScore,
    Income,
    LoanAmount,
    DTIRatio,
    MonthsEmployed,
    EmploymentType,
    LoanPurpose,
    InterestRate,
    DefaultFlag,
    risk_score,
    credit_score_band,
    dti_band,
    income_band,
    employment_stability,
    loan_burden_category,

    -- FINAL DECISION
    CASE
        WHEN risk_score >= 6 THEN 'REJECT — Auto Decline'
        WHEN risk_score BETWEEN 3 AND 5 THEN 'REVIEW — Manual Underwriting'
        ELSE 'APPROVE — Auto Approve'
    END AS lending_decision,

    -- DECISION REASON
    CASE
        WHEN risk_score >= 6 AND CreditScore < 500 AND DTIRatio > 0.4
            THEN 'Dual failure: Poor credit + Over-leveraged'
        WHEN risk_score >= 6 AND Income < 50000
            THEN 'Insufficient income for loan size'
        WHEN risk_score BETWEEN 3 AND 5 AND CreditScore < 580
            THEN 'Borderline credit — request collateral'
        WHEN risk_score BETWEEN 3 AND 5 AND DTIRatio > 0.4
            THEN 'High debt load — verify income'
        WHEN risk_score <= 2
            THEN 'Strong profile — fast-track'
        ELSE 'Standard review'
    END AS decision_reason

FROM feature_data;
-- FRAMEWORK PERFORMANCE VALIDATION
SELECT
    lending_decision,
    COUNT(*) AS total_customers,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct_of_portfolio,
    SUM(DefaultFlag) AS actual_defaults,
    ROUND(AVG(DefaultFlag) * 100, 2) AS actual_default_rate_pct,
    ROUND(AVG(CreditScore), 0) AS avg_credit_score,
    ROUND(AVG(Income), 0) AS avg_income,
    ROUND(AVG(DTIRatio)::NUMERIC, 3) AS avg_dti,
    SUM(LoanAmount * DefaultFlag) AS loss_in_segment,
    -- How much of total portfolio loss does this tier account for?
    ROUND(
        SUM(LoanAmount * DefaultFlag) * 100.0
        / SUM(SUM(LoanAmount * DefaultFlag)) OVER ()
    , 1) AS pct_of_total_loss
FROM scored
GROUP BY lending_decision
ORDER BY actual_default_rate_pct DESC;


-- STRATEGIC RECOMMENDATIONS PER TIER
SELECT
    lending_decision,
    decision_reason,
    COUNT(*) AS customer_count,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY lending_decision)
    , 1) AS pct_within_decision
FROM scored
GROUP BY lending_decision, decision_reason
ORDER BY lending_decision, customer_count DESC;


-- ==============================================================================
-- STRATEGIC RECOMMENDATIONS (EXECUTIVE SUMMARY)
-- ==============================================================================

/*TIER 1 — REJECT (risk score ≥ 6)
 → Auto-decline. Default rate in this group: ~16-26%
 → Do NOT extend credit without co-signer + collateral
 → Represent ~15% of applicants but ~35% of total losses

 TIER 2 — MANUAL REVIEW (risk score 3-5)
 → Require: last 3 months bank statements, income proof, employer verification
 → Apply risk-based pricing: interest rate +2-3% above standard
 → Consider loan amount cap at 3x monthly income
 → Default rate in this group: ~10-13% (manageable with controls)

 TIER 3 — APPROVE (risk score 0-2)
 → Fast-track approval with standard documentation
 → Offer loyalty products: lower rates, higher limits for repeat customers
 → Default rate: ~5-7% — well within acceptable risk tolerance
 → This is the growth segment — bank should increase market share here

 PRIORITY ACTIONS
 1. Immediately restrict Business and Auto loan approvals for risk score ≥ 5
    → These two purposes drive the highest absolute losses
 2. Introduce risk-adjusted pricing across all tiers (currently flat rate)
 3. Retrain scoring model quarterly — borrower behavior shifts with RBI rate cycles
 4. Flag DTI > 0.6 as automatic escalation trigger regardless of credit score
 5. Investigate Excellent credit + Default anomalies for potential fraud patterns */

-- ==============================================================================
