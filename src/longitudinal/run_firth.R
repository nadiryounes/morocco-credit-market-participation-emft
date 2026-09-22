library(logistf)
library(dplyr)
library(marginaleffects)

df <- read.csv("B_ANALYSIS_DATA.csv")

# Missingness reporting
N_initial <- nrow(df)
df_cc <- na.omit(df[, c("CONSTRAINED_2023", "CREDIT_STATE_2019", "EMPLOYMENT_2019", "FIRM_AGE_2019", "EXPORTER_2019", "SECTOR_GROUP_2019", "STATE_R1_23")])
N_cc <- nrow(df_cc)

# Set references
df_cc$CREDIT_STATE_2019 <- factor(df_cc$CREDIT_STATE_2019, levels=c("NO_FINANCING_NEED", "CONSTRAINED_NON_APPLICANT", "CREDIT_APPLICANT"))
df_cc$SECTOR_GROUP_2019 <- factor(df_cc$SECTOR_GROUP_2019, levels=c("OTHER_SERVICES", "MANUFACTURING", "RETAIL"))
df_cc$log1p_EMP <- log1p(df_cc$EMPLOYMENT_2019)
df_cc$log1p_AGE <- log1p(df_cc$FIRM_AGE_2019)

# Fit primary model
mod <- logistf(CONSTRAINED_2023 ~ CREDIT_STATE_2019 + log1p_EMP + log1p_AGE + EXPORTER_2019 + SECTOR_GROUP_2019, data=df_cc)

# Extract results
summ <- summary(mod)
ci <- confint(mod)
res <- data.frame(
  Term = names(mod$coefficients),
  Beta = summ$coefficients,
  OR = exp(summ$coefficients),
  CI_Lower = exp(ci[,1]),
  CI_Upper = exp(ci[,2]),
  P_Value = summ$prob
)
write.csv(res, "B7_FIRTH_PRIMARY_MODEL.csv", row.names=FALSE)

# Diagnostics
diag <- data.frame(
  Metric = c("Convergence", "Iterations", "N_Final", "Events"),
  Value = c(mod$converged, mod$iter, N_cc, sum(df_cc$CONSTRAINED_2023))
)
write.csv(diag, "B8_FIRTH_MODEL_DIAGNOSTICS.csv", row.names=FALSE)

# Predictions - Baseline state
# Use avg_predictions from marginaleffects
pred_state <- avg_predictions(mod, variables = "CREDIT_STATE_2019")
write.csv(pred_state, "B9_ADJUSTED_PROBABILITY_BASELINE_STATE.csv", row.names=FALSE)

# Predictions - Continuous
emp_q <- quantile(df_cc$EMPLOYMENT_2019, probs=c(0.25, 0.5, 0.75))
age_q <- quantile(df_cc$FIRM_AGE_2019, probs=c(0.25, 0.5, 0.75))

df_cc$EMP_q <- df_cc$EMPLOYMENT_2019 # Dummy
pred_emp <- avg_predictions(mod, variables = list(log1p_EMP = log1p(emp_q)))
pred_emp$Variable <- "EMPLOYMENT"
pred_emp$Value <- emp_q

pred_age <- avg_predictions(mod, variables = list(log1p_AGE = log1p(age_q)))
pred_age$Variable <- "FIRM_AGE"
pred_age$Value <- age_q

cols <- c("Variable", "Value", "estimate", "std.error", "conf.low", "conf.high", "p.value")
write.csv(rbind(pred_emp[, cols], pred_age[, cols]), "B10_CONTINUOUS_PREDICTOR_PROBABILITIES.csv", row.names=FALSE)

# Robustness R1
df_cc_r1 <- df_cc[df_cc$STATE_R1_23 != "AMBIGUOUS", ]
df_cc_r1$CONSTRAINED_R1 <- ifelse(df_cc_r1$STATE_R1_23 == "CONSTRAINED_NON_APPLICANT", 1, 0)
mod_r1 <- logistf(CONSTRAINED_R1 ~ CREDIT_STATE_2019 + log1p_EMP + log1p_AGE + EXPORTER_2019 + SECTOR_GROUP_2019, data=df_cc_r1)

summ_r1 <- summary(mod_r1)
ci_r1 <- confint(mod_r1)
res_r1 <- data.frame(
  Term = names(mod_r1$coefficients),
  Beta = summ_r1$coefficients,
  OR = exp(summ_r1$coefficients),
  CI_Lower = exp(ci_r1[,1]),
  CI_Upper = exp(ci_r1[,2]),
  P_Value = summ_r1$prob
)
write.csv(res_r1, "B12_ROBUSTNESS_R1_MODEL.csv", row.names=FALSE)
print("Firth models done.")
