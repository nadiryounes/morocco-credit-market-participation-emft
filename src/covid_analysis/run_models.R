library(logistf)
library(marginaleffects)
library(dplyr)

df <- read.csv("C_ANALYSIS_DATA.csv")

# Set references
df$CREDIT_STATE_2019 <- factor(df$CREDIT_STATE_2019, levels = c("NO_FINANCING_NEED", "CONSTRAINED_NON_APPLICANT", "CREDIT_APPLICANT"))
df$SECTOR_GROUP_2019 <- factor(df$SECTOR_GROUP_2019, levels = c("OTHER_SERVICES", "MANUFACTURING", "RETAIL"))
df$log1p_EMP <- log1p(df$EMPLOYMENT_2019)
df$log1p_AGE <- log1p(df$FIRM_AGE_2019)

run_mod <- function(formula, data, out_file) {
  # complete case
  vars <- all.vars(formula)
  d <- data[, vars]
  d <- d[complete.cases(d), ]
  
  if (sum(d$CONSTRAINED_2023) < 20 || sum(1 - d$CONSTRAINED_2023) < 20) {
    cat("Insufficient events for formula", as.character(formula), "\n")
    return(NULL)
  }
  
  mod <- logistf(formula, data = d)
  
  # Export coefficients
  coefs <- mod$coefficients
  ci <- confint(mod)
  pvals <- mod$prob
  
  res <- data.frame(
    Term = names(coefs),
    Estimate = coefs,
    OR = exp(coefs),
    Conf_Low = ci[, 1],
    Conf_High = ci[, 2],
    P_Value = pvals
  )
  write.csv(res, out_file, row.names = FALSE)
  return(mod)
}

# C4: Primary Arrears
mod_arr <- run_mod(CONSTRAINED_2023 ~ CREDIT_STATE_2019 + FINANCIAL_ARREARS, df, "C4_PRIMARY_ARREARS_FIRTH.csv")

# C5: Primary Liquidity
mod_liq <- run_mod(CONSTRAINED_2023 ~ CREDIT_STATE_2019 + LIQUIDITY_STRESS, df, "C5_PRIMARY_LIQUIDITY_FIRTH.csv")

# C6: Secondary Adjusted Arrears
mod_arr_adj <- run_mod(CONSTRAINED_2023 ~ CREDIT_STATE_2019 + FINANCIAL_ARREARS + log1p_EMP + log1p_AGE, df, "C6_SECONDARY_ADJUSTED_ARREARS.csv")

# C7: Secondary Adjusted Liquidity
mod_liq_adj <- run_mod(CONSTRAINED_2023 ~ CREDIT_STATE_2019 + LIQUIDITY_STRESS + log1p_EMP + log1p_AGE, df, "C7_SECONDARY_ADJUSTED_LIQUIDITY.csv")

# C8: Adjusted Probabilities
# Standadized adjusted predicted probabilities for COVID exposure = 0 vs 1
# Averaging over the empirical distribution of CREDIT_STATE_2019.
get_probs <- function(mod, exp_var, d, label) {
    if (is.null(mod)) return(NULL)
    vars <- all.vars(mod$formula)
    df_mod <- d[, vars]
    df_mod <- df_mod[complete.cases(df_mod), ]
    
    # We want average predictions by exposure
    vlist <- list()
    vlist[[exp_var]] <- c(0, 1)
    
    pred <- avg_predictions(mod, variables = vlist, newdata = df_mod)
    pred_df <- as.data.frame(pred)
    pred_df$Model <- label
    pred_df$Exposure <- exp_var
    return(pred_df)
}

p1 <- get_probs(mod_arr, "FINANCIAL_ARREARS", df, "Primary")
p2 <- get_probs(mod_liq, "LIQUIDITY_STRESS", df, "Primary")
p3 <- get_probs(mod_arr_adj, "FINANCIAL_ARREARS", df, "Secondary")
p4 <- get_probs(mod_liq_adj, "LIQUIDITY_STRESS", df, "Secondary")

cols_p <- c("Model", "Exposure", "Value", "estimate", "std.error", "conf.low", "conf.high", "p.value")
if(TRUE) {
    # Note: Value is not correct name for variable in avg_predictions?
    # avg_predictions returns the variable name as a column.
    # We can reconstruct it:
    out_p <- data.frame()
    if(!is.null(p1)) out_p <- rbind(out_p, data.frame(Model="Primary", Exposure="FINANCIAL_ARREARS", Level=p1$FINANCIAL_ARREARS, Estimate=p1$estimate, Conf_Low=p1$conf.low, Conf_High=p1$conf.high))
    if(!is.null(p2)) out_p <- rbind(out_p, data.frame(Model="Primary", Exposure="LIQUIDITY_STRESS", Level=p2$LIQUIDITY_STRESS, Estimate=p2$estimate, Conf_Low=p2$conf.low, Conf_High=p2$conf.high))
    if(!is.null(p3)) out_p <- rbind(out_p, data.frame(Model="Secondary", Exposure="FINANCIAL_ARREARS", Level=p3$FINANCIAL_ARREARS, Estimate=p3$estimate, Conf_Low=p3$conf.low, Conf_High=p3$conf.high))
    if(!is.null(p4)) out_p <- rbind(out_p, data.frame(Model="Secondary", Exposure="LIQUIDITY_STRESS", Level=p4$LIQUIDITY_STRESS, Estimate=p4$estimate, Conf_Low=p4$conf.low, Conf_High=p4$conf.high))
    write.csv(out_p, "C8_ADJUSTED_PROBABILITIES.csv", row.names=FALSE)
}

# C9: Multiple Testing
if(!is.null(mod_arr) && !is.null(mod_liq)){
    p_arr <- mod_arr$prob["FINANCIAL_ARREARS"]
    p_liq <- mod_liq$prob["LIQUIDITY_STRESS"]
    p_adj <- p.adjust(c(p_arr, p_liq), method="holm")
    c9 <- data.frame(
        Exposure = c("FINANCIAL_ARREARS", "LIQUIDITY_STRESS"),
        Raw_P = c(p_arr, p_liq),
        Holm_Adjusted_P = p_adj
    )
    write.csv(c9, "C9_MULTIPLE_TESTING.csv", row.names=FALSE)
}

# C10: Exploratory
mod_dig <- run_mod(CONSTRAINED_2023 ~ CREDIT_STATE_2019 + DIGITAL_ADAPTATION, df, "C10_EXP_DIGITAL.csv")
mod_gov <- run_mod(CONSTRAINED_2023 ~ CREDIT_STATE_2019 + GOV_SUPPORT, df, "C10_EXP_GOV.csv")
mod_cred <- run_mod(CONSTRAINED_2023 ~ CREDIT_STATE_2019 + CREDIT_SEEKING, df, "C10_EXP_CREDIT.csv")

p_dig <- if(!is.null(mod_dig)) mod_dig$prob["DIGITAL_ADAPTATION"] else NA
p_gov <- if(!is.null(mod_gov)) mod_gov$prob["GOV_SUPPORT"] else NA
p_cred <- if(!is.null(mod_cred)) mod_cred$prob["CREDIT_SEEKING"] else NA
p_exp <- c(p_dig, p_gov, p_cred)
p_exp_adj <- p.adjust(p_exp, method="BH")

c10_res <- data.frame(
    Exposure = c("DIGITAL_ADAPTATION", "GOV_SUPPORT", "CREDIT_SEEKING"),
    Raw_P = p_exp,
    BH_FDR = p_exp_adj
)
write.csv(c10_res, "C10_EXPLORATORY_COVID_MODELS.csv", row.names=FALSE)

# C11: Strict Complete Wave
df_strict <- df[df$STRICT_SAMPLE == 1, ]
run_mod(CONSTRAINED_2023 ~ CREDIT_STATE_2019 + FINANCIAL_ARREARS, df_strict, "C11_STRICT_ARREARS.csv")
run_mod(CONSTRAINED_2023 ~ CREDIT_STATE_2019 + LIQUIDITY_STRESS, df_strict, "C11_STRICT_LIQUIDITY.csv")

# C12: EVER
run_mod(CONSTRAINED_2023 ~ CREDIT_STATE_2019 + EVER_ARREARS, df, "C12_EVER_ARREARS.csv")
run_mod(CONSTRAINED_2023 ~ CREDIT_STATE_2019 + EVER_LIQUIDITY, df, "C12_EVER_LIQUIDITY.csv")

cat("Models finished.\n")
