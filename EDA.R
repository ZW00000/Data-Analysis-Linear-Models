# ==============================================================================
# Ames Housing Dataset - Bayesian Normal Linear Modeling (Coursework)
# 完整实验代码汇总 (涵盖 Part A 与 Part B)
# ==============================================================================

# ------------------------------------------------------------------------------
# 准备工作：加载必要的包
# ------------------------------------------------------------------------------
# 如果未安装，请先取消注释运行：
# install.packages("corrplot")
# install.packages("BayesFactor")
library(corrplot)
library(BayesFactor)

# ------------------------------------------------------------------------------
# 第一部分 (Part A)：EDA and preprocess
# ------------------------------------------------------------------------------
cat("\n==================== [Part A] EDA and preprocess ====================\n")

# 1. 加载训练集数据
train_data <- read.csv("AmesTrain.csv", stringsAsFactors = FALSE)
cat("训练集初始维度:", dim(train_data)[1], "行", dim(train_data)[2], "列\n")

# 2. 因变量正态化：对售价取对数
train_data$Log_Sale_Price <- log(train_data$Sale_Price)

# 画图：转换前后的分布对比
par(mfrow = c(1, 2))
hist(train_data$Sale_Price, breaks = 20, probability = TRUE,
     main = "Original Sale_Price", xlab = "Sale Price", col = "lightgray")
lines(density(train_data$Sale_Price), col = "red", lwd = 2)
hist(train_data$Log_Sale_Price, breaks = 20, probability = TRUE,
     main = "Log-transformed Sale_Price", xlab = "log(Sale Price)", col = "lightblue")
lines(density(train_data$Log_Sale_Price), col = "blue", lwd = 2)
par(mfrow = c(1, 1))

# 3. 离散变量处理：将房屋质量与状况转换为有序因子
quality_levels <- c("Very_Poor", "Poor", "Fair", "Below_Average", "Average", 
                    "Above_Average", "Good", "Very_Good", "Excellent", "Very_Excellent")

# 转换整体质量
train_data$Overall_Qual <- factor(train_data$Overall_Qual, 
                                  levels = quality_levels, ordered = TRUE)

# 转换整体状况 (补充遗漏)
train_data$Overall_Cond <- factor(train_data$Overall_Cond, 
                                  levels = quality_levels, ordered = TRUE)

# 画图：质量与对数售价的箱线图
boxplot(Log_Sale_Price ~ Overall_Qual, data = train_data,
        main = "Log(Sale_Price) by Ordered Overall_Qual",
        xlab = "Overall Quality", ylab = "Log(Sale Price)",
        col = "lightblue", las = 2)

# 画图：状况与对数售价的箱线图
boxplot(Log_Sale_Price ~ Overall_Cond, data = train_data,
        main = "Log(Sale_Price) by Ordered Overall_Cond",
        xlab = "Overall Condition", ylab = "Log(Sale Price)",
        col = "lightblue", las = 2)

# 4. 多重共线性检查与热力图
numeric_vars <- sapply(train_data, is.numeric)
train_numeric <- train_data[, numeric_vars]
if("Sale_Price" %in% colnames(train_numeric)) { train_numeric$Sale_Price <- NULL }

cor_matrix_all <- cor(train_numeric, use = "pairwise.complete.obs")
corrplot(cor_matrix_all, method = "color", type = "upper", 
         tl.col = "black", tl.cex = 0.7, addCoef.col = "black", number.cex = 0.5,
         title = "Correlation Matrix", mar=c(0,0,1,0))

# 5. 高潜特征散点图矩阵
top_features <- c("Log_Sale_Price", "Gr_Liv_Area", "Total_Bsmt_SF", "Garage_Area")
pairs(train_numeric[, top_features], 
      main = "Scatterplot Matrix of Top Features",
      pch = 20, col = rgb(0.2, 0.4, 0.6, 0.4))

# ------------------------------------------------------------------------------
# 第二部分 (Part A)：模型拟合与选择 (Model Fitting & Selection)
# ------------------------------------------------------------------------------
cat("\n==================== [Part A] model fitting and selection ====================\n")

# 1. 准备干净的建模子集
candidate_vars <- c("Log_Sale_Price", "Gr_Liv_Area", "Total_Bsmt_SF", 
                    "Garage_Area", "Overall_Qual", "Overall_Cond", "Year_Built")
model_data <- na.omit(train_data[, candidate_vars])
cat("用于建模的干净数据维度:", dim(model_data)[1], "行", dim(model_data)[2], "列\n")

# 2. 贝叶斯因子 (Bayes Factor) 模型选择
# 注意：包含因子变量时必须使用 generalTestBF
cat("\n正在计算贝叶斯因子，请稍候...\n")
bf_all_models <- generalTestBF(Log_Sale_Price ~ ., data = model_data)
best_bf_models <- head(bf_all_models, 5)
cat("\n--- 排名前 5 的最优贝叶斯模型 ---\n")
print(best_bf_models)

# 3. BIC 模型选择 (作为对比与验证)
full_lm <- lm(Log_Sale_Price ~ ., data = model_data)
best_bic_model <- step(full_lm, direction = "backward", k = log(nrow(model_data)), trace = 0)
cat("\n--- BIC 选出的最终模型结构 ---\n")
print(summary(best_bic_model))

# ------------------------------------------------------------------------------
# 第三部分 (Part B)：后验分析与 OLS 对比 (Posterior Analysis & OLS)
# ------------------------------------------------------------------------------
cat("\n==================== [Part B] posterior analysis and prediction ====================\n")

# 1. 从最佳贝叶斯模型中抽取后验样本
set.seed(123)
posterior_samples <- posterior(best_bf_models[1], iterations = 10000)

# 2. 绘制参数的直方图
par(mfrow = c(2, 3))
params_to_plot <- c("mu", "Gr_Liv_Area-Gr_Liv_Area", "Total_Bsmt_SF-Total_Bsmt_SF", 
                    "Garage_Area-Garage_Area", "Year_Built-Year_Built")

for(param in params_to_plot) {
  if(param %in% colnames(posterior_samples)) {
    display_name <- strsplit(param, "-")[[1]][1]
    hist(posterior_samples[, param], breaks = 50, probability = TRUE,
         main = paste("Posterior of", display_name), 
         xlab = "Value", col = "skyblue", border = "white")
    abline(v = mean(posterior_samples[, param]), col = "red", lwd = 2)
  }
}
par(mfrow = c(1, 1))

# 3. 参数对比：贝叶斯 vs OLS
bayes_estimates <- colMeans(posterior_samples)
ols_estimates <- coef(full_lm)
cat("\n【贝叶斯后验均值】:\n")
print(round(bayes_estimates, 6))
cat("\n【OLS 模型系数】:\n")
print(round(ols_estimates, 6))


# ==============================================================================
# 第四部分 (Part B)：测试集预测与评估 (Test Set Prediction)
# ==============================================================================

# 1. 严格对齐测试集的预处理
test_data <- read.csv("AmesTest.csv", stringsAsFactors = FALSE)

# 处理训练集中未出现的 "Fair" 等级
# 将测试集中 Overall_Cond 为 "Fair" 的房屋，强制降级为 "Below_Average"，防止 OLS 预测崩溃
if("Fair" %in% test_data$Overall_Cond) {
  test_data$Overall_Cond[test_data$Overall_Cond == "Fair"] <- "Below_Average"
}

# 变量转换与训练集保持绝对一致
test_data$Overall_Qual <- factor(test_data$Overall_Qual, 
                                 levels = quality_levels, ordered = TRUE)
test_data$Overall_Cond <- factor(test_data$Overall_Cond, 
                                 levels = quality_levels, ordered = TRUE)

# 2. 提取训练集均值用于贝叶斯预测的中心化回调
train_means <- colMeans(model_data[, c("Gr_Liv_Area", "Total_Bsmt_SF", "Garage_Area", "Year_Built")])
bayes_coefs <- colMeans(posterior_samples)

# 3. 手动构建贝叶斯预测值 (对数尺度)
bayes_pred_log <- rep(bayes_coefs["mu"], nrow(test_data))
bayes_pred_log <- bayes_pred_log + 
  (test_data$Gr_Liv_Area - train_means["Gr_Liv_Area"]) * bayes_coefs["Gr_Liv_Area-Gr_Liv_Area"] +
  (test_data$Total_Bsmt_SF - train_means["Total_Bsmt_SF"]) * bayes_coefs["Total_Bsmt_SF-Total_Bsmt_SF"] +
  (test_data$Garage_Area - train_means["Garage_Area"]) * bayes_coefs["Garage_Area-Garage_Area"] +
  (test_data$Year_Built - train_means["Year_Built"]) * bayes_coefs["Year_Built-Year_Built"]

# 关键修复 2：同时叠加 Overall_Qual 和 Overall_Cond 两个分类变量的效应
for (i in 1:nrow(test_data)) {
  # 叠加 Quality 系数
  qual_level <- as.character(test_data$Overall_Qual[i])
  qual_coef_name <- paste0("Overall_Qual-", qual_level)
  if (qual_coef_name %in% names(bayes_coefs)) {
    bayes_pred_log[i] <- bayes_pred_log[i] + bayes_coefs[qual_coef_name]
  }
  
  # 叠加 Condition 系数
  cond_level <- as.character(test_data$Overall_Cond[i])
  cond_coef_name <- paste0("Overall_Cond-", cond_level)
  if (cond_coef_name %in% names(bayes_coefs)) {
    bayes_pred_log[i] <- bayes_pred_log[i] + bayes_coefs[cond_coef_name]
  }
}

# 4. 指数反变换与真实值提取
bayes_pred_price <- exp(bayes_pred_log)
actual_price <- test_data$Sale_Price

# 提取 OLS 模型预测值（因为修复了 "Fair"，现在 predict 函数不会报错了）
ols_pred_log <- predict(full_lm, newdata = test_data)
ols_pred_price <- exp(ols_pred_log)

# 5. 计算 RMSE 并在控制台打印
bayes_rmse <- sqrt(mean((bayes_pred_price - actual_price)^2, na.rm = TRUE))
ols_rmse <- sqrt(mean((ols_pred_price - actual_price)^2, na.rm = TRUE))

cat("\n--- 测试集预测误差评估 (RMSE) ---\n")
cat("贝叶斯模型 RMSE:", round(bayes_rmse, 2), "美元\n")
cat("OLS 模型 RMSE  :", round(ols_rmse, 2), "美元\n")

# 6. 绘制预测对比散点图
par(mfrow = c(1, 2))
plot(actual_price, bayes_pred_price, main = "Bayesian: Actual vs Predicted",
     xlab = "Actual Sale Price", ylab = "Predicted Sale Price",
     pch = 16, col = rgb(0.2, 0.6, 0.8, 0.6))
abline(a = 0, b = 1, col = "red", lwd = 2, lty = 2) 

plot(actual_price, ols_pred_price, main = "OLS: Actual vs Predicted",
     xlab = "Actual Sale Price", ylab = "Predicted Sale Price",
     pch = 16, col = rgb(0.8, 0.4, 0.2, 0.6))
abline(a = 0, b = 1, col = "red", lwd = 2, lty = 2)
par(mfrow = c(1, 1))

# ------------------------------------------------------------------------------
# 附赠：用于报告中 "Model Checking" 的残差图
# ------------------------------------------------------------------------------
par(mfrow = c(1, 2))
res <- residuals(best_bic_model)
qqnorm(res, main = "Normal Q-Q Plot"); qqline(res, col = "red")
plot(predict(best_bic_model), res, main = "Residuals vs Fitted", 
     xlab = "Fitted values", ylab = "Residuals")
par(mfrow = c(1, 1))


install.packages("knitr")


