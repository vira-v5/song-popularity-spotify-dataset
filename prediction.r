
## Load packages ---------------------------------------------------------------
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(reshape2)

library(MLmetrics)
#library(e1071)

if (!require(MLeval)) {
  install.packages("MLeval")
  library(MLeval)
}


## Load data -------------------------------------------------------------------
data <- read.csv("input/datasets_500146_927168_Spotify-2000.csv",
                 stringsAsFactors = FALSE, fileEncoding="UTF-8-BOM")

## EDA -------------------------------------------------------------------------

## Renaming columns for more intuitive inspection

df <- data %>% 
  rename(
    "Length" = "Length..Duration.", 
    "Loudness" = "Loudness..dB.",
    "BPM" = "Beats.Per.Minute..BPM.", 
    "Genre" = "Top.Genre"
  ) %>%
  # removing comas from 'Length' so we can turn it into integer
  mutate(Length, Length = gsub(x = Length, pattern = ",", replacement = "")) %>%
  # Turn 'Length' variable into integer
  mutate(Length, Length = as.integer(Length)) 

## Inspecting the data
str(df)

## Counting missing values
cm <- colSums(is.na(df))

cm

## Descriptive statistics
summary(df)


## Visualization ---------------------------------------------------------------

## Histogram
df_hist <- df %>% 
  select(c(6:15)) %>%
  gather() %>%
  ggplot(aes(x=value,fill=key)) +
  geom_histogram(binwidth = 5, color = "black") +
  facet_wrap(~ key, scales = "free") +
  theme_minimal()

df_hist

## Checking correlation between numerical variables
## Generating dataframe for correlation matrix
corr_df <- df %>%
  select_if(is.numeric) %>%
  select(-Index)

## Generating correlation matrix

corr_matrix <- round(cor(corr_df), 2)

## Scatterplots
energy_loudness_plot <- ggplot(data = df, aes(x = Energy, y = Loudness)) +
  geom_jitter(size = 0.4) +
  geom_smooth(method='lm', formula= y~x)

energy_loudness_plot

energy_acousticness_plot <- ggplot(data = df, 
                                   aes(x = Energy, y = Acousticness)) +
  geom_jitter(size = 0.4) +
  geom_smooth(method='lm', formula= y~x)

energy_acousticness_plot

danceability_valence_plot <- ggplot(data = df, 
                                    aes(x = Danceability, y = Valence)) +
  geom_jitter(size = 0.4) + 
  geom_smooth(method='lm', formula= y~x)

danceability_valence_plot

loudness_acousticness_plot <- ggplot(data = df, 
                                     aes(x = Loudness, y = Acousticness)) +
  geom_jitter(size = 0.4) + 
  geom_smooth(method='lm', formula= y~x)

loudness_acousticness_plot

popularity_loudness_plot <- ggplot(data = df, 
                                   aes(x = Popularity, y = Loudness)) +
  geom_jitter(size = 0.4) + 
  geom_smooth(method='lm', formula= y~x)

popularity_loudness_plot


## Data Pre-processing ---------------------------------------------------------

## Functions
 
## Create data frame with added variables
music <- df %>%

  ## Selecting the variables
  select(c(
    "Popularity", "Speechiness", "Acousticness", "Length",
    "Valence", "Energy", "Loudness", "Danceability", "BPM",
    "Year", "Genre", "Title"
  )) %>%

  ## Create categorical Year variable
  mutate(Year_cat = ifelse(Year > 2010, "2011-2019",
    ifelse(Year > 2000, "2001-2010",
      ifelse(Year > 1990, "1991-2000",
        ifelse(Year > 1980, "1981-1990",
          ifelse(Year > 1970, "1971-1980",
            ifelse(Year > 1960, "1961-1970", "1956-1960")
          )
        )
      )
    )
  )) %>%
  mutate(Year_cat = as.factor(Year_cat)) %>%
  mutate(Year = as.character(Year)) %>% 

  ## Creating normalizes columns for all numeric variables
  mutate(across(where(is.numeric),
    .fns = list(norm = ~ scale(.x))
  )) %>%

  ## Create Length_Name_Track variable
  mutate(Length_Name_Track = nchar(Title)) %>%

  ## Create Binarized Popularity variable
  mutate(Popularity_Bi = ifelse(Popularity > median(Popularity),
    "Popular", "Unpopular"
  )) %>%
  mutate(Popularity_Bi = as.factor(Popularity_Bi)) %>%

  ## Create Multiclass Popularity variable using Quantiles
  mutate(Popularity_Multi = ifelse(Popularity > quantile(Popularity, 0.66),
    "Popular",
    ifelse(Popularity > quantile(Popularity, 0.33), "Normal", "Unpopular")
  )) %>%
  mutate(Popularity_Multi = as.factor(Popularity_Multi))


## Outlier Analysis ------------------------------------------------------------
## data frame with just scaled variables
just_norm <- music %>% 
  select(contains("norm"))

## checking the boxplot to see whether there are potential outliers
music_boxplot <- just_norm %>% 
  gather() %>%
  ggplot() +
  geom_boxplot(mapping = aes(x= key, y=value, fill=key), color = "black") +
  facet_wrap(~ key, scales = "free") +
  theme_minimal()

music_boxplot

## Finding multivariate outliers with Cook's distance
mod <- lm(Popularity_norm ~ ., data=just_norm)
cooksd <- cooks.distance(mod)

## Plot cook's distance
plot(cooksd, pch="*", cex=2, main="Influential Observation by Cooks distance")  
## Add cutoff line
abline(h = 4/nrow(just_norm), col="red")  
## Add labels
text(x=1:length(cooksd)+1, y=cooksd, 
     labels=ifelse(cooksd> 4/nrow(just_norm),names(cooksd),""), 
     col="red")  

## Data frame with influential observations removed
music_final <- music[-as.numeric(names(cooksd)[(cooksd > 4/nrow(just_norm))]), ]  

## Histogram after influential observations removed
df_hist2 <- music_final %>% 
  select(contains("norm")) %>%
  gather() %>%
  ggplot(aes(x=value,fill=key)) +
  geom_histogram(binwidth = 0.3, color = "black") +
  facet_wrap(~ key, scales = "free") +
  theme_minimal()

df_hist2


## Prediction ------------------------------------------------------------------

## Function to plot confusion matrix
confusionMPlot <- function(confMmod) {
  confuseTable1 <- data.frame(confMmod$table)
  plotTable1 <- confuseTable1 %>%
    mutate(
      Predictions_Results =
        ifelse(confuseTable1$Prediction != confuseTable1$Reference,
          "Incorrect Prediction", "Correct Prediction"
        )
    ) %>%
    group_by(Reference)
  ggplot(
    data = plotTable1,
    aes(x = Prediction, y = Reference, fill = Predictions_Results)
  ) +
    geom_tile() +
    geom_text(aes(label = Freq),
      vjust = .5, fontface = "bold", alpha = 1,colour = "white") +
    theme_minimal() +
    ggtitle("Confusion Matrix")
}

## Function to extract summary metrics for multiclass classification
evaluation_details_multi <- function(cm) {
  plot(c(100, 0), c(100, 0), type = "n", xlab = "", ylab = "",
    main = "Model Performance Metrics", xaxt = "n", yaxt = "n")
  text(10, 85, "Sensitivity", cex = 1.2, font = 2)
  text(10, 70, round(as.numeric(mean(cm$byClass[1:3])), 3), cex = 1.2)
  text(30, 85, "Specificity", cex = 1.2, font = 2)
  text(30, 70, round(as.numeric(mean(cm$byClass[4:6])), 3), cex = 1.2)
  text(50, 85, "Precision", cex = 1.2, font = 2)
  text(50, 70, round(as.numeric(mean(cm$byClass[7:9])), 3), cex = 1.2)
  text(70, 85, "Recall", cex = 1.2, font = 2)
  text(70, 70, round(as.numeric(mean(cm$byClass[10:12])), 3), cex = 1.2)
  text(90, 85, "F1", cex = 1.2, font = 2)
  text(90, 70, round(as.numeric(mean(cm$byClass[13:15])), 3), cex = 1.2)

  ## Add the accuracy information
  text(30, 40, names(cm$overall[1]), cex = 1.5, font = 2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex = 1.4)
  text(70, 40, names(cm$overall[2]), cex = 1.5, font = 2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex = 1.4)
}

## Function to extract summary metrics for binary classification
evaluation_details <- function(cm) {
  plot(c(100, 0), c(100, 0),
    type = "n", xlab = "", ylab = "",
    main = "Model Performance Metrics", xaxt = "n", yaxt = "n")
  text(10, 85, names(cm$byClass[1]), cex = 1.2, font = 2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex = 1.2)
  text(30, 85, names(cm$byClass[2]), cex = 1.2, font = 2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex = 1.2)
  text(50, 85, names(cm$byClass[5]), cex = 1.2, font = 2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex = 1.2)
  text(70, 85, names(cm$byClass[6]), cex = 1.2, font = 2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex = 1.2)
  text(90, 85, names(cm$byClass[7]), cex = 1.2, font = 2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex = 1.2)

  # add in the accuracy information
  text(30, 35, names(cm$overall[1]), cex = 1.5, font = 2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex = 1.4)
  text(70, 35, names(cm$overall[2]), cex = 1.5, font = 2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex = 1.4)
}

# Function to plot model performance of 2 models 
create_comparison_plot <- function(df, model1, model2){
  
  new_df = filter(df, Models == model1 | Models== model2)
  
  plot = ggplot(new_df) + 
    geom_bar(aes(x = Models, y = value, fill = variable),
             stat = "identity", position = "dodge", width =  0.7) +
    scale_fill_manual("Metrics\n",
                      values = c("salmon2", "brown1", "brown4"),
                      labels = c(" Accuracy", " Recall", "Precision")) +
    labs(x = "\nModels", y = "Metric value\n", 
         title = "Summary Model Performance") +
    theme_minimal()
  
  plot
}

## Function to turn important metrics into a vector
get_metrics_multi = function(pred_mod) {
  sensi = round(mean(pred_mod$byClass[1:3]),3)
  speci = round(mean(pred_mod$byClass[4:6]),3)
  prec = round(mean(pred_mod$byClass[7:9]),3)
  rec = round(mean(pred_mod$byClass[10:12]),3)
  f1 = round(mean(pred_mod$byClass[13:15]),3)
  acc = round(pred_mod$overall[1],3)
  kappa = round(pred_mod$overall[2],3)
  metrics = c(Sensitivity = sensi, Specificity = speci, Precision = prec,
              Recall = rec,F1 = f1, acc, kappa)
  return(data.frame(metrics))
}

get_metrics_bi = function(pred_mod) {
  sensi = round(pred_mod$byClass[1],3)
  speci = round(pred_mod$byClass[2],3)
  prec = round(pred_mod$byClass[5],3)
  rec = round(pred_mod$byClass[6],3)
  f1 = round(pred_mod$byClass[7],3)
  acc = round(pred_mod$overall[1],3)
  kappa = round(pred_mod$overall[2],3)
  metrics = c(sensi, speci, prec, rec, f1, acc, kappa)
  return(data.frame(metrics))
}

### Visualization to decide which features to include in predictive models

## Create dataframe with only normalized features
norm_df <- music_final  %>%
  select(contains("norm"), -Length_Name_Track)

## Df with significant normalized predictors 
norm_final <- music_final %>%
  select(Year_cat ,contains("norm"), -BPM_norm)

## Scatterplots between Popularity and significant normalized predictors
facet_scatter_plot_norm <- norm_final %>%
  select(-Year_cat) %>%
  gather(key = "key", value = "Predictors", -Popularity_norm) %>%
  ggplot(aes(Predictors, Popularity_norm)) +
  geom_point() +
  geom_smooth(method = "lm") +
  facet_wrap(. ~ key, scales = "free") +
  theme_minimal() +
  labs(
    title = "Scatter Plots between predictors and Popularity",
    x = "Normalized Predictors",
    y = "Normalized Song Popularity"
  )

facet_scatter_plot_norm

## Facet plots between Popularity and significant normalized predictors with 
## theinteraction effect of year
scatter_inter_plot <- norm_final %>%
  gather(
    key = "key", value = "Predictors",
    -c(Popularity_norm, Year_cat)
  ) %>%
  ggplot() +
  aes(
    x = Predictors, y = Popularity_norm, group = Year_cat,
    color = Year_cat
  ) +
  geom_smooth(method = "lm") +
  facet_wrap(. ~ key, scales = "free") +
  theme_minimal() +
  labs(
    title = "The relationship between the predictors and song popularity \n 
       with the moderator Year",
    x = "Normalized predictors",
    y = "Normalized Song Popularity"
  )

scatter_inter_plot

## Predictive Models -----------------------------------------------------------

## 1. KNN classifier
## Model 1.1: KNN Multiclass classification
## Optimizing for accuracy

## Selecting data frame for KNN Multiclass classification
KNN_df_multi <- select(
  music_final, Popularity_Multi, Speechiness_norm,
  Acousticness_norm, Length_norm,
  Valence_norm, Loudness_norm, Danceability_norm,
  Year_cat, Energy_norm
)

## Generating a train and test set for Model 1.1.1
set.seed(8)
trn_index_multi <- createDataPartition(
  y = KNN_df_multi$Popularity_Multi,
  p = 0.70, list = FALSE
)
trn_knn_multi <- KNN_df_multi[trn_index_multi, ]
tst_knn_multi <- KNN_df_multi[-trn_index_multi, ]

## Train model 1.1.1
set.seed(8)
Popularity_multi_knn1.1 <- train(Popularity_Multi ~ .,
  method = "knn",
  data = trn_knn_multi,
  trControl = trainControl(
    method = "cv",
    number = 10,
    returnResamp = "all"
  ),
  tuneGrid = data.frame(k = c(2, 3, 5, 8, 10, 20))
)

## Plotting accuracy vs k-neighbours
plot(Popularity_multi_knn1.1)

## Evaluation on test set model 1.1.1
predicted_outcomes_mod1.1 <- predict(Popularity_multi_knn1.1, tst_knn_multi)
knn_confMmod1.1 <-
  confusionMatrix(predicted_outcomes_mod1.1, tst_knn_multi$Popularity_Multi)

## plot confusion matrix model 1.1.1
confusionMPlot(knn_confMmod1.1)

## Get overall statistics
evaluation_details_multi(knn_confMmod1.1)

## Train model 1.1.2
set.seed(8)
Popularity_multi_knn1.2 <- train(
  Popularity_Multi ~ Speechiness_norm + 
    Energy_norm + 
    Loudness_norm + 
    Danceability_norm,
  method = "knn",
  data = trn_knn_multi,
  trControl = trainControl(method = "cv",number = 10,returnResamp = "all"),
                              tuneGrid = data.frame(k = c(2, 3, 5, 8, 10, 20))
)

## Plotting accuracy vs k-neighbours
plot(Popularity_multi_knn1.2)

## Evaluation on test set model 1.1.2
predicted_outcomes_mod1.2 <- predict(Popularity_multi_knn1.2, tst_knn_multi)
knn_confMmod1.2 <-
  confusionMatrix(predicted_outcomes_mod1.2, tst_knn_multi$Popularity_Multi)

## plot confusion matrix model 1.1.2
confusionMPlot(knn_confMmod1.2)

# Get overall statistics
evaluation_details_multi(knn_confMmod1.2)

## Model 1.2 - KNN Binary classification
# Optimizing for recall

# Selecting data frame for KNN binary
KNN_df_bi <- select(
  music_final, Popularity_Bi, Speechiness_norm,
  Acousticness_norm, Length_norm, Valence_norm, Loudness_norm,
  Danceability_norm, Year_cat, Energy_norm
)

## Generating a train and testsest for Model 1.2.1
set.seed(8)
trn_index_bi <- createDataPartition(
  y = KNN_df_bi$Popularity_Bi,
  p = 0.70, list = FALSE
)
trn_knn_bi <- KNN_df_bi[trn_index_bi, ]
tst_knn_bi <- KNN_df_bi[-trn_index_bi, ]

## Training model 1.2.1
set.seed(8)
Popularity_Bi_knn2.1 <- train(Popularity_Bi ~ .,
  method = "knn", data = trn_knn_bi,
  trControl = trainControl(
    method = "cv", number = 5,
    classProbs = TRUE,
    summaryFunction = prSummary
  ),
  metric = "Recall",
  tuneGrid = data.frame(k = c(2, 3, 5, 8, 10, 20))
)

## Plot Recall vs K-neighbors
plot(Popularity_Bi_knn2.1)

## Evaluation on test set model 1.2.1
predicted_outcomes_mod2.1 <- predict(Popularity_Bi_knn2.1, tst_knn_bi)
knn_confMmod2.1 <-
  confusionMatrix(predicted_outcomes_mod2.1, tst_knn_bi$Popularity_Bi)

evaluation_details(knn_confMmod2.1)

## Plot confusion matrix model 1.2.1
confusionMPlot(knn_confMmod2.1)

## ROC curve model 1.2.1
set.seed(8)
ROC_mod1.2.1 <- train(Popularity_Bi ~ .,
  method = "knn", data = trn_knn_bi,
  trControl = trainControl(
    method = "cv",
    number = 10,
    summaryFunction = twoClassSummary,
    savePredictions = TRUE,
    classProbs = TRUE
  ),
  metric = "Recall", tuneGrid = data.frame(k = c(2, 3, 5, 8, 10, 20))
)
ROC_plot_mod1.2.1 <- evalm(ROC_mod1.2.1)

## Training model 1.2.2
set.seed(8)
Popularity_Bi_knn2.2 <- train(Popularity_Bi ~ Speechiness_norm + Energy_norm
                              + Loudness_norm + Danceability_norm,
                              method = "knn", data = trn_knn_bi,
                              trControl = trainControl(
                                method = "cv", number = 5,
                                classProbs = TRUE,
                                summaryFunction = prSummary
                              ),
                              metric = "Recall",
                              tuneGrid = data.frame(k = c(2, 3, 5, 8, 10, 20))
)

## Plot Recall vs K-neighbors
plot(Popularity_Bi_knn2.2)

## Evaluation on test set model 1.2.2
predicted_outcomes_mod2.2 <- predict(Popularity_Bi_knn2.2, tst_knn_bi)
knn_confMmod2.2 <-
  confusionMatrix(predicted_outcomes_mod2.2, tst_knn_bi$Popularity_Bi)

evaluation_details(knn_confMmod2.2)

## Plot confusion matrix model 1.2.2
confusionMPlot(knn_confMmod2.2)

# ROC curve model 1.2.2
set.seed(8)
ROC_mod1.2.2 <- train(Popularity_Bi ~ Speechiness_norm + Energy_norm
                    + Loudness_norm + Danceability_norm,
                    method = "knn", data = trn_knn_bi,
                    trControl = trainControl(
                      method = "cv",
                      number = 10,
                      summaryFunction = twoClassSummary,
                      savePredictions = TRUE,
                      classProbs = TRUE
                    ),
                    metric = "Recall", tuneGrid = data.frame(k = c(2, 3, 5, 8, 10, 20))
)
ROC_plot_mod1.2.2 <- evalm(ROC_mod1.2.2)

## Logistic Regression Classifier
## Model 2.1 Logistic regression

## Logistic regression df
Logistic_reg_df <- select(
  music_final, Popularity_Bi, Speechiness_norm,
  Acousticness_norm, Length_norm,
  Valence_norm, Loudness_norm, Danceability_norm,
   Year_cat, Energy_norm
)

## Generating a train -and testsest for model 2.1
set.seed(8)
trn_index_lgr <- createDataPartition(
  y = Logistic_reg_df$Popularity_Bi,
  p = 0.70, list = FALSE
)
trn_lgr <- Logistic_reg_df[trn_index_lgr, ]
tst_lgr <- Logistic_reg_df[-trn_index_lgr, ]

## training model 2.1
set.seed(8)
Popularity_lgr2.1 <- train(Popularity_Bi ~ .,
  method = "glm",
  family = binomial(link = "logit"), 
  data = trn_lgr,
  trControl = trainControl(method = "cv", number = 10)
)

## evaluation on test set model 2.1
predicted_outcomes_mod2.1 <- predict(Popularity_lgr2.1, tst_lgr)
lgr_confMmod2.1 <- confusionMatrix(predicted_outcomes_mod2.1, 
                                   tst_lgr$Popularity_Bi)

evaluation_details(lgr_confMmod2.1)

## Plot confusion matrix model 2.1
confusionMPlot(lgr_confMmod2.1)

## ROC curve model 2.1
set.seed(8)
ROC_mod2.1 <- train(Popularity_Bi ~ .,
  method = "glm",
  family = binomial(link = "logit"), data = trn_lgr,
  trControl = trainControl(
    method = "cv", number = 10,
    summaryFunction = twoClassSummary,
    savePredictions = TRUE,
    classProbs = TRUE
  )
)
ROC_plot_mod2.1 <- evalm(ROC_mod2.1)

## Model 2.2 - Logistic regression with interaction

## Training model 2.2
set.seed(8)
Popularity_lgr_inter2.2 <- train(Popularity_Bi ~ . + Danceability_norm:Year_cat +
  Energy_norm:Year_cat + Loudness_norm:Year_cat,
method = "glm",
family = binomial(link = "logit"), data = trn_lgr,
trControl = trainControl(method = "cv", number = 5)
)

## Evaluation on test set model 2.2
predicted_outcomes_mod2.2 <- predict(Popularity_lgr_inter2.2, tst_lgr)
confMmod2.2 <- confusionMatrix(predicted_outcomes_mod2.2, tst_lgr$Popularity_Bi)

evaluation_details(confMmod2.2)

## Plot confusion matrix model 2.2
confusionMPlot(confMmod2.2)

## ROC curve model 2.2
set.seed(8)
ROC_mod2.2 <- train(Popularity_Bi ~ . + Danceability_norm:Year_cat +
  Energy_norm:Year_cat + Loudness_norm:Year_cat,
method = "glm", family = binomial(link = "logit"),
data = trn_lgr,
trControl =
  trainControl(
    method = "cv", number = 10,
    summaryFunction = twoClassSummary,
    savePredictions = TRUE,
    classProbs = TRUE
  )
)
ROC_plot_mod2.2 <- evalm(ROC_mod2.2)

## ROC curves for every binary classification model
Overall_ROC <- evalm(list(ROC_mod1.2.1, ROC_mod1.2.2,ROC_mod2.1, ROC_mod2.2),
  gnames = c("Mod 1.2.1", "Mod 1.2.2", "Mod 2.1", "Mod 2.2")
)
Overall_ROC$roc

## ROC curves for every submodel in model
Overall_ROC_mod2 <- evalm(list(ROC_mod1.2.1, ROC_mod1.2.2),
                  gnames= c("Model 1.2.1", "Model 1.2.2"))

Overall_ROC_mod3 <- evalm(list(ROC_mod2.1, ROC_mod2.2),
                          gnames= c("Model 2.1", "Model 2.2"))

## Model 3 - Multiple Linear Regression
## Df for regression
regr_df <- select(
  music_final, Popularity_norm, Speechiness_norm,
  Acousticness_norm, Length_norm, Valence_norm, Loudness_norm,
  Danceability_norm, Year_cat, Energy_norm
)

## Generating a train - and test set for Model 3
set.seed(8)
trn_index_regr <- createDataPartition(
  y = regr_df$Popularity_norm,
  p = 0.70, list = FALSE
)
trn_regr <- regr_df[trn_index_regr, ]
tst_regr <- regr_df[-trn_index_regr, ]

## Training model 3.1
set.seed(8)
Popularity_regr3.1 <- train(Popularity_norm ~ .,
  method = "lm", data = trn_regr,
  trControl =
    trainControl(
      method = "cv",
      number = 10,
      returnResamp = "all"
    )
)

# evaluation on test set model 3.1
predicted_outcomes_mod3.1 <- predict(Popularity_regr3.1, tst_regr)
RMSE(predicted_outcomes_mod3.1, tst_regr$Popularity_norm)

## Model 3.2: Regression with interaction effect

## Training model 3.2
set.seed(8)
Popularity_regr_inter3.2 <- train(
  Popularity_norm ~ . +
  Danceability_norm:Year_cat +
  Energy_norm:Year_cat +
  Loudness_norm:Year_cat,
  method = "lm", data = trn_regr,
  trControl = trainControl(method = "cv", number = 10, returnResamp = "all")
  )

## Evaluation on test set model 3.2
predicted_outcomes_mod3.2 <- predict(Popularity_regr_inter3.2, tst_regr)
RMSE(predicted_outcomes_mod3.2, tst_regr$Popularity_norm)

## Compare classification models -----------------------------------------------

## Create df with important metrics
Models <- c(
  "Model 1.1.1", "Model 1.1.2",
  "Model 1.2.1", "Model 1.2.2",
  "Model 2.1", "Model 2.2"
)
Accuracy <- c(0.404, 0.346, 0.588, 0.537, 0.639, 0.633)
Recall <- c(0.702, 0.682, 0.545, 0.473, 0.591, 0.577)
Precision <- c(0.404, 0.366, 0.585, 0.53, 0.642, 0.639)
df_metrics <- data.frame(Models, Accuracy, Recall, Precision) %>%
  melt(id = c("Models"))

# Create overall performance graph comparing all the classification models
ggplot(df_metrics) +
  geom_bar(aes(x = Models, y = value, fill = variable),
    stat = "identity", position = "dodge", width = 0.7
  ) +
  scale_fill_manual("Metrics\n",
    values = c("salmon2", "brown1", "brown4"),
    labels = c(" Accuracy", " Recall", "Precision")
  ) +
  labs(x = "\nModels", y = "Metric value\n", 
       title = "Summary Model Performance") +
  theme_minimal()

## Comparing performance of two multiclass classification models ---------------

M1.1_vs_M1.2 <- create_comparison_plot(df_metrics, "Model 1.1.1", "Model 1.1.2")
M1.1_vs_M1.2

## Comparing performance of the four binary classification models --------------

Models_bi <- c(
  "Model 1.2.1", "Model 1.2.2",
  "Model 2.1", "Model 2.2"
)

Accuracy_bi <- c(0.588, 0.537, 0.639, 0.633)
Recall_bi <- c(0.545, 0.473, 0.591, 0.577)
Precision_bi <- c(0.585, 0.53, 0.642, 0.639)
df_metrics_bi <- data.frame(Models_bi, Accuracy_bi, Recall_bi, Precision_bi) %>%
  melt(id = c("Models_bi"))

## Create overall performance graph comparing all the classification models
ggplot(df_metrics_bi) +
  geom_bar(aes(x = Models_bi, y = value, fill = variable),
           stat = "identity", position = "dodge", width = 0.7
  ) +
  scale_fill_manual("Metrics\n",
                    values = c("salmon2", "brown1", "brown4"),
                    labels = c(" Accuracy", " Recall", "Precision")
  ) +
  labs(x = "\nModels", y = "Metric value\n", 
       title = "Summary Model Performance") +
  theme_minimal()

## Create a table with classification metrics ----------------------------------

## Name the models in the table
models <- c("Model 1.1.1","Model 1.1.2","Model 1.2.1", "Model 1.2.2",
            "Model 2.1", "Model 2.2")

## Name the type of classification in the table
classifiers <- c(" KNN Multiclass full set", " KNN Multiclass reduced set",
                 " KNN Binary full set", " KNN Multiclass reduced set",
                 " Log. reg. full set", " Log. reg. reduced set")

## Merge these two together
x <- data.frame("Model No." = models, "Specifications" = classifiers)

## Merge the metrics of the six models together
l <- c(get_metrics_multi(knn_confMmod1.1), 
          get_metrics_multi(knn_confMmod1.2),
          get_metrics_bi(knn_confMmod2.1),
          get_metrics_bi(knn_confMmod2.2),
          get_metrics_bi(lgr_confMmod2.1),
          get_metrics_bi(confMmod2.2))

d <- t(data.frame(l))

colnames(d) <- c("Sensitivity", "Specificity", "Precision", "Recall", "F1", 
                 "Accuracy","Kappa")

rownames(d) <- NULL

## Create the final table
metric_table <- cbind(x,d)

metric_table

## Comparing performance linear regreession models -----------------------------

## Create dataframe
linear_models <- c("Model 3.1", "Model 3.2")
RMSE <- c(0.885, 0.869)

df_linear <- data.frame(linear_models, RMSE) %>%
  melt(id=c("linear_models"))

ggplot(df_linear) +
  geom_bar(aes(x = linear_models, y = value),
           stat = "identity", position = "dodge", width = 0.7) +
  labs(x = "\nModels", y = "RMSE", title = "Summary Model Performance") +
  theme_minimal()

## END -------------------------------------------------------------------------
