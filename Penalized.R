## scp -r ~/Desktop/School\ Projects/AmazonEmployeeAccess teh77@becker.byu.edu:~/
## R CMD BATCH --no-save --no-restore Penalized.R &

library(tidymodels)
library(tidyverse)
library(embed)
library(vroom)
library(doParallel)

detectCores() #How many cores do I have?
cl <- makePSOCKcluster(7)
registerDoParallel(cl)


df_train <- vroom('train.csv') %>%
  mutate(ACTION = as.factor(ACTION))
df_test <- vroom('test.csv')

my_recipe <- recipe(ACTION ~ ., data=df_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_smote(all_outcomes(), neighbors = 5)

prep <- prep(my_recipe)
baked <- bake(prep, new_data = NULL)

logRegModel <- logistic_reg(mixture = tune(),
                            penalty = tune()) %>%
  set_engine("glmnet")

logReg_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logRegModel)

tuning_grid <- grid_regular(penalty(), 
                            mixture(), 
                            levels = 3)

folds <- vfold_cv(df_train, v = 5, repeats = 1)

cv_results <- logReg_workflow %>%
              tune_grid(resamples = folds, 
                        grid = tuning_grid, 
                        metrics = metric_set(roc_auc))

met <- collect_metrics(cv_results)

best_tune <- cv_results %>%
  select_best("roc_auc")

final_workflow <- logReg_workflow %>%
  finalize_workflow(best_tune) %>%
  fit(data = df_train)

amazon_predictions <- predict(final_workflow, 
                              new_data = df_test, 
                              type = 'prob')

submission <- amazon_predictions %>%
  bind_cols(df_test) %>%
  select(id, .pred_1) %>%
  rename(Action = .pred_1)

write_csv(submission, 'submission_penalized.csv')

#Stop Parallel
stopCluster(cl)

