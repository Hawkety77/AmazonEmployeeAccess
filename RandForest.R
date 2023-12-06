## scp -r ~/Desktop/School\ Projects/AmazonEmployeeAccess teh77@becker.byu.edu:~/
## R CMD BATCH --no-save --no-restore Penalized.R &

library(tidymodels)
library(tidyverse)
library(embed)
library(vroom)
library(doParallel)
library(themis)

detectCores() #How many cores do I have?
cl <- makePSOCKcluster(7)
registerDoParallel(cl)

df_train <- vroom('train.csv') %>%
  mutate(ACTION = as.factor(ACTION))
df_test <- vroom('test.csv')

my_recipe <- recipe(ACTION ~ ., data=df_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding
  # step_smote(all_outcomes(), neighbors = 5)

prep <- prep(my_recipe)
baked <- bake(prep, new_data = NULL)

my_model <- rand_forest(trees = 1000, 
                        min_n = 24, 
                        mtry = 1) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rand_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model) %>%
  fit(data = df_train)

# tuning_grid <- grid_regular(min_n(),
#                            levels = 4)
# 
# folds <- vfold_cv(df_train, v = 5, repeats = 1)
# 
# tree_tune <- tune_grid(my_model, 
#                       my_recipe, 
#                       folds, 
#                       control = control_grid(save_workflow = TRUE), 
#                       grid = tuning_grid)
# 
# collect_metrics(tree_tune)
# show_best(tree_tune, metric = 'roc_auc')
# 
# best_tune <- tree_tune %>%
#   select_best("roc_auc")
# 
# final_workflow <- rand_workflow %>%
#   finalize_workflow(best_tune) %>%
#   fit(data = df_train)

amazon_predictions <- predict(rand_workflow, 
                              new_data = df_test, 
                              type = 'prob')

submission <- amazon_predictions %>%
  bind_cols(df_test) %>%
  select(id, .pred_1) %>%
  rename(Action = .pred_1)

write_csv(submission, 'submission_randforest.csv')

#Stop Parallel
stopCluster(cl)


