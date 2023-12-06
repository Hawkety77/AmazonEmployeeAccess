library(tidymodels)
library(tidyverse)
library(embed)
library(vroom)
library(doParallel)
library(themis)
library(tidymodels)
library(rsample)
library(stacks)

detectCores() #How many cores do I have?
cl <- makePSOCKcluster(7)
registerDoParallel(cl)

untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples()

folds <- vfold_cv(df_train, v = 5, repeats = 1)

df_train <- vroom('train.csv') %>%
  mutate(ACTION = as.factor(ACTION))
df_test <- vroom('test.csv')

my_recipe <- recipe(ACTION ~ ., data=df_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding
# step_smote(all_outcomes(), neighbors = 5)

prep <- prep(my_recipe)
baked <- bake(prep, new_data = NULL)

my_model <- rand_forest(trees = 500, 
                        min_n = tune()) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rand_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model)

tuning_grid <- grid_regular(min_n(),
                            levels = 4)

randfor_folds_fit <- bike_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq),
            control = untunedModel)

tree_tune <- tune_grid(my_model, 
                       my_recipe, 
                       folds, 
                       control = control_grid(save_workflow = TRUE), 
                       grid = tuning_grid)

collect_metrics(tree_tune)
show_best(tree_tune, metric = 'roc_auc')

best_tune <- tree_tune %>%
  select_best("roc_auc")

final_rand_workflow <- rand_workflow %>%
  finalize_workflow(best_tune) %>%
  fit(data = df_train)

### Logistic Regression ###

df_train <- vroom('train.csv') %>%
  mutate(ACTION = as.factor(ACTION))
df_test <- vroom('test.csv')

my_recipe <- recipe(ACTION ~ ., data=df_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

prep <- prep(my_recipe)
baked <- bake(prep, new_data = NULL)

logRegModel <- logistic_reg() %>%
  set_engine("glm")

logReg_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logRegModel) %>%
  fit(data = df_train)

### Predict ###

bike_stack <- stacks() %>%
  add_candidates(logReg_workflow) %>%
  add_candidates(final_rand_workflow)

fitted_bike_stack <- bike_stack %>%
  blend_predictions() %>%
  fit_members()

collect_parameters(fitted_bike_stack, "tree_folds_fit")

# Make Predictions
amazon_predictions <- predict(fitted_bike_stack, new_data = df_test, type = 'prob')

submission <- amazon_predictions %>%
  bind_cols(df_test) %>%
  select(id, .pred_1) %>%
  rename(Action = .pred_1)

write_csv(submission, 'submission_stacked.csv')

#Stop Parallel
stopCluster(cl)

