library(tidymodels)
library(tidyverse)
library(embed)
library(vroom)
library(themis)

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

amazon_predictions <- predict(logReg_workflow, 
                              new_data = df_test, 
                              type = 'prob')

submission <- amazon_predictions %>%
  bind_cols(df_test) %>%
  select(id, .pred_1) %>%
  rename(Action = .pred_1)

write_csv(submission, 'submission_logistic.csv')
