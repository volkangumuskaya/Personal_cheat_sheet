### In this example we run a notebook in `databricks` from another notebook, and log with parent-child nested structure using `MLflow`

# Notebook1: Main run

```python
mlflow.set_experiment(experiment_path)
mlflow.start_run(run_name='parent_run_from_main',nested=True)

now = datetime.datetime.now()

mlflow.log_param('now',now)
mlflow.log_param('parent',1)
mlflow.log_param('child',0)
run = mlflow.active_run()
parent_id=run.info.run_id
print("Active run_id: {}".format(run.info.run_id))
mlflow.end_run()
```

## Call sub runs in  a loop by running a notebook

```python

notebook_name='notebook_name'

for i in [0,2]:
  for j in [0,2]:
    dbutils.notebook.run(notebook_name, 60, 
                           {"widget1": i,
                            "widget2": j,
                            "parent_mlflow_run":parent_id
                            })
```

# Notebook 2: Sub run

```python
arg1='arg1'
```
## Function to log things
```python
def log_everything(arg1,parent_id):  
  try:
    mlflow.end_run()
  except:
    print('..')
  
  mlflow.set_experiment(experiment_path)
  mlflow.start_run(run_id=parent_id,nested=True)
  mlflow.start_run(run_name='run_name',nested=True)
  mlflow.log_param('now',now)
  mlflow.log_param('parent',0)
  mlflow.log_param('child',1)

mlflow.log_artifact("/dbfs/FileStore/..."+pngname+".png")

  path="/dbfs/FileStore/.../name.html"
  df.to_html(path)
  mlflow.log_artifact(path)
  path="dbfs:/FileStore/.../name.html"
  dbutils.fs.rm(path)
  mlflow.end_run()
    
  run = mlflow.active_run()
  print("Active run_id: {}".format(run.info.run_id))
  mlflow.end_run()
  return(run.info.run_id)
```

## Run the function
```python
parent_mlflow_run=dbutils.widgets.get("parent_mlflow_run")
print("parent_mlflow_run:",parent_mlflow_run)

log_everything(demo_day,parent_mlflow_run)
```
