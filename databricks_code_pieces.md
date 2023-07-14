# Initialize
```sql
use database fxs_eu_ntwkdaor_efsx_001 --To avoid repeating this as a prefix before table names
set spark.sql.autoBroadcastJoinThreshold=-1
set spark.databricks.optimizer.adaptive.enabled=True
```

```python
spark.conf.set('spark.sql.shuffle.partitions','auto')
```
# browsing the file store
```python
dbutils.fs.ls('dbfs:/FileStore/')
```
# removing file
```python
dbutils.fs.rm('dbfs:/FileStore/jars/yourfile.egg')
```
# move file
```python
# dest_path = 'abfss://...'
# orig_path='/FileStore/fig.html'
# dbutils.fs.mv(orig_path,dest_path,False)
```

# some basic sql
```sql
select format_number(count(*),0) from table_name
```

```sql
select
(select count(*) from table1 ) as c1,
(select count(distinct user_name) from table1 ) as c2
```
## count per group and percentage
```sql
select type,model,
format_number(count(*) ,0) as count_,
round(count(*)/(select count (*) from table)*100,3) as count_percent

from table 
group by type,model
order by count_percent desc
```

## more on count and percentage using cte (common table expression)
```sql
with cte as (
  select type,model,
  count(*) _count
  from table
  group by type,model,
)
select *,
SUM(_count) OVER(PARTITION BY type) AS sub_total,
_count/(SUM(_count) OVER(PARTITION BY type)) +0.00 as _percentage
from cte
order by type,model
```

## haversine distance
```sql
%sql
CREATE FUNCTION if NOT EXISTS haversine(Lat1 DOUBLE, Lng1 DOUBLE, Lat2 DOUBLE, Lng2 DOUBLE) RETURNS DOUBLE RETURN 
    2 * 6335 
        * sqrt(
            pow(sin((radians(Lat2) - radians(Lat1)) / 2), 2)
            + cos(radians(Lat1))
            * cos(radians(Lat2))
            * pow(sin((radians(Lng2) - radians(Lng1)) / 2), 2)
        );
```
# widgets
```python
dbutils.widgets.text("state", "CA")
dbutils.widgets.dropdown("state_dd", "CA", ["CA", "IL", "MI", "NY", "OR", "VA"])
dbutils.widgets.remove("state")
state_value=str(dbutils.widgets.get("state"))
```

```sql
select * from table
where type is null
order by rand() limit 100
```

# sql query via python
```python
%python
df=spark.sql("select * from table")
```

## write from delta table
```python
sdf=spark.sql("select * table")

dbutils.fs.rm(file_path_test[5:],recurse=True)
sdf.repartition(1).write.format("parquet").mode("overwrite").save(file_path_test[5:])
```

# create paequet file
```sql
CREATE TABLE IF NOT EXISTS db.table
USING PARQUET
LOCATION path
AS
SELECT * FROM db.source_table
```

# read/write 
```python
df.coalesce(1).write.parquet(".....parquet")
df = spark.read.format("parquet").load(".....parquet")
df.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save(".....csv")


usersDF = (spark.read
  #.option("sep", "\c")
  .option("header", True)
  .option("inferSchema", True)
  .csv(path))
```
