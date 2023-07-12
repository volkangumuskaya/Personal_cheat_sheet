# Initialize
```sql
use database fxs_eu_ntwkdaor_efsx_001 --To avoid repeating this as a prefix before table names
set spark.sql.autoBroadcastJoinThreshold=-1
set spark.databricks.optimizer.adaptive.enabled=True
```

```python
spark.conf.set('spark.sql.shuffle.partitions','auto')
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

