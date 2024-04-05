# Get ammount of sucessfull user activity of type sign off during a period of time
```SQL
SELECT count(*)
FROM [aiml].[dbo].[gcp_activity]
WHERE 
	OPERTN_NM = '%insert OPERTN_NM or operation type here'
    --AND EXECTD_DT BETWEEN '%insert start EXECTD_DT date here' AND '%insert end EXECTD_DT date here' --if nedded, filter per time period
``` 

# Get all OPERTN_NM types
```SQL
SELECT distinct(OPERTN_NM)
FROM [aiml].[dbo].[gcp_activity]
``` 

# Get all gcp_activity columns
```SQL
SELECT COLUMN_NAME
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = N'gcp_activity';
``` 

