# Get ammount of sucessfull user activity of type sign off during a period of time
```SQL
SELECT count(*)
FROM [aiml].[dbo].[gcp_activity]
WHERE 
	OPERTN_NM = '%insert operation type here'
    AND EXECTD_DT BETWEEN '%insert start time here. example: 2023-05-09' AND '%insert end date here'
	AND STAT_CD = 200
``` 