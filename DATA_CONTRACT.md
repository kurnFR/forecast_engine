# Data Contract
2
 
3
## Source Tables
4
 
5
### mv_ai_region_monthly
6
 
7
Purpose:
8
- Monthly regional targets
9
 
10
Required Columns:
11
| Column | Type |
12
|----------|----------|
13
| regioncode | text |
14
| periode | date |
15
| target_value | numeric |
16
 
17
### dimdate.networkeddays
18
 
19
Purpose:
20
- Working-day calendar
21
 
22
Required Columns:
23
| Column | Type |
24
|----------|----------|
25
| date | date |
26
| networkeddays | integer |
27
 
28
## Forecast Output
29
 
30
Primary Key:
31
(regioncode, periode)
32
 
33
Required Fields:
34
- p10
35
- p50
36
- p90
37
- selected_model
38
- model_wape
39
- forecast_timestamp
