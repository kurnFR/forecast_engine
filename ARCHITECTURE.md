# Forecast Engine Architecture
2
 
3
## Overview
4
 
5
The Forecast Engine produces regional monthly Sell-In forecasts using:
6
 
7
- Historical Baseline
8
- ETS
9
- SARIMA
10
- XGBoost
11
- Weighted Ensemble
12
 
13
## Pipeline
14
 
15
Source Data
16
↓
17
Data Validation
18
↓
19
Feature Engineering
20
↓
21
Backtesting
22
↓
23
Model Scoring
24
↓
25
Model Selection
26
↓
27
Forecast Generation
28
↓
29
Prediction Intervals
30
↓
31
Database Output
32
 
33
## Major Components
34
 
35
### data/
36
Data extraction and preprocessing.
37
 
38
### features/
39
Feature generation for classical and ML models.
40
 
41
### backtest/
42
Leakage-safe rolling validation.
43
 
44
### models/
45
Forecast model implementations.
46
 
47
### forecast/
48
Forecast orchestration and selection.
49
 
50
### output/
51
Database persistence layer.
52
 
53
## Design Principles
54
 
55
- No target-month leakage
56
- Reproducible backtests
57
- Deterministic outputs
58
- Production-safe validation
