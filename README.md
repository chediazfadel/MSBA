# **<ins>University of Utah MSBA Summer 2023 Capstone</ins>** 
[![](https://img.shields.io/badge/R-RMarkdown_Notebooks-276DC3?logo=R)](https://github.com/chediazfadel/msba_capstone/tree/main/RMarkdown) [![](https://img.shields.io/badge/R-HTML_Notebooks-276DC3?logo=R)](https://github.com/chediazfadel/msba_capstone/tree/main/HTML)

## Business Problem and Project Objective
[Maverik](https://www.maverik.com/) is interested in producing more accurate financial plans and initial ROI documents for future convenience store locations. Considerable uncertainty is associated with deploying new locations in their network and being able to properly allocate resources and accurately predict profitability is crucial to the prosperity of their business. To this end, this project aims to augment veracity of financial plans and ROI documents by leveraging the store-level time series and qualitative data collected by Maverik. This will be done by employing an ensemble of forecasting and supervised regression models designed to provide daily store level sales forecasts of multiple key product categories. Success of this project will be benchmarked against Maverik’s existing Naive forecasting solution and the proposed model will be tuned to minimize:
- Akaike Information Criterion (AIC)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

Key abilities of the final must include:
- Calculation of daily level, seasonally sensitive forecast for each of the sales metrics defined by Maverik.
- Functionality to create updated forecasts given new data.

The ability to accurately forecast store sales will enable Maverik to optimize the expenditure of scarce resources by pursuing the most profitable locations and minimizing misallocation of said resources when opening a new store. This will lead to better investment, decreased waste, and more reliable financial evaluations.

## Group Solution 
Five models will be developed to leverage Maverik's historical sales data: Vector AutoRegressive Model, Prophet, Support Vector Regression, Extreme Gradient Boosting, and an ARIMA/ETS Ensemble. Each model will be tuned to minimize RMSE, which will be the metric used to select the final model.

While SVR yielded the best RMSE, we were unsuccessful in developing the model's ability to update itself as new data is observed. Ultimately, the ARIMA/ETS Ensemble was chosen as our final model to recommend to Maverik.


#### Ensemble Model Performance
|   | RMSE | MAPE |
| ----------- | ----------- | ----------- |
| Inside Sales 2-week prediction | 210,026 | 11.9% |
| Inside Sales 3-week prediction | 209,926 | 9.0% |
| Inside Sales 6-month prediction | 50,273 | 3.1% |
| Food Service 2-week prediction | 25,378 | 7.72% |
| Food Service 3-week prediction | 23,941 | 7.4% |
| Food Service 6-month prediction | 11,350 | 4.3% |
| Diesel 2-week prediction | 335,975 | 20.6% |
| Diesel 3-week prediction | 222,596 | 18.9% |
| Diesel 6-month prediction | 26,183 | 3.8% |
| Unleaded 2-week prediction | 201,561 | 12.6% |
| Unleaded 3-week prediction | 201,656 | 12.8% |
| Unleaded 6-month prediction | 27,316 | 2.3.75% |

## Personal Contribution
### EDA [![](https://img.shields.io/badge/R-EDA-276DC3?logo=R)](https://github.com/chediazfadel/msba_capstone/blob/main/EDA-chediazfadel.md)

