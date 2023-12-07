# **<ins>University of Utah MSBA Capstone</ins>** 
[![](https://img.shields.io/badge/R-RMarkdown_Notebooks-276DC3?logo=R)](https://github.com/chediazfadel/MSBA/tree/main/RMarkdown) [![](https://img.shields.io/badge/R-HTML_Notebooks-276DC3?logo=R)](https://github.com/chediazfadel/MSBA/tree/main/HTML)

## Business Problem and Project Objective
[Maverik](https://www.maverik.com/) is interested in producing more accurate financial plans and initial ROI documents for future convenience store locations. Considerable uncertainty is associated with deploying new locations in their network and being able to properly allocate resources and accurately predict profitability is crucial to the prosperity of their business. To this end, this project aims to augment veracity of financial plans and ROI documents by leveraging the store-level time series and qualitative data collected by Maverik. This will be done by employing an ensemble of forecasting and supervised regression models designed to provide daily store level sales forecasts of multiple key product categories. Success of this project will be benchmarked against Maverik’s existing Naive forecasting solution and the proposed model will be tuned to minimize:
- Akaike Information Criterion (AIC)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

Key abilities of the final model must include:
- Calculation of daily level, seasonally sensitive forecasts for each of the sales metrics defined by Maverik.
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
| Unleaded 6-month prediction | 27,316 | 3.75% |

## Personal Contribution
### EDA [![](https://img.shields.io/badge/R-EDA-276DC3?logo=R)](https://github.com/chediazfadel/MSBA/blob/main/RMarkdown/EDA%20-%20Che%20Diaz%20Fadel.Rmd)

As well as gaining a greater holistic understanding of the provided data, my EDA aims to answer the following questions:

- What preprocessing/cleaning is required?
  - what is the scale of missing values? How should they be handled?
  - Do erroneously encoded categorical variables need to be corrected?
  - How do date attributes need to be standardized or formatted to ensure accurate seasonality calculations?
  - Is the existing data “tidy”?
    - Each variable has its own column
    - Each observation has its own row
    - Each value has its own cell
- Which features are most correlated with each of the target sales metrics?
- What explanatory variables are collinear and how should that be handled?
- What affect does seasonality have on the target variables?

While my EDA notebook answers all of these questions, I'll mention here two key contributions:

I found that the data does not have any explicitly `NA` values, but does utilize character strings to indicate such. Maverik has indicated that “N/A”, “None”, or any analogous value indicates the absence of that attribute at the site rather than missing data. In this case, it would be best to apply a single value for these cases instead of using explicit `NA` and removing.

The entire data set spans from 2021-01-12 to 2023-08-16 and all 38 stores are present for one year and one day. Given that `open_date` is not uniformly distributed, network-wide seasonality will have to be calculated either on a sales per store basis or by standardizing the date by attributing a day ID similar to `week_id`. Maverik expressed the importance of aligning days in a standardized manner, which is why `week_id` is included. I constructed the standardized `day_id` to ensure the same day was represented similarly across years during modeling.

### Modeling [![](https://img.shields.io/badge/R-Modeling-276DC3?logo=R)](https://github.com/chediazfadel/MSBA/blob/main/RMarkdown/Modeling_chediazfadel.Rmd)

I elected to train two types of models: **Extreme Gradient Boosting (XGB)** and an **ARIMA/ETS Ensemble**. XGBoost is a very powerful and versatile model that can handle high-dimensionality but often requires a lot of data. While the preliminary results were promising, in the end I was not able to implement the required functionality given the allotted time.

The ensemble method was certainly more successful:
- Able to calculate daily level, seasonally sensitive forecasts
- Will update forecasts given new data
- Beat Maverik's performance benchmarks

This approach was extremely computationally expensive, but on a business scale not prohibitively so. This modeling process also gave me the opportunity to take a deeper dive into time-series forecasting than I ever had before, and I very much enjoyed the experience.

## Business Value
While the final model we presented on can certainly be improved upon, there is still some value to be gleaned from it.

While all businesses are interested in growth, as recently demonstrated with their acquisition of Kum & Go, Maverik is particularly interested in site expansion. When engaging in an expansionary strategy, more accurate forecasts provide:
- reduced risk of over-leveraging
- minimized opportunity cost (not expanding enough)
- improved inventory and labor management
- improved site/real-estate selection
- better evaluation and discovery of preferred acquisition targets

Depending on the sales metric (and prediction time), our model achieved a reduction of RMSE between 10%-60%.

## Difficulties

While the scale of the data itself was not particularly immense, fitting and forecasting 365 times for each site, for each model adds up very quickly. The set of models available to address this problem is vast. Researching and experimenting with the myriad models was extremely time consuming and left less time than was desired for actually modeling for the given business case. My previous experience with time-series forecasting was wholly insufficient for this project and therefore extensive self-study and research was required to produce something minimally acceptable.

## What Was Learned

I found this to be a very enriching experience. In preparation for this project, I took a deep dive into the book **Forecasting: Principles and Practice** by Rob J Hyndman and George Athanasopoulos. Rob Hyndman is also the author of the `fpp3` R package so I in turn learned a lot about implementing forecasting concepts using the functions and infrastructure he devised. It became clear very early that time-series forecasting is a very dense and nuanced topic with an incredible variety of ways to reach a solution with your data. Plenty is still left to learn but I look forward to employing what I've gained during this project to further excel my career.
