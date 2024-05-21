# Effects of Environemtal Factors of CAHOOTS Calls

## Author: Judie Gatson
### Date: 05/21/24

## Description
Welcome! In this project, I have analyzed data from CAHOOTS against environmental data to find patterns on how calls are affected. 

## Program and packages

This project runs in Python and uses the following packages:
- Numpy
- Pandas
- pandas.api.types
- matplotlib.pyplot
- matplotlib.dates
- seaborn
- datetime

In addition to the CAHOOTS data, there are three main sources of data:
- Weather Underground: https://www.wunderground.com/history/monthly/us/or/eugene
- EPA: https://www.epa.gov/outdoor-air-quality-data/air-quality-index-daily-values-report
- Time and Date: https://www.timeanddate.com/sun/usa/eugene

There are some key functions in the project that I created and used to clean and analyze the data.
- season(df): This function takes in a dataframe with a datetime column and returns the dataframe with a new column called seasons. This function will add a season (spring, summer, fall, winter) that corresponds to the date. Although the equinoxes and solstices vary by a few days each year, in this function they are given a hard coded date for ease of processing.
- group season(df, season): This function takes in a df that contains a season column, and returns a new df that is grouped by the season passed through the parameter. It will also add a new column that gives the % of calls received dring the given row observation.
- week_col(df): This function takes in a df with a datetime column and returns an array that gives the week of each date time object.
- aqi_index(df: This function takes in a DF with an "overal aqi' column and returns an array of corresponding aqi indexes.
- 
