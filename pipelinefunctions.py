import pandas as pd
import datetime as dt
import numpy as np
import os

import pathlib

import pyodbc

from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

import statsmodels.api as sm
import statsmodels.tsa as tsa

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

#This lets us save our linear regression models.
import pickle

import math

def alpha_generation(edf = None, stocks = None, companyname=None, ticker= None, th_ab = 2, th_bl = -2, index='HC_Index', delta=1, project_path=None):
    '''alhpa_genration makes calls on whether to long or short a stock based on consensus and GD revenue estimates for the company of interest.

    Arguments:
    edf = estimate samples dataframe.
    stocks = the daily stock price dataframe
    company = the name of hte company according to the master file
    ticker = the BBGticker name of hte company according to the master file.
    th_ab = the threshold above beforoe we do a call.
    th_bl the threshold below before we do a call.
    index = the index we are comparing to to determine alpha.

    This creates several files but returns nothing.
    It creates a call graph and whether or not each call made money.

    It also creates a graph showing the cumulative alpha over time.
    '''

    if companyname is None:
        raise ValueError("companyname is Nonetype in alpha_generation")

    if ticker is None:
        raise ValueError("ticker is NoneType in alpha_generation")

    #Note to self:
    # -> data1 should be replaced with edf
    # -> data2 should be replaced with stocks
    # name1 should be replaced with companyname
    # name2 should be replaced with ticker
    edf = edf[['Date', companyname+" Latest Consensus", companyname+" GD est", companyname+" Reporting Date"]].dropna()

    stocks = stocks[['Date', ticker, 'S&P 500', f'{index}']]
    stocks.rename(columns = {'Date':'Date2', ticker:companyname+' Share price'}, inplace = True)

    try:
        edf[companyname+' Est. Vs Consensus'] = (edf[companyname+' GD est'] - edf[companyname+' Latest Consensus'])*100/edf[companyname+' Latest Consensus']
    except TypeError:
        edf.to_pickle(project_path + f"/StrAndFloat.pkl")
        raise TypeError(f"Possible str and float error with GD est and Latest Consensus Numbers. Error file saved to StrAndFloat.pkl")

    def traderule(x):
        '''Determines if we go long or short'''
        if x>th_ab:
            z= "Long"
        elif x<th_bl:
            z = "Short"
        else:
            z = "Hold"
        return z

    edf['Trading Rule'] = edf[companyname+' Est. Vs Consensus'].apply(traderule, 1)

    # #get the dates by using index concept

    # #I really dislike how this is built as the error handling method of this loop will hide genuine errors like the date
    # #that you want not appearing in the stock prices sheet. I've had this issue freeze up my computer for a day or two before.
    # #I'm going to try to make my own.
    # def datecollector(x, delta):
    #     if isinstance(x, str):
    #         month, day , year = x.split("/")
    #         day = int(day)
    #         month = int(month)
    #         year = int(year)
    #         x = dt.datetime(year, month, day)
    #     Z= True
    #     i = 1
    #     while Z== True:
    #         try:
    #             k = stocks[stocks['Date2'] == x].index[0]
    #             Z = False
    #         except:
    #             try:
    #                 k = stocks[stocks['Date2'] == x+pd.DateOffset(-i*(delta/abs(delta)))].index[0]
    #                 Z = False
    #             except:
    #                 i = i+1

    #     return (stocks['Date2'][k+delta])

    # #edf['Date of Trade Entry'] = edf['Intersect Reporting date'] + pd.DateOffset(-entry_date)
    # edf['Date of Trade Entry'] = edf.apply(lambda x: datecollector(x[companyname+' Reporting Date'], entry_date), axis = 1)
    # edf['Date of Trade Exit'] = edf.apply(lambda x: datecollector(x[companyname+' Reporting Date'], exit_date), axis =1)

    def newdatecollector(x, delta):
        '''My version. basically it finds the date that you'd be buying stocks and the date you'd be selling stocks by
        referring to both the reporting date and the stock prices sheet.
        delta is the 'direction' you're looking in. A negative number means you're looking backwards for the entry date.
        A positive number means you're looking for the exit date
        '''
        
        if isinstance(x, str):
            month, day , year = x.split("/")
            day = int(day)
            month = int(month)
            year = int(year)
            x = dt.datetime(year, month, day)
        
        still_looking=True
        #Starts as 1 or -1 depending whether you're going up or down.
        i = 1*delta
        
        while still_looking ==True:
            if stocks[stocks['Date2']==x+dt.timedelta(days=i)].empty:
                if x < stocks['Date2'].max() and x > stocks['Date2'].min():
                    #Check that the date we're looking for is within the date range that my stock price data captures.
                    #Continue to decrement/increment in the direction we're intersted in.
                    i = i+delta
                else:
                    raise ValueError(f"The date we're looking for {x} is not within the date range {stocks['Date2'].min()} to {stocks['Date2'].max()}")
                    
            else:
                #Now we've found a date that corresponds to this stuff.
                return x + dt.timedelta(days=i)

    edf.set_index("Date", inplace=True)

    edf['Date of Trade Entry'] = edf[companyname+' Reporting Date'].apply(newdatecollector, delta=-delta)
    edf['Date of Trade Exit'] = edf[companyname+' Reporting Date'].apply(newdatecollector, delta=delta)

    edf.reset_index(inplace=True)

    #get the price of entry and exit
    data = edf.merge(stocks, how = 'left', left_on = 'Date of Trade Entry', right_on = 'Date2').drop(['Date2'], 1)
    data = data.merge(stocks, how = 'left', left_on = 'Date of Trade Exit', right_on = 'Date2', suffixes=('_entry', '_exit')).drop(['Date2'], 1)

    def returnon(x):
        if x[0] == 'Long':
            k = ((x[2]/x[1]) - 1)*100
        elif x[0] == "Short":
            k = ((x[1]/x[2]) - 1)*100
        else:
            k = 0
        return k

    data[companyname+' Returns'] = data[['Trading Rule', companyname+' Share price_entry', companyname+' Share price_exit']].apply(returnon, 1)

    ### doubts on long and short for index
    #data['S&P 500 Returns'] = data[['Trading Rule', 'S&P 500_entry', 'S&P 500_exit']].apply(returnon, 1)
    #data['Healtcare Index Returns'] = data[['Trading Rule', 'HC_Index_entry', 'HC_Index_exit']].apply(returnon, 1)

    # use this, as this includes Long Only for benchmark performance
    data['S&P 500 Returns'] = (data['S&P 500_exit'] - data['S&P 500_entry'])*100/data['S&P 500_entry']
    data['Healtcare Index Returns'] = (data[f'{index}_exit'] - data[f'{index}_entry'])*100/data[f'{index}_entry']

    data[companyname+' Alpha (w.r.t HC Index) in % points'] = data[companyname+' Returns'] - data['Healtcare Index Returns']
    data[companyname+' Alpha (w.r.t S&P 500) in % points'] = data[companyname+' Returns'] - data['S&P 500 Returns']

    #alpha correction
    def alpha_correction_for_hold(x):
        if x[0] == 'Hold':
            k = 0
        else:
            k = x[1]
        return k

    data[companyname+' Alpha (w.r.t HC Index) in % points'] = data[['Trading Rule', companyname+' Alpha (w.r.t HC Index) in % points']].apply(alpha_correction_for_hold, 1)
    data[companyname+' Alpha (w.r.t S&P 500) in % points'] = data[['Trading Rule', companyname+' Alpha (w.r.t S&P 500) in % points']].apply(alpha_correction_for_hold, 1)

    
    datapath = pathlib.Path(project_path + '/Alpha Generation/alpha graphs/')
    datapath.mkdir(parents=True, exist_ok=True)
    data.to_csv(project_path + f'/Alpha Generation/alpha graphs/{companyname} results_backtest.csv')
    print(f"Successfully made file for {companyname}")

    l = data.shape[0]
    datax2  = data[data['Trading Rule'].isin(['Long', 'Short'])]
    df1 = datax2.groupby('Trading Rule')[companyname+' Alpha (w.r.t HC Index) in % points'].agg('count').reset_index(name = 'Signal counts %')
    df2 = datax2[datax2[companyname+' Alpha (w.r.t HC Index) in % points']>=0].groupby('Trading Rule')[companyname+' Alpha (w.r.t HC Index) in % points'].agg('count').reset_index(name = 'Success %')
    df3 = datax2[datax2[companyname+' Alpha (w.r.t HC Index) in % points']>=0].groupby('Trading Rule')[companyname+' Alpha (w.r.t HC Index) in % points'].agg('mean').reset_index(name = 'Avg Success returns (%)')
    df4 = datax2[datax2[companyname+' Alpha (w.r.t HC Index) in % points']<0].groupby('Trading Rule')[companyname+' Alpha (w.r.t HC Index) in % points'].agg('mean').reset_index(name = 'Avg Failure returns (%)')
    df = df1.merge(df2, how ='left', on = "Trading Rule").merge(df3, how ='left', on = "Trading Rule").merge(df4, how ='left', on = "Trading Rule")

    #df['Success %'] = df['Success %']*100/l
    df['Success %'] = df['Success %']*100/df['Signal counts %']
    df['Signal counts %'] = df['Signal counts %']*100/l

    data['Cumulative Alpha'] = 0
    data['base'] = 100

    #data.dropna(axis=0, subset=['Healtcare Index Returns'], inplace=True)

    for i in range(data.index.min(),data.index.max()+1):
        if i == data.index.min():
            data.loc[i, 'Cumulative Alpha'] = 100*(1+data[companyname+' Alpha (w.r.t HC Index) in % points'][i]/100)
        else:
            data['Cumulative Alpha'][i] = data['Cumulative Alpha'][i-1]*(1+data[companyname+' Alpha (w.r.t HC Index) in % points'][i]/100)



    fig = px.bar(data, y=companyname+' Alpha (w.r.t HC Index) in % points', x='Date', text='Trading Rule')
    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.show()

    if not os.path.exists(project_path + '/Alpha Generation/alpha graphs'):
        os.makedirs(project_path + '/Alpha Generation/alpha graphs')

    fig.write_image(project_path + f"/Alpha Generation/alpha graphs/{companyname} call profitability.jpeg")

    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data['Date'], y=data['Cumulative Alpha'], line_shape='spline', name="Cumulative Alpha"))
    fig2.add_trace(go.Scatter(x=data['Date'], y=data['base'], line_shape='spline', name="Base"))

    alpha = (data['Cumulative Alpha'][len(data)-1]/100 - 1)*100
    fig2.update_layout(title = "Cumulative Alpha: " + str(alpha), showlegend=False)
    fig2.show()

    fig2.write_image(project_path + f"/Alpha Generation/alpha graphs/{companyname} cumulative alpha 2018+.jpeg")

    return f"{companyname} done!"

def apply_adarsh_keep(x, month_lag=1, day = 15):
    '''Adarsh keeps the values that are month_lag behind quarter close. This apply function just flags which should be kept'''
    year, quarter = x['Quarter'].split(" ")
    #As quarter will be a string "Q4" or something right now
    quarter = int(quarter[-1])
    #Adarsh is interested in quarter ends so month is always the last month of the quarter
    month = quarter*3
    #As we're predicting what happened at 2 weeks past quarter close.
    year = int(year)

    if month+month_lag > 12:
        month = month + month_lag - 12
        year+=1
    else:
        month = month + month_lag

    q_date = dt.datetime(year, month, day)

    if x['Load Date'] < q_date:
        return True
    else:
        return False

def apply_new_keep(x):
    '''The quarter column no longer holds the information it used to. But I can still reconstruct it.
    '''

    #We need to convert the transaction date into an old school quarter date that apply_adarsh_keep would have used.

    year = x['Transaction Date'].year
    month = x['Transaction Date'].month

    if month in [1,2,3]:
        quarter_month=4
    elif month in [4,5,6]:
        quarter_month=7
    elif month in [7,8,9]:
        quarter_month=10
    elif month in [10,11,12]:
        #Do this so that we can backpedal by one day to get the last day of the quarter.
        quarter_month=1
        year = year+1
    else:
        raise ValueError(f"We have a fucked up month in the Transaction Date of: {x}")

    quarter_date = dt.datetime(year, quarter_month, 1) - dt.timedelta(days=1)

    if x['Load Date'] < quarter_date:
        return True
    else:
        return False
    
def apply_quarter_layout(x):
    ###OBSOLETE###
    #apply_quarter_layout assumes that all companies report on the exact same quarters every time. Which is not hte case.
    #This function has been replaced by find_quarteR_number.
    '''quarter needs to be laid out differently so the pivot table will handle it correclty.
    We need <YEAR> <QUARTER> instead of <QUARTER> <YEAR>'''

    quarter, year = x['Quarter'].split(" ")
    return f"{year} {quarter}"

def find_quarter_number(x, date_range=None):
    '''apply_quarter_layout has issues as not all companies report nicely on the quarters that we're used to.
    find_quarter_number fixes this issue by being passed the date_range of when the company actually reported and creates its own quarters from that.
    This time though, 'quarters' are just numbers going from 0,1,2,3....
    
    Quarters are just integers now'''

    i = 0

    while date_range[i] < x['Transaction Date'] and i+1 < len(date_range):
        i = i + 1

        
    return i

def convert_format(df_backup=None, actuals_backup=None, project_path=None):
    '''Converts format of df_backup and actuals_backup into something that can be used.
    Mostly done by adding Month, Year and Quarter Columns.
    Makes the Date column the index of actuals
    Removes ECRI from teh column names in df_backup'''
    

    #Adding Month, Year and Quarter
    data = list(df_backup['Transaction Date'].str.split("/"))
    new_data = pd.DataFrame(data, columns = ["Month", "Year"])

    #The reason we're doing this is as sometimes df's index doesn't count up from 0:len(df)-1. It would skip numbers and be all over the place
    #This was a problem as the index of new_data would be "correct", and I'd be unable to concatenate the two on their indices.
    df_backup.reset_index(inplace=True)
    df_backup.drop("index", axis=1,inplace=True)

    try:
        df_backup = pd.concat([df_backup, new_data], axis=1)
    except:
        df_backup.to_pickle("/Convert_format_df_backup_error.pkl")
        new_data.to_pickle("/Convert_format_new_data_error.pkl")
        raise ValueError("The shapes aren't matching up for df_backup. Made a couple error files in the project_path.")

    #This is totally wrong. What the hell... This only applies in situtations where the company's quarters line up nicely with normal
    def return_quarter(x):
        month = x['Month']
        year = x['Year']
        month = int(month)
        quarter = int(math.ceil(month/3))
        return f"Q{quarter} {year}"
    df_backup['Quarter'] = df_backup.apply(return_quarter, axis=1)

    df_backup['Total Spend'] = df_backup['Total Spend'].astype(float) 

    if "Date" in actuals_backup.columns:
        actuals_backup.set_index("Date", inplace=True)

    if "ECRI Facility ID" in df_backup.columns:
        df_backup.rename(columns={'ECRI Facility ID': 'Vendor Facility ID'}, inplace=True)
        
    return [actuals_backup, df_backup]

def convert_time(X, column):
    '''    Turns the annoying dd/mm/yyyy formats we get in the data into datetime objects
    Also is able to handle mm/dd/yyyy and Month-Year formats
    
    column - the column in X you want to change.'''
    
    if isinstance(X[column].iloc[0], str):

        tran_X = X[column].str.split("/", expand=True)
        
        def apply_2partdate(x): #Here the columnsa re month, year
            return dt.datetime(int(x[1]), int(x[0]), 1)
        
        def apply_3partdate(x): #Here the columns in tran_X are day, month, year
            return dt.datetime(int(x[2]), int(x[1]), 1)
        
        def apply_stringdate(x): #Where it's like Aug-19
            return dt.datetime(int("20"+str(x[1])), x[0], 1)

        
        if tran_X.shape[1] == 2: #date is of format "month/year"
            X[column] = tran_X.apply(apply_2partdate, axis=1)
        elif tran_X.shape[1] == 3:
            X[column] = tran_X.apply(apply_3partdate, axis=1)
        elif tran_X.shape[1] == 1:
            tran_X = tran_X[0].str.split("-", expand=True)
            
            month_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep':9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
            tran_X.replace({0: month_dict}, inplace=True)
            
            X[column] = tran_X.apply(apply_stringdate, axis=1)
                
        else:
            print(tran_X)
            raise ValueError("Some strange things going down.")
        #else:
        #    raise ValueError(f"There are {tran_X.shape[1]} parts to the date")
        
    return X

def convert_actuals(company=None):
    if company == None:
        raise ValueError("Need to pass a company to get actuals for.")

    #This is for the newer versions where i can have dates for companies that have non standard reporting quarters.
    #This just selects the date columns and the GD Manufacturer column.
    #ndf = pd.read_excel("BBG_Revenue _Actuals.xlsx", engine='openpyxl')
    #ndf = ndf[['GD Manufacturer Name']+ndf.columns[10:].tolist()]
    # ndf = ndf[ndf['GD Manufacturer Name']==company]
    # ndf.drop('BBG Company Name', axis=1, inplace=True)
    # ndf = ndf[ndf.columns[1:]].T.reset_index(drop=True)
    # ndf.columns = ['Date', 'Revenue']

    #Changed code to also include quarters. This is so that pre_make_consensus can handle situations where quarters don't mean what they generally do.
    #ie, Q1 != 31-Mar
    ndf = pd.read_excel("BBG_Revenue _Actuals.xlsx", engine='openpyxl')

    #The 'GD Manufacturer Name' is not always exactly equal to the 'BBG Company Name'
    #As the company variable actually holds the 'BBG Company Name' value, when you refer to ndf['GD Manufacturer Name']==company, you might not get anything at all
    #For example, I've had to deal with 'Laboratory Corporation of America Holdings' vs 'Laboratory Corp of America Holdings'
    #The below two lines just make sure I'm using hte 'GD Manufacturer Name' value instead of hte 'BBG Company Name' one.
    #company_names = ndf[['GD Manufacturer Name', 'BBG Company Name']]
    #try:
        #company = company_names[company_names['BBG Company Name']==company].drop_duplicates()['GD Manufacturer Name'].values[0]
    #except IndexError as e:
        #We check for the BBG spelling of the company name first.
        #However for whatever reason it's not always the used spelling and we have to use the 'GD Manufacturer Name' spelling instead
        #This error is caught below:
        #if str(e) == "index 0 is out of bounds for axis 0 with size 0":
            #company = company

    #We just find which column index "Q1 2014" is in, then remember that one
    q1col = 0
    counter = 0
    for column in ndf.columns:
        if column == 'Q1 2014':
            q1col = counter
        counter += 1

    #We select columns from "Q1 2014" onwards (and also include "GD Manufcaturer Name")
    ndf = ndf[['GD Manufacturer Name']+ndf.columns[q1col:].tolist()]
    ndf = ndf[(ndf['GD Manufacturer Name']==company)|(ndf['GD Manufacturer Name']=='Quarters')]

    # ndf = ndf[['GD Manufacturer Name']+ndf.columns[10:].tolist()]
    # ndf = ndf[(ndf['GD Manufacturer Name']==company)|(ndf['GD Manufacturer Name']=='Quarters')]
    # ndf.drop('BBG Company Name', axis=1, inplace=True)

    if 'BBG Company Name' in ndf.columns:
        ndf.drop('BBG Company Name', axis=1, inplace=True)
    
    ndf = ndf[ndf.columns[1:]].T.reset_index(drop=True)

    #if ndf has 3 columns, they will be: ['Quarters', 'Date', 'Revenue']
    #Else if ndf has 4 columns, they will be: ['Quarters', 'Date', 'Revenue', 'Reporting Date'] (Reporting Date is the date
    # that teh quearterly earnings are published)
    if ndf.shape[1] == 3:
        ndf.columns = ['Quarters', 'Date', 'Revenue']
    elif ndf.shape[1] ==4:
        ndf.columns = ['Quarters', 'Date', 'Revenue', 'Reporting Date']
    else:
        ndf.to_excel("Messed up number of columns ndf.xlsx")
        raise ValueError(f"ndf has {ndf.shape[1]} columns when it should only have 3 or 4. Saving ndf to pipelinefunction's root folder")

    #Some dates may not have values and have NaNs. This messes up the typecasting.
    ndf.dropna(inplace=True)

    ndf['Revenue'] = ndf['Revenue'].astype('float')
    ndf.dropna(inplace=True)

    #We may also have custom date_ranges depending on the company's quarters
    #Ignore the first 2 years. This is so that seasonality and lasso_reg can work on the first date that we pass.
    date_range = ndf['Date']
    
    #for i in np.arange(0, len(date_range)):
    for i in date_range.index:
        #Check that date ranges are actually strings.
        if isinstance(date_range[i], str):
            #I assumed that the dates could either be in forms "dd-mm-yyy" or "dd/mm/yyyy"
            wrong_format = date_range[i]
            if "-" in wrong_format:
                seperator="-"
            elif "/" in wrong_format:
                seperator = "/"
            else:
                raise ValueError(f"Neither '-' nor '/' are in the wrongly formatted date ({wrong_format})")
            day, month, year = wrong_format.split(seperator)
            day = int(day)
            month = int(month)
            year = int(year)

            date_range[i] = dt.datetime(year, month, day)
        
    ndf['Date'] = date_range
    
    ndf.set_index("Date", inplace=True)
    
    date_range = ndf.index
        
    # try:
    #     date_range = date_range[date_range>dt.datetime(2016,1,1)]
    # except:
    #     raise ValueError(f"date_range in pipe.convert_actuals is problematic: {date_range}")


    #This is for the older versions of  Naresh's BBG files.

    # #Get Actuals from Naresh's file into a useful format.
    # ndf = pd.read_excel("GlobalData - Actuals - All Public Companies SKU.xlsx")
    # #Just select the columns with data in them.
    # ndf = ndf[['GD Manufacturer Name']+ndf.columns[10:].tolist()]
    # ndf = ndf[ndf['GD Manufacturer Name']==company]
    # ndf = ndf[ndf.columns[1:]].T
    # ndf.reset_index(inplace=True)
    # ndf.columns=['Date', 'Revenue']
    # ndf.set_index("Date", inplace=True)

    return ndf, date_range

def apply_interpret_quarters(x):
    '''Helper Function for pre_make_consensus'''
    #If it's a string we have a bunch of stuff we gotta do
    if isinstance(x['Consensus for quarter'],str):
        year, quarter = x['Quarter String'].split(" ")
        year = int(year)
        quarter = int(quarter[1])
        month = quarter*3 + 1
        if month > 12:
            year = year+1
            month = 1
        #We're adding +1 as we want the last day in teh quarter month. To do this we go into the first day of the NEXT quarter.then -1 day.
        first_date = dt.datetime(year, month, 1)
        actual_date = first_date - dt.timedelta(days=1)
    else: #Otherwise assume it's a datetime.
        actual_date = x['Consensus for quarter']
    
    return actual_date

def apply_timestamp_to_datetime(x, column=None):
    '''Sometimes one column ends up being numpy timestamps which really screws with my dt.datetime objects. This converts the column you want into dt.datetimes.
    '''

    if column is None:
        raise ValueError("No column value passed for apply_timestamp_to_datetime!")

    try:
        year = x[column].year
        month = x[column].month
        day = x[column].day
    except AttributeError:
        print(f"x[column] = {x[column]}")
    
    return dt.datetime(year, month, day)

def create_reporting_dataframe(company=None, ticker=None, path=None):
    '''Here we create the reporting dataframe that we'll use to merge with the df dataframe.
    It opens the master file by itself and handles everything from there.
    It's one argument is 'company', which is the name of the company you're interested in.
    '''
    
    if company is None:
        raise ValueError("No company passed to 'create_main_dataframe'")
        
    if ticker is None:
        raise ValueError("No ticker passed to 'create_main_dataframe'")
        
    if path is None:
        raise ValueError("No path passed to 'create_main_dataframe'")
    
    #We need reporting dates from rdf. And we get the actual quarter close dates from main.
    filename = path+f"/{ticker}.xlsx"
    try:
        rdf = pd.read_excel(filename, sheet_name="Dates", engine='openpyxl')
    except FileNotFoundError:
        rdf = pd.read_excel(path+"/Estimations.xlsx", sheet_name="Dates", engine='openpyxl')
        
    main = pd.read_excel("C:\\Users\\james.spencer\\OneDrive - GlobalData PLC\\Desktop\\Current Work\\CodingProjects\\ActualsPrediction\\GlobalData - Actuals - All Public Companies SKU.xlsx", engine='openpyxl', sheet_name='Data_WithDates')
    
    #Select only the quarters row and the date/value rows of the company we're interested in
    main = main[(main['GD Manufacturer Name']=='Quarters')|(main['GD Manufacturer Name']==company)]
    
    #This is me discarding all the columsn that aren't relevant to quarter-date-revenue
    main = pd.DataFrame(main['Category']).merge(right=main[main.columns[11:]], left_index=True, right_index=True)
    
    main = main.T
    main.columns = ['Quarter', 'Date', 'Value']
    
    main.drop('Category', inplace=True)

    #There may be 'unnamed' rows due to me adding extra data in the master file that needs to be trimmed out.
    for row in main.index:
        if 'Unnamed' in row:
            main.drop(row, inplace=True)
            
    def change_quarters(x):
        quarter, year = x['Quarter'].split(" ")
        return f"{year}{quarter}"

    #Change quarter into the format I want "2014Q1" instead of "Q1 2014"
    main['Quarter'] = main.apply(change_quarters, axis=1)
    
    main.set_index('Quarter', inplace=True)
    
    main['Date'] = main['Date'].apply(lambda x: pd.Timestamp(x))
    
    main.drop('Value', axis=1, inplace=True)
    main.columns=['Consensus for quarter']
    
    rdf = get_reporting_table(ticker=ticker, path=path)
    
    main = main.merge(right=rdf, left_index=True, right_index=True)
    
    return main

def get_reporting_table(ticker = None, path = None):
    '''This gets the reporting dates for the corresponding company for make_reporting_dataframe
    '''
    
    if path is None:
        raise ValueError(f"No path passed to get_reporting_table")
        
    if ticker is None:
        raise ValueError("No ticker passed to get_reporting_table")
        
    filename = path+f"/{ticker}.xlsx" 
    try:
        rdf = pd.read_excel(filename, sheet_name=2, engine='openpyxl')
        
        #There's a bunch of extra rows that we discard in only grabbing these two
        rdf = rdf[(rdf['Consensus Estimate']=='Revision_Date')|(rdf['Consensus Estimate']=='Reporting Date')]
    except FileNotFoundError:
        #In this situation, we're dealing with an aggregate file, so some extra stuff has to go in to handle it.

        #Header=None or else it makes the first company's information the header which messes with that company
        rdf = pd.read_excel(path + "/Estimations.xlsx", sheet_name="Dates", engine='openpyxl',header=None)

        #We're only interested in the rows in rdf that hold data corresponding to our ticker of interest.
        rows_of_interest = []

        #We are going to iterate through every row in rdf[1] looking for the ticker of interest.
        for i in np.arange(len(rdf[1])):
            
            #You want the rows that correspond to your ticker. This will give you the Consensus Estimate and Reporting Date rows
            #But you still want the Revision_Date rows, which is what the next IF statement is for.
            if rdf[1][i] == ticker:
                rows_of_interest.append(i)
                
                #Revision Date rows dont have anything in column 1, the ticker column, but they're always the next row down from
                #Consensus Estimate rows. So you check column 0 in your current row to see if it is the 'Consensus Estimate' row.
                #If it is, then you know the next row down is the Revision Date row so you grab it as well.
                if rdf[0][i] == 'Consensus Estimate':
                    rows_of_interest.append(i+1)

        #Now you should have a dataframe with 3 rows:
        # - Consensus Estimate (really just the quarters column)
        # - Revision Date (important for reasons that I've forgotten)
        # - Reporting Date
        rdf = rdf.loc[rows_of_interest]

        #if you have the wrong number of rows I'll throw an error and we'll figure it out.
        if rdf.shape[0] != 3:
            rdf.to_pickle(path+"/create_reporting_dataframe improper length.pkl")
            raise ValueError(f"rdf in create_reporting_dataframe has improper length of {rdf.shape[0]}! It should have a length of 3 rows. Saving file to 'create_reporting_dataframe improper length.pkl'")
            
            
        #This makes the Quarter row our header, which we convert into our index.
        rdf.columns = rdf.iloc[0]
        #This "deletes" the first row as its already our header
        rdf = rdf.iloc[1:]
    
    #Only want the rows talking about quarters.
    rdf =rdf[rdf.columns[2:]].T
    
    #Originally quarters were inthe index and were not named
    rdf.reset_index(inplace=True)
    if len(rdf.columns)==3:
        rdf.columns = ['Quarter', 'Revision Date', 'Reporting Date']
    else:
        rdf.to_pickle(path+"/get_reporting_table error.pkl")
        raise ValueError(f"rdf in get_reporting_table has {len(rdf.columns)} columns, which is wrong! Saving to the project folder as 'get_reporting_table error.pkl'")
    
    
    #Change quarter to proper format "2014Q1" instead of "Q1 2014"
    def apply_change_quarters(x):
        year, quarter = x['Quarter'].split(" ")

        return f"{year}{quarter}"

    rdf['Quarter'] = rdf.apply(apply_change_quarters, axis=1)
    
    if "Revision Date" in rdf.columns:
        rdf.drop("Revision Date", axis=1, inplace=True)
    rdf.set_index("Quarter", inplace=True)
    
    return rdf

def pre_make_consensus(ticker=None, company=None, path=None):
    '''This makes the consensus estimates file for one company at a time.
    It does this by opening individual ticker files and then looking at the consensus file section.
    This function relies heavily on create_reporting_dataframe to do its job
    '''

    if path is None:
        raise ValueError(f"No path provided.")

    if isinstance(ticker, list):
        raise ValueError(f"Passed several tickers into ticker: {ticker}")
    elif ticker is None:
        raise ValueError("Nonetype passed for ticker argument in pre_make_consensus")
    else:
        #I assume that I've just been passed one company as a string here.
        pass

    if company is None:
        raise ValueError("You didn't give me a company name in pre_make_consensus")
    if not isinstance(company, str):
        raise TypeError(f"the company argument, {company} is not of type string in pre_make_consensus")

    filename = path+f"/{ticker}.xlsx"
    try:
        #Naresh suddenly gave me a bunch of companies all aggregated into one excel file, so oopening files one by one may no longer work.
        #If that's the case, I'll have some code here that tries to open it normally, and if that fails, open the aggregated files and work from there.
        df_backup = pd.read_excel(filename, sheet_name=1, engine='openpyxl') #sheet_name=1 corresponds to the "Estimates" tab, I should probably change this though.
    except FileNotFoundError:
        df_backup = pd.read_excel(path+"/Estimations.xlsx", sheet_name="Estimates", engine='openpyxl')
        
        #We only want to take out very specific columns, most of them are trash because they relate to other companies

        #We always want columns 0, 1 & 2. This is because they are the normal dates on which everything is based
        columns_of_interest = [0,1,2]

        i = 0

        #The company names are stored in df.loc[0], so we iterate through those to see which columns correspond to the company we're interested in.
        #we add [i, i+1] as each company has two columns. However, the second column's value is always blank. So you have to add it as it won't match up to ticker.
        for column in df_backup.loc[0].values:
            if column == ticker:
                print("match!")
                columns_of_interest = columns_of_interest + [i, i+1]
            else:
                pass
            i = i +1
            
        #Now we only take our column of interest and we're good to go! (hopefully)
        df_backup = df_backup[df_backup.columns[columns_of_interest]]
    reporting = create_reporting_dataframe(company=company, ticker=ticker, path=path)

    df_backup.to_pickle(path+f"/df_backup.pkl")
    reporting.to_pickle(path+f"/reporting.pkl")

    df = df_backup.copy()
    df.columns = df.iloc[0]

    #Rename all columns so they're using company name instead of ticker name.
    # new_columns = []

    # for column in df.columns:
    #     if ticker in column:
    #         new_columns

    #Change columns to someting more usable with Adarsh's code

    new_columns = ["Dates", "Quarters", "Consensus for quarter"]

    for columns in df.columns.tolist()[1::2][1:]:
        new_columns.append(f"{columns} Consensus Value")
        new_columns.append(f"{columns} Revision Date")
        
    df.columns = new_columns

    #Cut out weird sub-columns that don't do anything.
    df = df.loc[2:]

    new_columns = []

    for column in df.columns:
        if ticker in column:
            column = column.replace(ticker, company)
        else:
            pass
        
        new_columns.append(column)
        
    df.columns = new_columns

    #Our df 'Consensus for quarter' column is fundamentally wrong if the company doesn't report on normal quarters.
    #So we have to replace it with the column in our 'reporting' dataframe

    df.drop("Consensus for quarter", inplace=True, axis=1)
    df = df.merge(right=reporting, left_on='Quarters', right_index=True)
    df.dropna(inplace=True)
    #df.drop("Reporting Date", inplace=True, axis=1)

    try:
        df[f'{company} Revision Date'] = df.apply(apply_timestamp_to_datetime, axis=1, column = f"{company} Revision Date")
    except:
        df.to_pickle(path + f"/df.pkl")
        print("Woops")
        raise ValueError("Woops")

    df[f'{company} Consensus Time Delta'] = df[f"{company} Revision Date"] - df['Consensus for quarter']

    consensus_df = pd.DataFrame(df['Consensus for quarter'].drop_duplicates()).set_index("Consensus for quarter")
    consensus_df.index.rename("Date",inplace=True)

    #I really don't know why you need the .apply(lambda x: x.date()) in the back, I just know that you do.
    reporting['Consensus for quarter'] = pd.to_datetime(reporting['Consensus for quarter'], unit='s').apply(lambda x: x.date())
    reporting['Reporting Date'] = pd.to_datetime(reporting['Reporting Date'], unit='s').apply(lambda x: x.date())

    reporting.set_index("Consensus for quarter", inplace=True)

    times = []
    dates = []
    consensi = []

    #Incapable of converting df['Consensus for quarter'] from timestamp to datetime
    #for some reason. Think it's bugs with pandas.
    #So here I have to make a list where I do each date one at a time and then iterate through them.
    consensus_for_quarter_converted = []

    for date in df['Consensus for quarter'].drop_duplicates().tolist():
        consensus_for_quarter_converted.append(dt.datetime(date.year, date.month, date.day))


    for date in consensus_for_quarter_converted:
        df.to_pickle(path+f"/df.pkl")
        reporting.to_pickle(path+f"/reporting.pkl")
        #try:
        wow = df[df['Consensus for quarter']==date]
        #consensus_times = wow[wow[f'{company} Consensus Time Delta']<=pd.Timedelta("15 days")][f"{company} Consensus Time Delta"]

        reporting.index = pd.to_datetime(reporting.index)
        reporting_date = reporting.loc[date]['Reporting Date']#.values[0].astype('M8[ms]').astype('O')
    
        #reporting_date is now a dt.date object, which you can't compare with a dt.datetime object. Pain in the ass. But we convert.
        #We have to do this one by one instead as a whole row operation as pd.to_datetime and my other tricks
        #all insist on converting into timestamp instead for some damn reason.
        reporting_date = dt.datetime(reporting_date.year, reporting_date.month, reporting_date.day)

        consensus_times = wow[wow['Dates']<=reporting_date][f"{company} Consensus Time Delta"]
        best_time = consensus_times.max()
        times.append(best_time.days)
        consensus_value = wow[wow[f'{company} Consensus Time Delta']==best_time][f"{company} Consensus Value"].values[0]

        dates.append(date)
        consensi.append(consensus_value)
        # except IndexError:
        #     print(f"Index error in pipe.pre_make_consensus for date: {date}")
        #     pass
        # except KeyError:
        #     print(f"Key error in pipe.pre_make_consensus for date: {date}")
        #     pass
    reporting.to_pickle(path + f"/reporting.pkl")
    df.to_pickle(path + f"/df1.pkl")

    new_consensus = pd.DataFrame(zip(dates, consensi), columns = ['Dates', f'{company} Latest Consensus'])
    new_consensus.set_index("Dates", inplace=True)

    consensus_df = consensus_df.merge(right=new_consensus, left_index=True, right_index=True, how='outer')

    return consensus_df

def do_download(company=None, connection=None):
    '''Downloads all information pertaining to one company at a time, then returns 
    a dataframe with all that info'''

    if company==None:
        raise ValueError(f"Did not receive any company to download in do_download")

    df_list = []
    
    companylist = company.split(' ')
    companyfirst = companylist[0]
    
    sql = f"SELECT * FROM ECRI_DataImport WHERE [GD_ManufacturerName] LIKE ('%{companyfirst}%')"
    data = pd.read_sql(sql, connection,chunksize=1000000) #I'll only pull a million records at a time.
    for chunk in data:
        df_list.append(pd.DataFrame(chunk))
    
    #Sometimes df_list is just an empty list so pd.concat() fails
    #This happens when we have no data that's relevant at all and adarsh_keep just returns blanks
    try:
        ndf = pd.concat(df_list)
    except ValueError as e:
        if str(e) == "No objects to concatenate":
            ndf = pd.DataFrame({})
            raise ValueError(f"There was no data that was downloaded for {company}")

    return ndf

def download_sql(order=None, project_path=None):
    '''Iterates through all the companies in the order file and orders them one by one before saving them into the original folder'''
    if order==None:
        raise ValueError(f"Did not receive any companies to download")
    if project_path==None:
        raise ValueError(f"Not passed a project_path destination to save the files to.")

    #AWS log in information

    driver = '{ODBC Driver 17 for SQL Server}'
    server = '10.55.21.236'
    #server = '63.33.172.39'
    database = 'Medical_ECRI'
    uid = 'Robertson'
    pwd = 'RpA*z3$UCP8'

    connection = pyodbc.connect(f"DRIVER={driver};SERVER={server};DATABASE={database};UID={uid};PWD={pwd};timeout=600")

    #cursor = connection.cursor()


    for company in order:
        df = do_download(company=company, connection=connection)
        df.to_pickle(project_path + f"/{company}.pkl")

def consistent_facilities(X,actuals,top,percent,date_range=None):
    
    def apply_alter_quarter(x):
        '''Change how quarters are displayed so that they are automatically sorted correctly'''
        quarter, year = x['Quarter'].split(" ")
        return f"{year} {quarter}"
    
    X['Quarter'] = X.apply(find_quarter_number, axis=1, date_range=date_range)
    
    if not "New Total Spend" in X.columns:
        X.rename(columns={"Total Spend": "New Total Spend"}, inplace=True)
    nX = pd.DataFrame(X.groupby(['Vendor Facility ID', 'Quarter'])['New Total Spend'].sum()).reset_index()
    nX = nX.pivot(index='Quarter', columns='Vendor Facility ID', values='New Total Spend').fillna(0)
    nX.to_csv(r'C:\\Users\\Aidan.Robertson\\Downloads\\nX df.csv')
    
    new_nX_index = []
    
    for quarter in nX.index:
        new_nX_index.append(date_range[quarter])


    try:
        #nX.index = pd.date_range(dt.datetime(2014,1,1), last_date, freq='Q')
        nX.index = new_nX_index
    except:
        raise ValueError(f"Couldn't fit {pd.date_range(dt.datetime(2014,1,1), last_date, freq='Q')} into\n{nX.index}")
    lag = pd.read_csv('most_consistent_facilities.csv')
    
    most_consistent = lag[lag['Facility ID'].isin(nX.columns)]
    most_consistent = most_consistent.reset_index(drop=True)
    
    top = round(percent*(len(most_consistent)))
    
    best = most_consistent['Facility ID'][0:top]
    best.to_csv(r'C:\\Users\\Aidan.Robertson\\Downloads\\fac_con_values.csv')
    consist_aggregated = nX[best].sum(axis=1)
    
    consistent_agg = pd.DataFrame(consist_aggregated, columns=[f'top {top}'])
    
    return consistent_agg
    
def find_best_facilities(X, actuals, top, date_range=None):
    '''Return a list of the most correlated facilities with the actuals from most correlated to least.
    X = the data dataframe
    actuals = the actuals of the company
    mode = Takes 1 or 2 as arguments. In Mode 1, will return only correlated_agg. 
    In Mode 2, will return [correlated_agg, X]. Where X is a subset of the X where only the top 50% more correlated facilities are included.
    '''
    
    def apply_alter_quarter(x):
        '''Change how quarters are displayed so that they are automatically sorted correctly'''
        quarter, year = x['Quarter'].split(" ")
        return f"{year} {quarter}"

    X['Quarter'] = X.apply(find_quarter_number, axis=1, date_range=date_range)
    
    #X['altered quarter'] = X.apply(apply_alter_quarter, axis=1)
    
    if not "New Total Spend" in X.columns:
        X.rename(columns={"Total Spend": "New Total Spend"}, inplace=True)
    nX = pd.DataFrame(X.groupby(['Vendor Facility ID', 'Quarter'])['New Total Spend'].sum()).reset_index()
    nX = nX.pivot(index='Quarter', columns='Vendor Facility ID', values='New Total Spend').fillna(0)
    nX.to_csv(r'C:\\Users\\Aidan.Robertson\\Downloads\\nX df.csv')
    # #Find out what the last quarter of the nX data is. and make our date range go up to and include it.
    # last_quarter = nX.index[-1]
    # year, quarter = last_quarter.split(" ")
    # quarter = int(quarter[-1])
    # year = int(year)
    
    # last_date = dt.datetime(year, quarter*3, 1) + pd.Timedelta("31 days")

    new_nX_index = []

    # for date in nX.index:
    #     year, quarter = date.split(" ")
    #     year = int(year)
    #     month = int(quarter[1])*3 + 1 #Ultimately we want the last day of each quarter. To get this we take the first day of the NEXT quarter, then subtract one day.
    #     if month==13: #In the case we roll over to 1-Jan-20XX
    #         year = year+1
    #         month = 1 
    #     date = dt.datetime(year, month, 1) - pd.Timedelta("1 day")
    #     new_nX_index.append(date)

    for quarter in nX.index:
        new_nX_index.append(date_range[quarter])


    try:
        #nX.index = pd.date_range(dt.datetime(2014,1,1), last_date, freq='Q')
        nX.index = new_nX_index
    except:
        raise ValueError(f"Couldn't fit {pd.date_range(dt.datetime(2014,1,1), last_date, freq='Q')} into\n{nX.index}")
    
    actuals_matrix= nX.merge(right=actuals, left_index=True, right_index=True, how='inner')
    actuals_matrix.to_csv(r'C:\\Users\\Aidan.Robertson\\Downloads\\actuals matrix.csv')
    best = pd.DataFrame(actuals_matrix.corr()['Revenue'].sort_values(ascending=False))
    best.to_csv(r'C:\\Users\\Aidan.Robertson\\Downloads\\fac_corr_values.csv')
    best.drop('Revenue', inplace=True)
    best.columns=['correlations']
    

    best = best[0:top]
    
    best_aggregated = nX[best.index].sum(axis=1)
    
    correlated_agg = pd.DataFrame(best_aggregated, columns=[f'top {top}'])
    
    if mode == 1:
        #return pd.concat([X,correlated_agg], axis=1)
        return correlated_agg
    
    elif mode == 2:
        #This is only really relevant for mode==2...
        top_50_percent = best[0:math.floor(len(best)/2)] #These are the indexes of the top 50% most correlated facilities.
    
    
        X = X[X['Vendor Facility ID'].isin(top_50_percent.index)]
        
        return pd.concat([correlated_agg, X], axis=1)
    
    else:
        raise ValueError(f"mode {mode} ISNT A VALID CHOICE")

def reformat_csv(X):
    #Do the first one to get rid of ellipses like "Standardized Vendor Name...Long"
    new_columns = []
    for column in X.columns.tolist():
        new_columns.append(column.replace("...", " - "))
    X.columns = new_columns    

    #Do the second one to get rid of single periods like "Transaction.Date"
    new_columns = []
    for column in X.columns.tolist():
        new_columns.append(column.replace(".", " "))
    X.columns = new_columns
    return X

def keeptimeperiod(X=None, date=None, daycutoff=15, project_path=None, date_range=None):
    '''Basically keeps everything that was reported at least 15 days after the quarter it was in
    Also only keeps a subset of facilities that have reported during the last quarter.'''

    if X.empty:
        raise ValueError("X is empty in keeptimeperiod")

    #month = date.month
    #year = date.year
    
    #Or should this be Load Date? I'm still confused.
    #This should be Load Date. Change at a later point.
    X = X[(X['Load Date'] < date + pd.Timedelta("15 days")) & (X['Transaction Date']<date)]
    
    #quarter = math.ceil(month/3)
    
    #qoi = f"{year} Q{quarter}"
    
    #We're interested in the latest quarter, now the Quarter column is just filled with integers so fin dht ebiggest one.
    qoi = X['Quarter'].max()

    #only consider the facilities that have reported during the latest quarter
    facilities = X[X['Quarter']==qoi]['Vendor Facility ID'].unique()
    
    #Remove all other facilities that did not report during the latest quarter.
    X = X[X['Vendor Facility ID'].isin(facilities)]

    #X.to_excel(project_path + f"/Fuse {date.day}-{date.month}-{date.year} X stuff.xlsx")

    # X['keep'] = X.apply(apply_new_keep, axis=1)

    try:
        #X['keep'] = X.apply(apply_adarsh_keep, axis=1)
        X['keep'] = X.apply(apply_new_keep, axis=1)
    except Exception as e:
        X.to_excel(project_path + "/adarshkeep issue.xlsx")
        raise ValueError (str(e))

    #Comment this out later and replace with above block
    #X['keep'] = X.apply(apply_new_keep,axis=1)
    
    

    X = X[X['keep']]
    
    X.drop('keep', axis=1, inplace=True)

    return X

def PivotTable(X=None, date_range=None):
    '''Creates a pivot table with columns by facillity type and index is by date
    return this pivot table.'''
    #X['Quarter'] = X.apply(apply_quarter_layout, axis=1)
    X['Quarter'] = X.apply(find_quarter_number, axis=1, date_range=date_range)

    #Now group New Total Spend by quarter and vendor facility ID then make a pivot table.
    #X = X.groupby(['Quarter', 'Facility Type'])['New Total Spend'].sum()
    #X = X.groupby(features)['New Total Spend'].sum()
    X = X.groupby(['Quarter', 'Facility Type'])['New Total Spend'].sum()

    X=pd.DataFrame(X)

    X.reset_index(inplace=True)

    #Converting to final_Input format / feature engineering

    #Adarsh found that he gets teh most accurate results when __ (which feature set gives best results again).
    #He always aggregates all hospitals/bed sizes/wtv. I should explore breaking these down to see if I can get
    #anything from that. Or other features?

    def apply_new_index(x):
        '''Our indexs are currently of form "2014 Q1", when we want a datetime.'''
        quarter, year = x['Quarter'].split(" ")
        year = int(year)
        quarter = int(quarter[-1])
        month = (quarter*3) + 1 #as we want last day of quarter. So we actually -1 day after all this

        #edge case of month = 13
        if month > 12:
            month = 1
            year+=1

        date = dt.datetime(year,month,1) - dt.timedelta(days=1)

        return date

    #X['Date'] = X.apply(apply_new_index, axis=1)
    def apply_get_date(x, date_range=None):
        ''' Here I turn the quarter numbers into dates using date range.'''

        return date_range[x['Quarter']]

    #X.to_excel("Date.xlsx")

    X['Date'] = X.apply(apply_get_date, date_range=date_range, axis=1)

    X.drop("Quarter", axis=1, inplace=True)

    X = X.pivot(index='Date', columns='Facility Type', values='New Total Spend')
    X['total'] = X.sum(axis=1)
    X.reset_index(inplace=True)
    
    #This line ensures that there is no feature that has "NA" in it at any point.
    #In my automated feature selection, I want to ensure that what i pick in Q1 2014 is still applicable in Q2 2019 or whatever...
    #How do I do this though? Should I do one big groupby at the start?.... Probably.....
    X.dropna(axis=1, inplace=True)

    X.set_index('Date', inplace=True)
    
    return X

def TopX(X=None, top=25, actuals= None, on=False, date=None):
    '''Append a new column about the revenue of the <top> most correlated facilities
    actuals - need to have an "actuals" frame to be passed or you can't find the most correlated
    top - an integer argument. This just states how many correlated facilities you want to contain in your TopX columns
    on - Whether or not you even want TopX to append a column, if on=False, you don't append anything.
    X - the pivot table you make in PivotTable
    date - the date we are treating as the current date. Cut off everything after this point'''
    if not on:
        return X
    
    if actuals is None:
        raise ValueError("TopX needs to be passed actuals to work!")
    
    def apply_alter_quarter(x):
        '''Change how quarters are displayed so that they are automatically sorted correctly'''
        quarter, year = x['Quarter'].split(" ")
        return f"{quarter} {year}"
    
    actuals = actuals[actuals.index<=date]
    X = X[X['Load Date']<=date]

    df = X

    #Create an altered quarter that will be naturally sorted while using groupby
    #As Q4 2019 and Q3 2019 aren't sorted correctly but 2019 Q4 and 2019 Q3 are.
    df['altered quarter'] = df.apply(apply_alter_quarter, axis=1)

    ndf = pd.DataFrame(df.groupby(['Vendor Facility ID', 'altered quarter'])['New Total Spend'].sum()).reset_index()
    
    #pivot table so that correlation matrix can be made.
    ndf = ndf.pivot(index='altered quarter', columns='Vendor Facility ID', values='New Total Spend').fillna(0)
    
    #Find out what the last quarter of the ndf data is. and make our date range go up to and include it.
    last_quarter = ndf.index[-1]
    year, quarter = last_quarter.split(" ")
    quarter = int(quarter[-1])
    year = int(year)
    
    #We now have our last date
    last_date = dt.datetime(year, quarter*3, 1) + pd.Timedelta("31 days")

    #Create the date_range index for it
    ndf.index = pd.date_range(dt.datetime(2014,1,1), last_date, freq='Q')

    #return [ndf, actuals]
    
    #ndf= ndf.merge(right=actuals, left_index=True, right_index=True, how='inner')
    
    ndf = pd.concat([ndf, actuals], axis=1)

    best = pd.DataFrame(ndf.corr()['Revenue'].sort_values(ascending=False))
    best.drop('Revenue', inplace=True)
    best.columns=['correlations']
    best = best[0:top]

    best_aggregated = ndf[best.index].sum(axis=1)
    
    df = pd.DataFrame(best_aggregated, columns=[f'top {top}'])

    #print(df)
    
    #columns = df.columns.tolist()

    return df

def Xlog(X=None, on=False, project_path=None):
    '''Takes log of dataframe if this is turned on, otherwise just passes the df trhough'''

    if not on:
        return X

    if X.empty and on:
        raise ValueError("please pass something to log, or turn Xlog off")

    #So I don't get tripped up with log(0)
    X.replace(to_replace=0, value=1, inplace=True)

    try:
        return np.log(X.astype('float64'))
    except:
        X.to_pickle(f"{project_path}/Failure to take log.pkl")
        raise ValueError(f"Failed to take log of: \n {X}")

def create_quarterly_data(date=None, df=None, actuals=None, date_range=None, project_path=None):
    ####OBSOLETE#########################
    '''This function is responsible for making the X_log data that the model will then use to predict the next quarter.
    Args:

    date - Current date?
    date_range - Does this one also change?
    df
    actuals
    project_path


    returns: (X_log, y_log)
    X_log (including current quarter)
    y_log

    '''

    df = reformat_csv(df)
    actuals = reformat_csv(actuals)

    df['Quarter'] = df.apply(apply_quarter_layout, axis=1)

    df = convert_time(X=df, column='Transaction Date')
    df = convert_time(X=df, column='Load Date')

    #This is not always a great idea. Sometimes general buying behaviour is inherently spiky.

    #If correlated_agg throws an error, it's because the previous line used actuals_backup instead of actuals
    #Argh shouldn't I have done this after keeptimeperiod? This creates a large "leakage" problem where I have access to data I otherwise wouldn't have....
    correlated_agg = find_best_facilities(X=df, actuals = actuals, date_range=date_range, top=100)
    #correlated_agg = pipe.find_best_facilities(X=df, actuals = actuals_backup, date_range=date_range, top=100)



    df = keeptimeperiod(X=df, date=date, project_path=project_path)

    df = PivotTable(X=df, date_range=date_range)

    #df = pd.concat([df,topX], axis=1)
    df.dropna(inplace=True)

    df = df.merge(right=correlated_agg, how='inner', left_index=True, right_index=True)
    #df = pd.concat([df, correlated_agg], axis=1)

    df = Xlog(X=df, on=True, project_path=project_path)

    df = Seasonality(X=df, on=True, actuals=actuals)#, pattern=[-0.693147,-0.693147,-0.693147,2.484907])#actuals=actuals)

    ##########################################################
    ##########################################################

    my_columns = df.columns.tolist()

    X_log = df[my_columns].dropna(axis=1) #dropna as sklearn can't handle nans very well.
    #X.drop("Hospital", axis=1,inplace=True)
    #X.drop("total", axis=1, inplace=True)
    #actuals.set_index("Date", inplace=True)
    y = actuals[['Revenue']]

    #y.index = pd.date_range(dt.datetime(2014,1,1), last_date, freq='Q')
    y = y[y.index<=date]

    #You want X_log and y_log to start at the same time. So the first dates of both ahve to match.
    #Here I cut out all dates in X that are before the first date in y
    X_log = X_log[X_log.index>=y.index.min()]

    #Do something to make sure they're the same length at this point.

    X_index = set(X_log.index)
    y_index = set(y.index)

    both_indexs = X_index.intersection(y_index)

    y = y.loc[both_indexs]
    X_log = X_log.loc[both_indexs]

    y.sort_index(inplace=True)
    X_log.sort_index(inplace=True)



    #Creating my train and test sets.
    #We exclude the last data set as that's the most recent quarter, and the one we're going to predict on.
    #X_log = X[:-1]
    #y_log = y[:-1]

    #Create our test data.
    X_test_log = np.array(X_log.iloc[-1]).reshape(1, -1)
    test_date = X_log.index[-1]
    y_test_log = np.log(y.iloc[y.shape[0]-1])


    X_log = X_log[:-1]
    if test_date in X_log.index:
        raise ValueError(f"test_date: {test_date} is in X_log.index")
    y_log = np.log(y[:-1])

    #Only want last 12 periods (3 years) of data to train on.
    if len(X_log)>=12:
        X_log = X_log[-11:]
        y_log = y_log[-11:]

    return X_log, y_log

def make_ErrorObject(**error_object):
    '''Make and save an object holding a lot of hopefully useful information'''

    name = error_object['name']
    date = error_object['date']
    project_path = error_object['project_path']

    # error_object = {}
    # error_object['Date'] = date
    # error_object['date_range'] = date_range
    # error_object['X_log'] = X_log
    # error_object['y_log'] = y_log
    # error_object['x_index'] = X_index
    # error_object['y_index'] = y_index
    # error_object['actuals'] = actuals
    # error_object['both_indexs'] = both_indexs
    # error_object['df'] = df
    # error_object['new_thing']
    pickle.dump(error_object, open(f"{project_path}/{name}{date.day}-{date.month}-{date.year} log.pkl", "wb"))

    print('Error Object printed!')

def make_date_ranges(date_range = None, df = None):
    '''date_range holds all the quarterly reporting dates from beginning of hospital database to present day.
    we only start our analysis from Q1 2016, so we iterate from there onwards.
    Also have to take care of situation where there is no data in Voldemort at date_range[0]
    In these cases, have to start our dates of interest at a point where I would have had 8 quarters of data already...
    '''

    #We only want stuff from at LEAST Q1 2016 onwards as we need that seasonality signal (This won't always be true...)
    dates_of_interest = date_range[date_range>dt.datetime(2016,1,1)]
    
    #We need the data to contain at least 8 quarters worth of data. So we find the 9th oldest quarter:
    cut_off_quarter = df['Quarter'].sort_values(ascending=True).drop_duplicates().iloc[9]

    data_cut_off_date = df[df['Quarter'] == cut_off_quarter]['Transaction Date'].min()

    dates_of_interest = dates_of_interest[dates_of_interest>=data_cut_off_date]

    return dates_of_interest, date_range

def predict_one_quarter(date=None, df=None, actuals=None, date_range=None, project_path=None, regression_type=None, regularization_param=None, company = None, consistent_agg=None):
    '''This function is in charge of predicting the next quarter of revenue for a given company
    Args:

    date - THE ONLY ONE THAT CHANGES BETWEEN DIFFERENT INSTANCES OF THIS FUNCTION
    date_range - Does this one also change?
    df
    actuals
    project_path
    regression_type


    returns:
    Here we have the variables our returned values get appended to.
    lasso_reg
    coefficients.append(lasso_reg.coef_)
    dates.append(date)
    preds.append(y_hat)
    test_logs.append(X_test_log)
    X_logs_list.append(X_log)
    y

    '''
    #try:
    # df = reformat_csv(df)
    # actuals = reformat_csv(actuals)

    #df['Quarter'] = df.apply(apply_quarter_layout, axis=1)

    # df = convert_time(X=df, column='Transaction Date')
    # df = convert_time(X=df, column='Load Date')

    #This is not always a great idea. Sometimes general buying behaviour is inherently spiky.

    #If correlated_agg throws an error, it's because the previous line used actuals_backup instead of actuals
    '''
    correlated_agg = find_best_facilities(X=df, actuals = actuals, date_range=date_range, top=100)
    #correlated_agg = pipe.find_best_facilities(X=df, actuals = actuals_backup, date_range=date_range, top=100)

    ddf = df.copy()

    if df.shape[0] <= 8:
        raise ValueError(f"pre-keeptimeperiod df.shape[1] = {df.shape[0]}")
    '''
    df = keeptimeperiod(X=df, date=date, project_path=project_path)
    df.to_csv(r'C:\\Users\\Aidan.Robertson\\Downloads\\input after ktp.csv')

    '''
    if df.shape[0] <= 8:
        df.to_pickle(f"{project_path}/postkeeptimeperioddf{company}{date}.pkl")
        pickle.dump(error_object, open(f"{project_path}/{name}{date.day}-{date.month}-{date.year} log.pkl", "wb"))
        raise ValueError(f"post-keeptimeperiod df.shape[1] = {df.shape[0]}")
    '''
    ndf = df.copy()

    df = PivotTable(X=df, date_range=date_range)
    df.to_csv(r'C:\\Users\\Aidan.Robertson\\Downloads\\input after Pivot.csv')

    '''
    if df.shape[0] <= 9:
        error_object = {}
        error_object['name'] = 'Post PivotTable'
        error_object['ndf'] = ndf
        error_object['df'] = df
        error_object['ddf'] = ddf
        error_object['date'] = date
        error_object['project_path'] = project_path
        error_object['date_range'] = date_range
        error_object['actuals'] = actuals



        make_ErrorObject(**error_object)

        #df.to_pickle(f"{project_path}/df_PivotTable{date.day}-{date.month}-{date.year}.pkl")
        raise ValueError(f"post-PivotTable for date {date.day}-{date.month}-{date.year} df.shape[0] = {df.shape[0]}")
    '''
    #df = pd.concat([df,topX], axis=1)
    df.dropna(inplace=True)

    df = df.merge(right=consistent_agg, how='inner', left_index=True, right_index=True)
    df.to_csv(r'C:\\Users\\Aidan.Robertson\\Downloads\\input after corr_lag.csv')

    #df = pd.concat([df, correlated_agg], axis=1)
    #df.to_excel('input data before model.xlsx')
    '''
    if df.shape[0] <= 8:
        df.to_pickle(f"{project_path}/postcorrdf{company}{date}.pkl")
        raise ValueError(f"post-correlated_agg df.shape[1] = {df.shape[0]}")
    '''
    df = Xlog(X=df, on=True, project_path=project_path)
    df.to_csv(r'C:\\Users\\Aidan.Robertson\\Downloads\\input after Xlog.csv')
    '''
    if df.shape[0] <= 8:
        raise ValueError(f"post-Xlog df.shape[1] = {df.shape[0]}")
    '''
    df = Seasonality(X=df, on=True, actuals=actuals)#, pattern=[-0.693147,-0.693147,-0.693147,2.484907])#actuals=actuals)
    #print(df)
    df.to_csv(r'C:\\Users\\Aidan.Robertson\\Downloads\\input after seasonality.csv')
    '''
    if df.shape[0] <= 8:
        raise ValueError(f"post-Seasonality df.shape[1] = {df.shape[1]}")
    '''

    ##########################################################
    ##########################################################

    my_columns = df.columns.tolist()

    X_log = df[my_columns].dropna(axis=1) #dropna as sklearn can't handle nans very well.
    X_log.to_csv(r'C:\\Users\\Aidan.Robertson\\Downloads\\input after df columns.csv')
    '''
    if X_log.shape[0] <= 1:
        raise ValueError(f"We messed up df. There's no data in it!")
    '''
    #X.drop("Hospital", axis=1,inplace=True)
    #X.drop("total", axis=1, inplace=True)
    #actuals.set_index("Date", inplace=True)
    y = actuals[['Revenue']]

    #y.index = pd.date_range(dt.datetime(2014,1,1), last_date, freq='Q')
    y = y[y.index<=date]

    #You want X_log and y_log to start at the same time. So the first dates of both ahve to match.
    #Here I cut out all dates in X that are before the first date in y
    X_log = X_log[X_log.index>=y.index.min()]
    X_log.to_csv(r'C:\\Users\\Aidan.Robertson\\Downloads\\input after cutoff.csv')
    '''
    if X_log.shape[0] <= 1:
        raise ValueError("X_log is too short!")
    '''
    #Do something to make sure they're the same length at this point.

    X_index = set(X_log.index)
    y_index = set(y.index)

    both_indexs = X_index.intersection(y_index)

    y = y.loc[both_indexs]
    X_log = X_log.loc[both_indexs]

    y.sort_index(inplace=True)
    X_log.sort_index(inplace=True)
    X_log.to_csv(r'C:\\Users\\Aidan.Robertson\\Downloads\\input after length match.csv')


    #Creating my train and test sets.
    #We exclude the last data set as that's the most recent quarter, and the one we're going to predict on.
    #X_log = X[:-1]
    #y_log = y[:-1]

    #Create our test data.
    X_test_log = np.array(X_log.iloc[-1]).reshape(1, -1)
    test_date = X_log.index[-1]
    test_log_x = pd.DataFrame(X_test_log)
    test_log_x.to_csv(r'C:\\Users\\Aidan.Robertson\\Downloads\\X_test_log.csv')
    '''
    if X_log.shape[0] <= 3:
        date = pd.Series(date)
        date.to_csv('date error.csv')
        raise ValueError(f"X_log has no data in it! (We must have removed it all!)")
    '''

    y_test_log = np.log(y.iloc[y.shape[0]-1])

    X_log.to_csv(r'C:\\Users\\Aidan.Robertson\\Downloads\\input after test data.csv')
    X_log = X_log[:-1]
    if test_date in X_log.index:
        raise ValueError(f"test_date: {test_date} is in X_log.index")
    y_log = np.log(y[:-1])

    #Only want last 12 periods (3 years) of data to train on.
    if len(X_log)>=12:
        X_log = X_log[-11:]
        y_log = y_log[-11:]

    #Why would I even remove my data...

    # adding a constant
    #lasso_reg = Lasso(alpha=regularization_param)
    #lasso_reg = LinearRegression()
    X_log.to_csv(r'C:\\Users\\Aidan.Robertson\\Downloads\\input before model.csv')
    if regression_type == 'Lasso':
        lasso_reg = Lasso(alpha=regularization_param)
    elif regression_type == 'Ridge':
        lasso_reg = Ridge(alpha=regularization_param,fit_intercept=True)
    elif regression_type == 'OLS':
        lasso_reg = LinearRegression()
    try:
        lasso_reg.fit(X_log, y_log)
    #Sometimes X_log and y_log's shapes don't match up.
    #This is a pain to troubleshoot as poolmap makes life hard.
    except ValueError as e:
        error_object = {}
        error_object['date'] = date
        error_object['date_range'] = date_range
        error_object['X_log'] = X_log
        error_object['y_log'] = y_log
        error_object['x_index'] = X_index
        error_object['y_index'] = y_index
        error_object['actuals'] = actuals
        error_object['both_indexs'] = both_indexs
        error_object['df'] = df
        make_ErrorObject(**error_object)

        raise ValueError(e)


    #coefficients.append(lasso_reg.coef_)

    #coefficients.append(lasso_reg.coef_)
    #designed_features.append(X.columns)
    #model_date.append(date)

    y_hat = np.exp(lasso_reg.predict(X_test_log))
    pred_value = pd.DataFrame(y_hat)
    pred_value.to_csv(r'C:\\Users\\Aidan.Robertson\\Downloads\\pred value.csv')

    #If I'm using Ridge, then y_hat is an array of length one which is really annoying
    if isinstance(y_hat, np.ndarray):
        y_hat = y_hat[0]

    # dates.append(date)
    # preds.append(y_hat)
    # test_logs.append(X_test_log)
    # X_logs_list.append(X_log)

    # date_preds[date] = pd.DataFrame(zip(dates, preds), columns = ['Date', 'Predictions'])
    # date_preds[date].set_index("Date", inplace=True)

    #There's no error, I'm just tyring to make a log of the information at every point in time.
    error_object = {}
    error_object['name'] = company
    error_object['date'] = date
    error_object['project_path'] = project_path
    error_object['date_range'] = date_range
    error_object['X_log'] = X_log
    error_object['y_log'] = y_log
    error_object['x_index'] = X_index
    error_object['y_index'] = y_index
    error_object['actuals'] = actuals
    error_object['both_indexs'] = both_indexs
    error_object['df'] = df
    make_ErrorObject(**error_object)

    #These are all the values we need to return. I'm packing them together as a dict to make unpacking easier.
    return {'date':date, 'y_hat':y_hat, 'X_test_log':X_test_log, 'X_log':X_log, 'y':y, 'reg_model':lasso_reg}
    '''
    except Exception as f:
        print(f)
        
        file= open(f'{date.year} {date.month} {date.day} {company} predict exception.txt', 'wb')
        file.write(f)
       
        return f
   
    '''

def Seasonality(X=None, on=False, actuals=None, pattern=None):


    '''Seasonality is an optional pipeline element. If you feel that the time series youre trying to estimate has a seasonal
    component, turn this on using the "on" argument.
    I"m planning to have this auto-detect likely seasonal patterns and auto-apply them to the analysis. However, for now I"m
    going to keep the entry of the seasonal pattern as manual. This will be done by entering a list of possible weights

    Arguments:

    on - Takes "True" or "False" as arguments. If "True", then Seasonality will run and put a seasonal feature into the linear
    data set. Otherwise Seasonality won't run and seasonal data will not be placed in.

    pattern - Takes "None" or a list with length==4 as arguments. If "None" is received, then the transformer will try to find
    its own pattern to apply (it should also print this pattern so the user has some idea of a good starting pattern).
    However, if the user chooses to apply their own pattern, they must do so in a 4 part list. Each element will be the
    supposed seasonal component for that quarter for quarters 1 - 4.
    For example: [1, 1, 1, 12].

    actuals - Takes "None" or the actuals dataframe. If on=True, pattern=None and actuals=None, the Seasonality transformer will
    throw an error.

    NOTE: I'm just assuming that the actuals always start at Q1 in a given year. This assumption may not be correct.
    Off the top of my head, this should only really effect the situations where you give patterns manually. You'd just
    have to reorder the list you pass. So it might be [Q3, Q4, Q1, Q2] instead of the normal order.'''
    #actuals = np.log(actuals)

    '''It doesn't make sense to pass both pattern and actuals, so if both are passed then I will throw an error.
    It doesn't make sense to turn seasonality on and pass neither pattern or actuals so that'll throw an error too.
    Long story short is that you should either have pattern OR actuals if you're wanting to use Seasonality'''
    
    if on==False:
        return X #quit early as Seasonality shouldn't do anything if it's off.
    
    if pattern==None and actuals.empty==True:
        raise ValueError("The Seasonality Transformer is on but has no information to infer a seasonality behaviour from!")
    
    if pattern!=None and actuals!=None:
        raise ValueError("Seasonality has been passed both a manual pattern and the actuals. I don't know which one to use so I'm raising this error.")

    #Need number of years to properly apply pattern
    #X = pd.DataFrame(X, columns = pivot_and_topX.get_feature_names())
    #X.set_index("Pivot__Date", inplace=True)

    number_of_years = X.iloc[-1].name.year - X.iloc[0].name.year +1
    
    if not actuals is None:
        #pattern = tsa.seasonal.seasonal_decompose(actuals['Revenue'], period=4).seasonal
        pattern = sm.tsa.seasonal_decompose(actuals['Revenue'], period=4).seasonal[0:4].values.tolist()
        pattern = pd.DataFrame(pattern*number_of_years, columns=['Seasonality'])
        pattern = pattern[:X.shape[0]]
        pattern.index = X.index
        return pd.concat([X, pattern], axis=1)
    elif not pattern is None:            
        #Note that pattern now has more rows than X. So you can't copy over X's index. Need to make same length
        pattern = pd.DataFrame(pattern*number_of_years, columns=['Seasonality'])

        #Take only the first N rows of pattern and discard the rest. Where N is the number of rows in X.
        #They should now be of equal length
        pattern = pattern[:X.shape[0]]
        #print(f"X.index: {X.index}")
        pattern.index = X.index
        #So if pattern = [1,1,1,12] and number_of_years=2, then
        # pattern*number_of_years = [1,1,1,12,1,1,1,12]
        return pd.concat([X, pattern], axis=1)
        # try:
        #     #print(pd.concat([X, pattern]), axis=1)
        #     return pd.concat([X, pattern], axis=1)
        # except:
        #     print("Fuck fuck")
        #     return [X, pattern]
    else:
        raise ValueError(f"actuals = {actuals}, pattern = {pattern}")
    
    raise ValueError("Somehow got to the end without returning anything....")

def results_unpacker(result_list = None):
    '''In which we turn the mess in result_list into a 2 level dictionary.
    The first level of keys are the regression types: OLS, Ridge, Lasso
    The 2nd level of keys are the regularization parameters: 0.01, 0.05 etc...

    So then result_list['OLS'][0.01] will return a dataframe with the index being Dates, and one column, 'prediction' which
    holds all the predictions that that 'OLS, 0.01' predictor made.

    Args:
    result_list: The collection of dates and y_hats for each date, reg_param, reg_type combination. Also includes a TON of other
                    diagnostic information

    Returns:
    predictions: A 2 level dictionary holding the predictions for each combination of 
    '''

    if result_list == None:
        raise ValueError("Didnt' get passed anything to results_unpacker")

    predictions = {}

    #This entire mess is to tease out our predictions for each date/regression type/regularization parameter from our multiprocessing output
    #After this, we will have a dictionary of dictionaries:
    #predictions = {reg_type1: {reg_param1: pred_df1, reg_param2: pred_df2 etc}}
    for reg_type in ['Lasso', 'Ridge', 'OLS']:
        for reg_param in [0.0025, 0.005, 0.01, 0.025]:
            preds = {'date':[], 'prediction':[]}
            for i in np.arange(len(result_list[f'{reg_type}, {reg_param}'])):
            #     print(result_list['Lasso, 0.0025'][i]['date'][0])
            #     print(result_list['Lasso, 0.0025'][i]['y_hat'][0])
            #     print(result_list['Lasso, 0.0025'][i]['X_test_log'][0])
            #     print(result_list['Lasso, 0.0025'][i]['X_log'][0])
            #     print("---------------------------------------")
                date = result_list[f'{reg_type}, {reg_param}'][i]['date']
                pred = result_list[f'{reg_type}, {reg_param}'][i]['y_hat']
                reg_model = result_list[f'{reg_type}, {reg_param}'][i]['reg_model']
                preds['date'].append(date)
                preds['prediction'].append(pred)
            pred_df = pd.DataFrame(preds).set_index('date')
            if reg_type in predictions.keys():
                predictions[reg_type][reg_param] = (pred_df, reg_model)
            else:
                predictions[reg_type] = {reg_param:(pred_df, reg_model)}
                
    return predictions

    #Now we go through these to find out which one is the best fit.
    #result_list['Lasso, 0.0025'][-1]['reg_model'].coef_