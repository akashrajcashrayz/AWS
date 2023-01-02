import streamlit as st
import plotly.figure_factory as ff
import pandas as pd
import plotly.graph_objects as go
from finta import TA
import datetime
import numpy as np
from PIL import Image
import sklearn
import keras
from sklearn import preprocessing
#import multiprocessing as mp
#pool = mp.Pool(mp.cpu_count())
#from tensorflow import keras
#import efficientnet.keras as efn
# Add histogram data

df = pd.read_csv('final_data_complete.csv')
df = df[['date','open','high','low','close','volume']]
#model = keras.models.load_model('best_model.h5')
#model = keras.models.load_model('best_num_model_11_X.h5')
model = keras.models.load_model('best_num_model_11_X1.h5')
#model1 = keras.models.load_model('effnet_noisy_epoch100_B7.h5')
#candle_model =keras.models.load_model('best_num_candle_model.h5') 
config = dict({'scrollZoom': True})  
# Group data together

df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M').dt.strftime('%Y-%m-%d %H:%M')
df['date_day'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M')
#year = st.selectbox('select year',(2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,2019, 2020, 2021, 2022))
#month = st.selectbox('select month',(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12))
#date1 = st.selectbox('select date',([z for z in range(1,32)]))

entry_date = st.date_input(
    "Start Date",datetime.date(2022, 10, 24),min_value = datetime.date(2008, 12, 11))
exit_date = st.date_input(
    "End Date",datetime.date(2022, 10, 24) ,max_value = datetime.date(2022, 10, 24))

period = st.number_input('Insert  period for DEMA',14)
threshold = st.number_input('Input Threshold',0.7)

af = st.number_input('Insert  AF for SAR',0.2)
amax = st.number_input('Insert  AMAX for SAR',0.2)

#chart_type = st.selectbox('chart_type',('normal','remove_candles','pred_pattern'))


def givimg(data_buy):
    buy_chart =  go.Scatter( y=data_buy,name = 'DEMA',marker_line_color="MediumPurple", marker_color="lightskyblue")

    fig = go.Figure(data=buy_chart)
    fig.update_layout(showlegend=False)

    #         #x axis
    fig.update_xaxes(visible=False)

    #         #y axis    
    fig.update_yaxes(visible=False)
    fig.write_image("temp.jpg")
    im = Image.open(r"temp.jpg")
    newsize = (64, 64)
    im1 = im.resize(newsize)
    import numpy as np
    I = np.asarray(im1)

    return I
def ret_pred(dk,threshold):
    dk = pd.DataFrame(dk)
    dk.columns = ['buy_prob','sell_prob']
    dk['dema_num_pred'] = 'non_trend'
    dk.loc[dk.loc[dk['buy_prob'] >= threshold].index,'dema_num_pred'] = 'bullish'
    dk.loc[dk.loc[dk['sell_prob'] >= threshold].index,'dema_num_pred'] = 'bearish'
    return dk['dema_num_pred']

def rem_candle(df):
  red_index = df.loc[(  df['open'] > df['DEMA']  )  & (df['close']  < df['DEMA'] )].index
  green_index = df.loc[(  df['open'] < df['DEMA']  )  & (df['close']  > df['DEMA'] )].index
  df.at[red_index,'red'] = 1
  df.at[green_index,'green'] = 1
  sig_index = df.loc[(  df['red'] == 1  )  | (df['green']  == 1 )].index
  df.at[sig_index,'sig'] = 1


  df['open'] = np.where(df['sig']!=1,np.NaN,df['open'])
  df['close'] = np.where(df['sig']!=1,np.NaN,df['close'])
  df['high'] = np.where(df['sig']!=1,np.NaN,df['high'])
  df['low'] = np.where(df['sig']!=1,np.NaN,df['low'])
  return df

def entryfunc(openn,high,low,close,DEMA,SAR):
    candle_size = close - openn
    if candle_size< 0:
        openwig = high - openn
        closewig = close - low
        closedema_size = DEMA - close
        perc_closedema_size = (closedema_size/abs(candle_size))*100    
        if (SAR > high) and (perc_closedema_size > 50) and ( openn >DEMA ) and ( close < DEMA ): 
            return 'bearish'


    if candle_size > 0:
        openwig = openn - low
        closewig = high - close
        closedema_size = close - DEMA
        perc_closedema_size = (closedema_size/abs(candle_size))*100    
        if (SAR < low) and (perc_closedema_size > 50) and ( openn <DEMA ) and ( close > DEMA ): 
            return 'bullish'        
    return 'non_trend'  
    
def entrext(dff):
    dff.loc[dff.loc[dff['dema_line_buy'] > 0].index,'sig'] = 'buy'
    #dff['sigsig']=dff['sig'].loc[dff['sig'].shift(-1) != dff['sig']]
    print(dff.loc[dff['dema_line_buy'] > 0].index)

    dff.loc[dff.loc[dff['SAR'] > dff['high']].index,'sig'] = 'exbuy'
    print(dff.loc[dff['SAR'] > dff['high']].index)
    #dff['exsigsig']=dff['exsig'].loc[dff['exsig'].shift(1) != dff['exsig']]



    dff.loc[dff.loc[dff['dema_line_sell'] > 0].index,'sig'] = 'sell'
    #dff['sigsigs']=dff['sigs'].loc[dff['sigs'].shift(-1) != dff['sigs']]
    print(dff.loc[dff['dema_line_sell'] > 0].index)

    dff.loc[dff.loc[dff['SAR'] < dff['low']].index ,'sig'] = 'exsell'
    #dff['exsigsigs']=dff['exsigs'].loc[dff['exsigs'].shift(1) != dff['exsigs']]
    print(dff.loc[dff['SAR'] < dff['low']].index)





    dx = dff.loc[dff['sig'].shift(1) != dff['sig']]
    signal_entry = list(dx['sig'].loc[dx['sig'] == 'buy'].index)
    signal_exit = list(dx.loc[dx['sig'] == 'exbuy'].index)
    signal_entrys = list(dx['sig'].loc[dx['sig'] == 'sell'].index)
    signal_exits = list(dx.loc[dx['sig'] == 'exsell'].index)
    print(signal_entry)
    print(signal_exit)
    print(signal_entrys)
    print(signal_exits)

    buy_enter_exit = []    
    for i in signal_entry:
        for j in signal_exit:

            if (j > i) :

                buy_enter_exit.append([i,j])
                break

    sell_enter_exit = []
    for i in signal_entrys:
        for j in signal_exits:
            if j > i :
                sell_enter_exit.append([i,j])
                break         

    return   buy_enter_exit,sell_enter_exit        

def candle_t(ind,df_buy_sell):
    candle_type = np.argmax(candle_model.predict([list(df_buy_sell[['high','low','close']].iloc[ind].values)]))
    if candle_type == 0:
        return 'bullish'
    if candle_type == 1:
        return 'non_trend'
    if candle_type == 2:
        return 'bearish'
    return None

def get_index_trades(dchec):
    dchec.loc[dchec.loc[(dchec['SAR'] < dchec['low']) & (dchec['prediction']== 'no_signal')].index,'prediction' ] = 'sell_exit'
    dchec.loc[dchec.loc[(dchec['SAR'] > dchec['high']) & (dchec['prediction']== 'no_signal')].index,'prediction' ] = 'buy_exit'
    dff = dchec.mask(dchec['prediction'].shift(1) == dchec['prediction'])




    signal_entry = list((dff.loc[dff['prediction'] == 'buy_signal']).index)
    signal_exit = list((dff.loc[dff['prediction'] == 'buy_exit']).index)

    signal_entrys = list((dff.loc[dff['prediction'] == 'sell_signal']).index)
    signal_exits = list((dff.loc[dff['prediction'] == 'sell_exit']).index)

    compz = []
    buy_enter_exit = [] 


    for i in signal_entry:
        compz.append([i,'buy_signal'])
    for i in signal_exit:
        compz.append([i,'buy_exit'])
    for i in signal_entrys:
        compz.append([i,'sell_signal'])
    for i in signal_exits:
        compz.append([i,'sell_exit'])    

    compz = (sorted(compz))
    in_trade = False
    trade_trend = 'no_trend'
    buy_entr = []
    buy_ext = []
    sell_entr = []
    sell_ext = []
    print(compz)
    buy_entr_ext_1 = []
    sell_entr_ext_1 = []
    for i in compz:
        if 0==0 :
            print(i[0])
            if (i[-1] == 'buy_signal') and (in_trade == False) and (trade_trend == 'no_trend'):
                in_trade = True
                trade_trend = 'buy_signal'      
                buy_entr.append(i)
            if (i[-1] == 'sell_signal') and (in_trade == False) and (trade_trend == 'no_trend'):
                in_trade = True
                trade_trend = 'sell_signal'  
                sell_entr.append(i)        

            if (i[-1] == 'buy_exit') and (in_trade == True) and (trade_trend == 'buy_signal'):
                in_trade = False        
                trade_trend = 'no_trend'
                buy_ext.append(i)  
                buy_entr_ext_1.append([buy_entr[-1][0],i[0],dff['close'].iloc[buy_entr[-1][0]],dff['close'].iloc[i[0]],'bullish' ,-dff['close'].iloc[buy_entr[-1][0]]+dff['close'].iloc[i[0]],dff['date'].iloc[buy_entr[-1][0]],dff['date'].iloc[i[0]]])

            if (i[-1] == 'sell_exit') and (in_trade == True) and (trade_trend == 'sell_signal'):      
                in_trade = False        
                trade_trend = 'no_trend'
                sell_ext.append(i)  
                sell_entr_ext_1.append([sell_entr[-1][0],i[0],dff['close'].iloc[sell_entr[-1][0]],dff['close'].iloc[i[0]],'bearish',dff['close'].iloc[sell_entr[-1][0]]-dff['close'].iloc[i[0]] ,dff['date'].iloc[sell_entr[-1][0]],dff['date'].iloc[i[0]]])       


    trades = buy_entr + buy_ext + sell_entr + sell_ext
    trades = (sorted(trades))
    print(trades)

    res_df = pd.DataFrame(sorted(buy_entr_ext_1 + sell_entr_ext_1))
    res_df.columns = ['entry_time1','exit_time1','entry_price','exit_price','trend','PNL','entry_time','exit_time']
    res_df['cumsum_PNL'] = res_df['PNL'].cumsum()    
    res_df = res_df[['entry_time','exit_time','entry_price','exit_price','trend','PNL','cumsum_PNL']]
    
    
    
    
    
    
    return buy_entr_ext_1,sell_entr_ext_1,res_df


def get_index_trades1(dchec):
    dchec.loc[dchec.loc[(dchec['SAR'] < dchec['low']) & (dchec['prediction']== 'no_signal')].index,'prediction' ] = 'sell_exit'
    dchec.loc[dchec.loc[(dchec['SAR'] > dchec['high']) & (dchec['prediction']== 'no_signal')].index,'prediction' ] = 'buy_exit'
    dff = dchec.mask(dchec['prediction'].shift(1) == dchec['prediction'])




    dff['predictionx'] = dff['prediction'].replace('no_signal',np.nan).fillna(method='ffill').dropna()
    dff['predictionx'] = dff['predictionx'].loc[dff['predictionx'].shift(1) != dff['predictionx']].dropna()
    dff['predictionx'] = pd.DataFrame(dff['predictionx']).dropna()

    signal_entry = list((dff.loc[dff['predictionx'] == 'buy_signal']).index)
    signal_exit = list((dff.loc[dff['predictionx'] == 'buy_exit']).index)

    signal_entrys = list((dff.loc[dff['predictionx'] == 'sell_signal']).index)
    signal_exits = list((dff.loc[dff['predictionx'] == 'sell_exit']).index)
    dff.loc[signal_entry,'buy_entry_exit'] = 'buy_signal'
    dff.loc[signal_exit,'buy_entry_exit'] = 'buy_exit'
    dff.loc[signal_entrys,'sell_entry_exit'] = 'sell_signal'
    dff.loc[signal_exits,'sell_entry_exit'] = 'sell_exit'
    dff['buy_entry_exit'] = dff['buy_entry_exit'].fillna(method='ffill')
    dff['sell_entry_exit'] = dff['sell_entry_exit'].fillna(method='ffill')
    dff['buy_entry_exit'] = dff['buy_entry_exit'].loc[dff['buy_entry_exit'].shift(1) != dff['buy_entry_exit']].dropna()
    dff['sell_entry_exit'] = dff['sell_entry_exit'].loc[dff['sell_entry_exit'].shift(1) != dff['sell_entry_exit']].dropna()
    
    buy_entr_ext_1 = []
    sell_entr_ext_1 = []
    start_buy = list(dff['buy_entry_exit'].loc[dff['buy_entry_exit'] == 'buy_signal'].index)
    end_buy = list(dff['buy_entry_exit'].loc[dff['buy_entry_exit'] == 'buy_exit'].index)
    if len(start_buy)>0 and len(end_buy) >0:
        start_buy = start_buy[0]
        end_buy = end_buy[-1]
        
        
        signal_entry = list(dff['buy_entry_exit'].loc[dff['buy_entry_exit'] == 'buy_signal'].loc[:end_buy].index)
        signal_exit = list(dff['buy_entry_exit'].loc[(dff['buy_entry_exit'] == 'buy_exit') ].loc[start_buy:].index)
        buydf = pd.DataFrame()

        buydf['buy_signal'] = signal_entry
        buydf['buy_exit'] = signal_exit
        buy_entr_ext_1 = buydf.values.tolist()
    


    start_sell = list(dff['sell_entry_exit'].loc[dff['sell_entry_exit'] == 'sell_signal'].index)
    end_sell = list(dff['sell_entry_exit'].loc[dff['sell_entry_exit'] == 'sell_exit'].index)
    if len(start_sell)> 0 and len(end_sell) >0:
        start_sell = start_sell[0]
        end_sell = end_sell[-1]
        signal_entrys = list(dff['sell_entry_exit'].loc[dff['sell_entry_exit'] == 'sell_signal'].loc[:end_sell].index)
        signal_exits = list(dff['sell_entry_exit'].loc[(dff['sell_entry_exit'] == 'sell_exit') ].loc[start_sell:].index)
        selldf = pd.DataFrame()

        selldf['sell_signal'] = signal_entrys
        selldf['sell_exit'] = signal_exits
        
        sell_entr_ext_1 = selldf.values.tolist()
    




    entr_buy = dff.loc[signal_entry][['date','close']].reset_index(drop = True)
    ext_buy = dff.loc[signal_exit][['date','close']].reset_index(drop = True)
    entr_sell = dff.loc[signal_entrys][['date','close']].reset_index(drop = True)
    ext_sell = dff.loc[signal_exits][['date','close']].reset_index(drop = True)
    entr_ex_buy = pd.concat([entr_buy, ext_buy], axis=1)
    entr_ex_buy['trend'] = 'bullish'

    entr_ex_buy.columns = ['entry_time','entry_price','exit_time','exit_price','trend']
    entr_ex_buy['PNL'] = entr_ex_buy['exit_price'] - entr_ex_buy['entry_price']

    entr_ex_sell = pd.concat([entr_sell, ext_sell], axis=1)
    entr_ex_sell['trend'] = 'bearish'
    entr_ex_sell.columns = ['entry_time','entry_price','exit_time','exit_price','trend']
    entr_ex_sell['PNL'] = entr_ex_sell['entry_price'] - entr_ex_sell['exit_price']
    res_df = pd.concat([entr_ex_buy, entr_ex_sell], axis=0).sort_values('entry_time')
    res_df['cumsum_PNL'] = res_df['PNL'].cumsum()
    
    
    
    print(buy_entr_ext_1,sell_entr_ext_1)
    
    
    return buy_entr_ext_1,sell_entr_ext_1,res_df


def loop_fun(i):

    if i < len(buy_enter_exit)  :  


        fig.add_vrect(x0=df_year['date'].iloc[buy_enter_exit[i][0]],x1 = df_year['date'].iloc[buy_enter_exit[i][1]], opacity=0.25, line_width=0, fillcolor="green",annotation_text=df_year['close'].iloc[buy_enter_exit[i][1]] - df_year['close'].iloc[buy_enter_exit[i][0]], annotation_position="top right",)
                
                
                
                #fig.add_vline(x=df_year['date'].iloc[i[1]], line_width=2, line_dash="dash", line_color="MediumPurple")
    if i < len(sell_enter_exit)  :  
        fig.add_vrect(x0=df_year['date'].iloc[sell_enter_exit[i][0]],x1 = df_year['date'].iloc[sell_enter_exit[i][1]], opacity=0.25, line_width=0, fillcolor="red",annotation_text=df_year['close'].iloc[sell_enter_exit[i][0]] - df_year['close'].iloc[sell_enter_exit[i][1]], annotation_position="top left",)
                #fig.add_vline(x=df_year['date'].iloc[i[1]], line_width=2, line_dash="dash", line_color="DarkSlateGrey")        
  



    
    #Put all the instructions here

    return fig



def get_da(df_year,demcol):

    buysellmean = 0.00310


    
    df_year = df_year.loc[abs(df_year[['DEMA11','DEMA']].pct_change(axis = 1).iloc[:,-1]) > buysellmean]
    data_for_num_model = df_year[demcol + ['open','high','low','close']].pct_change(axis = 1).iloc[:,1:]



    
    #data_for_img_model = df_year[demcol]
    demadata_num = []
    dema_img = []
    trends = []
    df_year['trend'] = 'non_trend'
    df_year['candle_size'] = df_year['close'] - df_year['open']
    
    df_year['openwig'] = df_year['high'] - df_year['open']
    df_year['closewig'] = df_year['close'] - df_year['low']
    df_year['closedema_size'] = df_year['DEMA'] - df_year['close']
    df_year['perc_closedema_size'] = (df_year['closedema_size']/df_year['candle_size'].apply(abs)).mul(100)    
    bearind = df_year.loc[(df_year['candle_size'] < 0 ) & (df_year['SAR'] > df_year['high']) &  (df_year['SAR'].shift(11) > df_year['high'].shift(11))  & (df_year['perc_closedema_size'] > 50) & ( df_year['open'] >df_year['DEMA'] ) & ( df_year['close'] < df_year['DEMA'] )].index 
    df_year.loc[bearind,'trend'] = 'bearish'

    #df_year.to_csv('trend_bearish.csv')

    df_year['openwig'] = df_year['open'] - df_year['low']
    df_year['closewig'] = df_year['high'] - df_year['close']
    df_year['closedema_size'] = df_year['close'] - df_year['DEMA']
    df_year['perc_closedema_size'] = (df_year['closedema_size']/df_year['candle_size'].apply(abs)).mul(100)      
    bullind = df_year.loc[(df_year['candle_size'] > 0 ) & (df_year['SAR'] < df_year['low'])&  (df_year['SAR'].shift(11) < df_year['low'].shift(11))& (df_year['perc_closedema_size'] > 50) & ( df_year['open'] <df_year['DEMA'] ) & ( df_year['close'] > df_year['DEMA'] )].index
    df_year.loc[bullind,'trend'] = 'bullish' 
    
    #df_year.to_csv('trend_bullish.csv')    

    
    
    
    pred_num = model.predict(data_for_num_model)
    #pred_img = model1.predict(np.asarray(data_for_img_model))
    
    
    
    dfnedded = df_year
    dfnedded = dfnedded.reset_index(drop = True)
    #dfnedded = dfnedded.iloc[11:,:]
    dfnedded['dema_num_pred'] = ret_pred(pred_num,threshold)
    print(dfnedded)
    #dfnedded['dema_num_pred']  = pd.DataFrame(pred_num).idxmax(axis="columns").replace(0,'bullish').replace(1,'non_trend').replace(2,'bearish')
    
    
#     dfnedded['dema_img_pred']  = #pd.DataFrame(pred_img).idxmax(axis="columns").replace(0,'bullish').replace(1,'non_trend').replace(2,'bearish')

    dfnedded['prediction'] = 'no_signal'
    dfnedded.loc[dfnedded.loc[(dfnedded['dema_num_pred'] == 'bearish')  & (dfnedded['trend'] == 'bearish')].index,'prediction'] = 'sell_signal'
    dfnedded.loc[dfnedded.loc[(dfnedded['dema_num_pred'] == 'bullish') & (dfnedded['trend'] == 'bullish')].index,'prediction'] = 'buy_signal'
    #dfnedded.to_csv('dfnedded.csv')

    buy_enter_exit,sell_enter_exit,res_df = get_index_trades1(dfnedded)
    
    
        
    
    test_list = list(dfnedded.loc[(dfnedded['prediction'] == 'buy_signal')  ].index)
    res = [test_list[i + 1] - test_list[i] for i in range(len(test_list)-1)]
    sublistt = [[z for z in range(i-10,i+1)] for i in test_list]
    flat_list = [item for sublist in sublistt for item in sublist]    
    sor_index = sorted(list(set(flat_list)))
    sor_index = [i for i in sor_index if i >= 11]
#     sor_indexx = [i if i>11: else pass for i in sor_index]
#     sor_index = sor_indexx
#     s = []

#     for i in sor_index:
#         if i >11:
#             s.append(i)

#     sor_index = s
    dfnedded['dema_line_buy'] = dfnedded.loc[sor_index]['DEMA']


    test_list = list(dfnedded.loc[ (dfnedded['prediction'] == 'sell_signal') ].index)
    res = [test_list[i + 1] - test_list[i] for i in range(len(test_list)-1)]
    sublistt = [[z for z in range(i-10,i+1)] for i in test_list]
    flat_list = [item for sublist in sublistt for item in sublist]    
    sor_index = sorted(list(set(flat_list)))
    sor_index = [i for i in sor_index if i >= 11]
#    sor_index = sor_indexx
#     s = []

#     for i in sor_index:
#         if i >11:
#             s.append(i)

#     sor_index = s
    
    dfnedded['dema_line_sell'] = dfnedded.loc[sor_index]['DEMA']
    
    
    
#     dfnedded['dema_line_buy'] = dfnedded['dema_line_buy'] 
#     dfnedded['dema_line_sell'] = dfnedded['dema_line_sell']
#     dfnedded['DEMA'] = dfnedded['DEMA']

    
    
    return dfnedded,buy_enter_exit,sell_enter_exit,res_df
    
    

    
    
    
    






if entry_date: 

    #df_year = df.loc[(df['year'] == year) & (df['month'] == month ) & (df['date_1'] == date1 )]

    df_year  = df.loc[(df['date_day'].dt.date >= entry_date) & (df['date_day'].dt.date <= exit_date)]
    
    if len(df_year) >0:
        df_year = df_year.reset_index()       
          
        df_year['SAR'] = TA.SAR(df_year,af = af,amax = amax)  

        df_year['DEMA'] = TA.DEMA(df_year,period = period) 
        demcol = ['DEMA']
        for i in range(1,12):
            demcol.append(f'DEMA{i}')

            df_year[f'DEMA{i}'] = df_year['DEMA'].shift(i)

        df_year = df_year.dropna().reset_index(drop = True)    
        demcol = list(reversed(demcol))

        #df_year.to_csv('df_year.csv')
        
        df_year,buy_enter_exit,sell_enter_exit,res_df = get_da(df_year,demcol)
        res_df.to_csv(f'results/{entry_date}_{exit_date}.csv')        
 
        #buy_enter_exit,sell_enter_exit = entrext(df_year) 
         
        
        #if chart_type == 'remove_candles':
            #df_year = rem_candle(df_year)
            
            
    
        price = go.Candlestick(x=df_year['date'],
                        open=df_year['open'],
                        high=df_year['high'],
                        low=df_year['low'],
                        close=df_year['close'],name = 'price')

        DEMA =  go.Scatter(x=df_year['date'],y=df_year['DEMA'],name = 'DEMA',marker_line_color="MediumPurple", marker_color="lightskyblue")
        SAR =  go.Scatter(x=df_year['date'],y=df_year['SAR'],name = 'SAR',mode='markers',
                           marker_line_color="midnightblue", marker_color="lightskyblue",
                           marker_line_width=0.5, marker_size=2)                            
 
        clos =  go.Scatter(x=df_year['date'],y=df_year['close'],name = 'close')


        BDEMA =  go.Scatter(x=df_year['date'] ,y=df_year['dema_line_buy'] - 15,name = 'dema_line_buy',marker_line_color="lightskyblue", marker_color="green")
        SDEMA =  go.Scatter(x=df_year['date'] ,y=df_year['dema_line_sell'] + 15,name = 'dema_line_sell',marker_line_color="DarkSlateGrey", marker_color="red")
 
                        
        fig = go.Figure(data=[DEMA,BDEMA,SDEMA,price,SAR])
        #fig = pool.map(loop_fun, range(max(len(buy_enter_exit),len(sell_enter_exit))))
        maxb = max(len(buy_enter_exit),len(sell_enter_exit))
        for i in np.arange(maxb):
            if i < len(buy_enter_exit)  :  


                fig.add_vrect(x0=df_year['date'].iloc[buy_enter_exit[i][0]],x1 = df_year['date'].iloc[buy_enter_exit[i][1]], opacity=0.25, line_width=0, fillcolor="green",annotation_text=df_year['close'].iloc[buy_enter_exit[i][1]] - df_year['close'].iloc[buy_enter_exit[i][0]], annotation_position="top right",)
            
            
            
            #fig.add_vline(x=df_year['date'].iloc[i[1]], line_width=2, line_dash="dash", line_color="MediumPurple")
            if i < len(sell_enter_exit)  :  
                fig.add_vrect(x0=df_year['date'].iloc[sell_enter_exit[i][0]],x1 = df_year['date'].iloc[sell_enter_exit[i][1]], opacity=0.25, line_width=0, fillcolor="red",annotation_text=df_year['close'].iloc[sell_enter_exit[i][0]] - df_year['close'].iloc[sell_enter_exit[i][1]], annotation_position="top left",)
            #fig.add_vline(x=df_year['date'].iloc[i[1]], line_width=2, line_dash="dash", line_color="DarkSlateGrey")        
        
        
        #fig = go.Figure(data=[DEMA,BDEMA,SDEMA,SAR,price])                       
                        
                        

        fig.update_layout(xaxis_rangeslider_visible=False,xaxis = dict(type = "category",categoryorder = "category ascending"))
        #fig.update_xaxes(rangebreaks=[dict(values=df_year['date'])]) # hide dates with no values


        # Plot!

        st.plotly_chart(fig,config=config, use_container_width=True)



        
        st.dataframe(res_df)


    else:
        st.write('Data is not available for this date')
