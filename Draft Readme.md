# This is the Readme file for Group #3 Project #2
Bootcamp: Columbia Fintech<br>
Cohort: March-Sep 2021<br>
![wsb.jpg](wsb.jpg)![wsb-chart.png](wsb-chart.png)
## Goal
Determine it is possible to determine a stocks movement by looking at comments on Wall Street Bets (WSB) on Reddit. The project was more an investigation of possible approaches than a final production produt.
## Rationale
WSB was a forum to start short squeeze on stock being shorted by Hedge Funds
[wsb short squeeeze]
## Approach
We took the follow steps to develop senstivity scenarios (what-if)
1. [Read WSB](#1---Read-WSB)
2. [Data Wrangling](#2---Data-Wrangling)
3. [Run LSTM Scenarios on Data](#3---Run-LSTM-Scenarios)
4. [Evalute other ML Models](#4---Evalute-other-ML-Models)
5. [Other](#5---Other)


### 1 - Read WSB
(Ludo)

### 2 - Data Wrangling<br>
![datawrangling.png](datawrangling.png)
2.1. read WSB Data<br>
The first step is to read the WSB into a dataframe
![WSB_DF.png](WSB_DF.png)
2.2. Add sentiment Data <br>
We then add sentiment data <br>

![analyzed_df.png](analyzed_df.png)
The derived columns of interest are:<br>
* 'date' - the date of the message.<br>
* 'title_body' - combined text of title and body
* 'GME_count' - count how how many times the ticker (GME) was mentioned <br>
* 'title_body_sent' - overall sentiment of title and body<br>
* 'GME_sent' - sentiment of text mentioning ticker (GME) <br>
* 'title_body_sent_sum' - sum of daily sentiment in title and body<br>
* 'GME_sent_sum' - sum of daily sentiment of text mentioning GME<br>

<br>
2.3. add stock data from alpaca<br>
We then read the stock data from Alpaca

![Alpaca_GME_df.png](Alpaca_GME_df.png)
2.4. Data Output
We then create a DataFrame feeding into Machine Learning<br>
![ML_GME_DF.png](ML_GME_DF.png)
all numbers daily 
* 'mentions' - number of mentions of ticker
* 'sentiment' - overall sentiment
* 'ticker_sent' - ticker sentiment
* 'pct_ch' - percent change of ticker price
* 'up_neu_dn' - is stuck up (+1) down (-1) or neutral (0)

### 3 - Run LSTM Scenarios<br>
![LSTM.png](LSTM.png)
3.1 We test the data with an LSTM Model with the following paramters:<br>
* Test/Train Split = 70/30
* model = Sequential()
* number_units = 5
* dropout_fraction = 0.2
* 4 layer with single output layer

To run the LSTM scenarios:<br>
3.2 We first import stock data and create the ML Data as described above<br>
ml_df = fetch_data ('GME', '2021-01-28', '2021-06-28' )<br>
3.3. We then create a feature list and a target list and select a stock from a ticker list<br>
* targ_list = ['pct_ch', 'up_neu_dn']<br>
* feat_list = [cur_tick + count_sufx] + feat_tmplt<br>
3.4. Loop through them to get output with the following fucnction:<br>
cur_loss = run_lstm(ml_df, cur_feat, cur_targ, fname , title)<br>
3.5 tabulate the output:<br> 
![AAPL_Analysis.png](AAPL_Analysis.png)
3.6 As can be seen from charting results, this needs a lot more experimentation:
![AAPL_up_neu_dn_ticker_sentpng.png](AAPL_up_neu_dn_ticker_sentpng.png)
![AAPL_up_neu_dn_sentimentpng.png](AAPL_up_neu_dn_sentimentpng.png)
![AAPL_up_neu_dn_AAPL_count_sumpng.png](AAPL_up_neu_dn_AAPL_count_sumpng.png)
![AAPL_pct_ch_ticker_sentpng.png](AAPL_pct_ch_ticker_sentpng.png)
![AAPL_pct_ch_sentimentpng.png](AAPL_pct_ch_sentimentpng.png)
![AAPL_pct_ch_AAPL_count_sumpng.png](AAPL_pct_ch_AAPL_count_sumpng.png)

### 4 - Evalute other ML Models
(Jimmy)

### 5 - Other
(TBD)
