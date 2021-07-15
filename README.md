![wsb.jpg](./Images/wsb.jpg)![wsb-chart.png](./Images/wsb-chart.png)

# Project2 (grp3): WSbets Sentiment Analysis and stock price prediction
Bootcamp: Columbia Fintech<br>
Cohort: March-Sep 2021<br>

**Creators:**
* Hassan Alam
* Julian Lopez
* Jimmy Unelus
* Ludovic Schneider

## Goal

Determine if it is possible to predict stocks price action by analysing comments in teh "famous" WallStreetBets (WSB) Subreddit group by applying Natural Language Programing (NLP) and Machine leanring models (including Long Short Term Memory -LSTM . The project was more an investigation of possible approaches than a final production produt. The idea was to asses and compare the value of WSB (unconventional data inputs for trading models) Vs traditional prices and finantial indicators.

## Rational

WSB was a forum intially created for retail investor to exchange trades ideas. It became "famous" in early 2021 with the GME story and its short squeeze strategy. WSB has over 10 million subscribers and individual investors would share ideas and joint forces in risky bets like the short squeeze where they pick the stocks that have the most short open position and together start to buy them to "squeeze" the Institutions (often large Hedge Funds) being short the stocks and forced to buy it back.

## Approach / Table of content

We took the follow steps to develop senstivity scenarios (what-if)
- [Project2 (grp3): WSbets Sentiment Analysis and stock price prediction](#project2-grp3-wsbets-sentiment-analysis-and-stock-price-prediction)
  - [Goal](#goal)
  - [Rational](#rational)
  - [Approach / Table of content](#approach--table-of-content)
    - [1 - Read WSB : API](#1---read-wsb--api)
    - [2 - Data Wrangling](#2---data-wrangling)
    - [3 - Common Word Sentiments](#3---common-word-sentiments)
    - [4 - Run LSTM Scenarios](#4---run-lstm-scenarios)
    - [5 - Evalute other ML Models](#5---evalute-other-ml-models)
    - [References:](#references)


### 1 - Read WSB : API

As a user, the first thing you need to do is to create a Reddit account and join/request access to the subreddit that you want to analyse. In our case it is WSBest. Note thic group is now private and hence it might take several days before you get approved. 

1.1. First API: Retrieving the data via Reddit API directly

It is not the best way to retrieve large dataset but it is a good way to stream/retrieve data in real time. It is also one of the only solution to access all the updates of each post. So if you need/want to keep all the information about a post (comments - votes - score...) you probably want to use this methodology.

The constraints of this API is that you are limited to 100 posts per requests. Hence we had to create a loop to allow the user to go back in time as much as desired by batches of 100 posts. This limitation makes it harder and a lot longer to build a large enough dataset to feed our machine learning experimentation.

Therefore we decided to use a different methodology/API to build our dataset. However we still wanted to give the user the flexibility to actually use this API especially if the user wants to retrieve the posts AND the comments/votes...happening afterwards. The user can also adjust the code to create a "live" feed of the last 100 WSBets posts.

We Created a function called: **reddit_direct_api(subreddit, max_batch, limit_posts)**
To run the function the user needs Reddit API keys and  Reddit account password saved in an .env
format : 
    client_id = os.getenv("REDDIT_CLIENT_ID")
    reddit_secret_key = os.getenv("REDDIT_SECRET_KEY")
    reddit_pw = os.getenv('REDDIT_PW')
To create your keys you need to create an account a "program" via the link https://www.reddit.com/prefs/apps.

![Reddit_API_result.png](./Images/PSAW_result.png)
1.2. Second API: Retrieving the data via the Pushift Databse

For this API, we used the wraper PSAW: Python Pushshift.io API Wrapper

Pushift is a database built for users looking to extract a large amount of data - posts. The database is updated realtime as soon as a new post is submitted to Reddit. However this is a one time transfer and hence this databse is missing all the "events" happening after its initial submission. Therefore PSAW is missing the score information, comments ...
We still decided to go with this API as it allowed us to build a very large amount of data relatively quickly. The file we created and used for the future analysis contains 200k rows.

We created the function: **pushiftapi (subreddit, start_year, start_month, start_date, end_year, end_month, end_date, max_posts)**

![PSAW_result.png](./Images/PSAW_result.png)

### 2 - Data Wrangling

2.1. read WSB Data<br>
The first step is to read the WSB into a dataframe
![WSB_DF.png](./Images/WSB_DF.png)
2.2. Add sentiment Data <br>
We then add sentiment data <br>
![analyzed_df.png](./Images/analyzed_df.png)
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

![Alpaca_GME_df.png](./Images/Alpaca_GME_df.png)
2.4. Data Output
We then create a DataFrame feeding into Machine Learning<br>
![ML_GME_DF.png](./Images/ML_GME_DF.png)
all numbers daily 
* 'mentions' - number of mentions of ticker
* 'sentiment' - overall sentiment
* 'ticker_sent' - ticker sentiment
* 'pct_ch' - percent change of ticker price
* 'up_neu_dn' - is stuck up (+1) down (-1) or neutral (0)

### 3 - Common Word Sentiments

3.1 Get Sentiment Value for title and body post<br>
We separate the title and body column into their own dataframe, drop na, do stemming, clean the text, obtain the sentiment values using nltk SentimentIntensityAnalyzer, and then we plot them<br>
![plot_setiments_for_reddit_title_and_body_posts.PNG](./Images/plot_setiments_for_reddit_title_and_body_posts.PNG)
3.2 Generate Wordlouds<br>
We generate wordclouds for both title and body, also wordclouds for each sentiment (negative, positive, and neutral)<br>
![Token_Visualization_of_Common_Words_Among_Post_Titles.PNG](./Images/Token_Visualization_of_Common_Words_Among_Post_Titles.PNG)![wsb_common_words_among_Bodies_post.PNG](./Images/wsb_common_words_among_Bodies_post.PNG)
The Positive Words
![wsb_common_positive_titles_words.PNG](./Images/wsb_common_positive_titles_words.PNG)![wsb_common_positive_bodies_words.PNG](./Images/wsb_common_positive_bodies_words.PNG)
The Negative Words
![wsb_common_negative_titles_words.PNG](./Images/wsb_common_negative_titles_words.PNG)![wsb_common_negative_bodies_words.PNG](./Images/wsb_common_negative_bodies_words.PNG)

### 4 - Run LSTM Scenarios

4.1 We test the data with an LSTM Model with the following paramters:<br>
* Test/Train Split = 70/30
* model = Sequential()
* number_units = 5
* dropout_fraction = 0.2
* 4 layer with single output layer

To run the LSTM scenarios:
4.2 We first import stock data and create the ML Data as described above<br>
ml_df = fetch_data ('GME', '2021-01-28', '2021-06-28' )<br>
4.3. We then create a feature list and a target list and select a stock from a ticker list<br>
* targ_list = ['pct_ch', 'up_neu_dn']<br>
* feat_list = [cur_tick + count_sufx] + feat_tmplt<br>
4.4. Loop through them to get output with the following fucnction:<br>
cur_loss = run_lstm(ml_df, cur_feat, cur_targ, fname , title)<br>
4.5 tabulate the output:<br> 
![AAPL_Analysis.png](./Images/AAPL_Analysis.png)
3.6 As can be seen from charting results, this needs a lot more experimentation:
![AAPL_up_neu_dn_ticker_sentpng.png](./Images/AAPL_up_neu_dn_ticker_sentpng.png)
![AAPL_up_neu_dn_sentimentpng.png](./Images/AAPL_up_neu_dn_sentimentpng.png)
![AAPL_up_neu_dn_AAPL_count_sumpng.png](./Images/AAPL_up_neu_dn_AAPL_count_sumpng.png)
![AAPL_pct_ch_ticker_sentpng.png](./Images/AAPL_pct_ch_ticker_sentpng.png)
![AAPL_pct_ch_sentimentpng.png](./Images/AAPL_pct_ch_sentimentpng.png)
![AAPL_pct_ch_AAPL_count_sumpng.png](./Images/AAPL_pct_ch_AAPL_count_sumpng.png)

### 5 - Evalute other ML Models: Stock Prediction with PyTorch
Stock market prediction is the act of trying to determine the future value of a company stock. The successful prediction of a stock’s future price could yield a significant profit, and this topic is within the scope of time series problems. Among the several ways developed over the years to accurately predict the complex and volatile variation of stock prices, neural networks, more specifically RNNs (Recurrent Neural Network), have shown significant application on the field. Here we are going to use the RNN model LSTM (Long short-term memory) with PyTorch to predict AMC stock market price 

We are going to predict the Close price of AMC stock, and the following is the data behavior over the years.
![amc_stock_price_chart.png](./Images/amc_stock_price_chart.png)

We slice the data frame to get the column we want and normalize the data. Then split the data into train and test sets. Before doing so, we must define the window width of the analysis. The use of prior time steps to predict the next time step is called the sliding window method.
![split_data_and_setup_train_and_test.png](./Images/split_data_and_setup_train_and_test.png)
The basic structure for building a PyTorch model is transforming them into tensors and defining some common values for both models regarding the layers.
![import_PyTorch.png](./Images/import_PyTorch.png)
An LSTM unit is composed of a cell, an input gate, an output gate, and a forget gate. The cell remembers values over arbitrary time intervals, and the three gates regulate the flow of information into and out of the cell.
![LSTM.png](./Images/LSTM.png)
Finally, we train the model over 100 epochs.
![run_100_epoch_and_display_Data_and_training_prediction.png](./Images/run_100_epoch_and_display_Data_and_training_prediction.png)


### References:

Trending stocks and cryptos on Reddit Wallstreetbets: 
https://trade-tip.com/reddit-sentiment-wallstreetbets.html

https://www.kaggle.com/sprakshith/beginner-s-guide-to-sentiment-analysis

https://www.kaggle.com/thomaskonstantin/reddit-wallstreetbets-posts-sentiment-analysis

Reddit API : https://github.com/reddit-archive/reddit/wiki/API

Pushift API : https://psaw.readthedocs.io/en/latest/ and https://pushshift.io

PyTorch : https://pytorch.org/

Stock Price Prediction with PyTorch: https://medium.com/swlh/stock-price-prediction-with-pytorch-37f52ae84632 