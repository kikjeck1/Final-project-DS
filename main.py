import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

with st.echo(code_location='below'):
    @st.cache
    def get_data():
        url = "https://raw.githubusercontent.com/pinkunicorn322/Final-project-DS/master/dns_data_smartphones_v3(11.06.22).csv"
        df = pd.read_csv(url, index_col=0)

        return df

    """
    Hello, I came up with two sizable projects.
    The first is about smarthones, its prices and the dynamic of the prices, based on data from https://www.dns-shop.ru.
    The second is about shipping. My goal was to draw a map with main delivery routes. I scrapped data from https://www.cdek.ru/ru.
    Please, check all the my files.
    """

    """
    Firstly, data scrapping was conducted. You can find it in file dns_data.ipynb. So here I only donwnload ready to use data.
    """



    min_price, max_price = st.select_slider(
        'Select a range of price',
        options=range(0, 100000),
        value=(0, 20000))

    st.write(f'The following smarthones lay between {min_price} and {max_price} rubles.')

    df = get_data()

    st.write(df[(min_price<= df.prices) & (df.prices<= max_price)])

    dns_data = df
    X = dns_data.drop(["prices", "smartphones", "links", "resol"], axis=1)
    y = dns_data.prices

    """
    Let's try to do some simple ML. Here our features.
    """
    st.write(X.head(3))

    # для работы с категориальными признаками
    X = pd.get_dummies(X)
    """
    I used also special tecnique to work with categorial variables. You can find it in the other file, but now it is not neccesary.
    """
    X.head(3)

    rs = 10

    options = st.multiselect(
        'What features do you prefer to use in linear regression?',
        ["diag", "memory", "oper", "bat"],
        ["diag", "memory", "oper", "bat"])

    cur_X = X.loc[:, options]
    X_train, X_test, y_train, y_test = train_test_split(cur_X, y,
                                                        test_size=0.4,
                                                        random_state=rs)

    regr = LinearRegression()
    regr.fit(X_train, y_train)
    # st.write(regr.coef_)
    st.write("Here the coefficients of fitted regression:", pd.DataFrame(data=[cur_X.columns, regr.coef_]).T)
    y_pred = regr.predict(X_test)
    st.write("Here r2 score - indicator of quality of our model.",skl.metrics.r2_score(y_true=y_test, y_pred=y_pred))
    """
    It is expected that it is not big enough. 
    """

    """
    You can see the one feature liner regression. It's difficult to visualize 4 features LR:(.
    """
    fig, ax = plt.subplots()
    # sns.set(rc={'axes.facecolor': 'cornflowerblue', 'figure.facecolor': 'cornflowerblue'})
    sns.set(style="ticks", context="talk")
    plt.style.use("dark_background")

    option = st.selectbox(
        'What feature do you prefer for making and regression and visualizing it?',
        ["diag", "memory", "oper", "bat"])

    sns.regplot(x=X[option], y=y, ax=ax, scatter_kws={'s':1, "alpha":0.5})
    st.pyplot(fig)

    """
    In one picture
    """
    fig, ax = plt.subplots(2, 2, figsize=(14, 14))
    sns.regplot(x=X.diag, y=y, ax=ax[0, 0], color="b", scatter_kws={'s':1, "alpha":0.5})
    sns.regplot(x=X.memory, y=y, ax=ax[0, 1], color="r", scatter_kws={'s':1, "alpha":0.5})
    sns.regplot(x=X.oper, y=y, ax=ax[1, 0], color="teal", scatter_kws={'s':1, "alpha":0.5})
    sns.regplot(x=X.bat, y=y, ax=ax[1, 1], scatter_kws={'s':1, "alpha":0.5})
    st.pyplot(fig)

    """
    Attention. In corresponding .ipynb file you find slightly more advanced LR, including usage companies as features.
    """

