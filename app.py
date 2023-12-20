import streamlit as st
import preprocessor
import helper
import sentiment
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title='WhatSense'
)

st.title("WhatSense")
st.sidebar.title("Whatsapp Chat & Sentiment Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp Chat .txt file (excluding media)")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")

    df = preprocessor.preprocess(data)
    print(df.head())
    # fetch unique users
    user_list = list(df["Author"].unique())
    # user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis for", user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(
            selected_user, df)
        st.header("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Total Texts")
            st.header(num_messages)
        with col2:
            st.subheader("Total Words")
            st.header(words)
        with col3:
            st.subheader("Media Shared")
            st.header(num_media_messages)
        with col4:
            st.subheader("Links Shared")
            st.header(num_links)

        # monthly timeline
        st.header("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig = px.line(timeline, x='time', y="Message",
                      labels={
                          "time": "Month",
                          "Message": "No. of Messages",
                      })
        fig.update_xaxes(tickangle=45, tickfont=dict(
            family='Rockwell', size=14), title_font_family="Arial")
        fig.update_layout({
            'plot_bgcolor': 'lightskyblue',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        fig.update_traces(line_color='darkslateblue')
        fig.update_yaxes(title_font_family="Arial")
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        # daily timeline
        st.header("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)

        fig = px.line(daily_timeline, x='only_date', y="Message",
                      labels={
                          "only_date": "Date",
                          "Message": "No. of Messages",
                      })
        fig.update_xaxes(tickangle=45, tickfont=dict(
            family='Rockwell', size=14),title_font_family="Arial")
        fig.update_yaxes(title_font_family="Arial")
        fig.update_layout({
            'plot_bgcolor': 'lightpink',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        fig.update_traces(line_color='brown')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        # activity map
        st.header('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            bd_df = pd.DataFrame({'Day':busy_day.index,
                                  'Messages':busy_day.values})
            fig = px.bar(bd_df, x='Day', y='Messages', width=450, height=450)
            fig.update_xaxes(tickangle=-45, tickfont=dict(
            family='Rockwell', size=14), title_font_family="Arial")
            fig.update_yaxes(title_font_family="Arial")
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            bm_df = pd.DataFrame({'Month':busy_month.index,
                                  'Messages':busy_month.values})
            fig = px.bar(bm_df, x='Month', y='Messages', height=450, width=450)
            fig.update_xaxes(tickangle=-45, tickfont=dict(
            family='Rockwell', size=14), title_font_family="Arial")
            fig.update_yaxes(title_font_family="Arial")
            st.plotly_chart(fig, theme="streamlit", use_container_width=False)

        st.header("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.header('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                busy_user_df = pd.DataFrame({'User':x.index,
                                  'Messages':x.values})
                fig = px.bar(busy_user_df, x='User', y='Messages', height=550, width=450)
                fig.update_xaxes(tickangle=-45, tickfont=dict(
                family='Rockwell', size=14), title_font_family="Arial")
                fig.update_yaxes(title_font_family="Arial")
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.header("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots(facecolor='k')
        plt.axis("off")
        plt.tight_layout(pad=0)
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user, df)
        most_common_df.rename(columns = {0:'word', 1:'freq'}, inplace = True)
        fig = px.bar(most_common_df, x = 'freq', y='word', orientation='h', height=700,
                     labels={
                          "freq": "Frequency",
                          "word": "Word",
                      })
        fig.update_layout(
            font=dict(
            family="Courier New, monospace",
            size=15)
            )
        
        fig.update_xaxes(title_font_family="Arial")
        fig.update_yaxes(title_font_family="Arial")
        # st.dataframe(most_common_df)
        st.header('Most commmon words')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, use_container_height=True)

        # Sentiment Analysis
        st.header("Sentiment Analysis")

        col1, col2 = None, None

        sent_df = sentiment.sentiment_table(selected_user, df)

        fig = px.pie(sent_df, values='count', names=sent_df.index, color=sent_df.index,
        color_discrete_map={'Positive':'limegreen',
                             'Negative':'firebrick'},
        title='Sentiment Pie Chart', labels={
            "index": "Sentiment",
            "sentiment": "No. of Messages",
        })
        fig.update_layout()
        st.plotly_chart(fig, theme="streamlit",
                        use_container_width=True, width=500)

style = """
<style>
    #MainMenu {
            visibility:hidden;
            }

    footer {
            visibility:visible;
         }
    footer:after {
            content:'and ❤️ by @sycoRax.';
            display:block;
            position:relative;
        }
        </style>
        """
st.markdown(style, unsafe_allow_html=True)
