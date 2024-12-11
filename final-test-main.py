import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import datetime as dt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import base64
import time

st.set_page_config(layout="wide", page_title="", initial_sidebar_state="expanded")

#global font
def set_global_font(font_name='Arial', font_size='16px'):
    st.markdown(
        f"""
        <style>
        body, p, div, span, a, button {{
            font-family: '{font_name}', sans-serif !important;
            font-size: {font_size} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
set_global_font('Montserrat', '16px')

# streamlit bg image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode()
    st.markdown(
         f"""
        <style>
        .stApp {{
            background-image: url("data:image/webp;base64,{base64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .main-title, h1, h2, h3, h4, h5, h6, p {{
            color: white ;
        }}
        

        """,
        unsafe_allow_html=True
    )

# image input
pic = 'supplystocks.jpg'
set_background(pic)

# load the data
@st.cache_data
def load_data():
    file_path = r'https://raw.githubusercontent.com/andrew-jxhn/blank-app/main/DataCoSupplyChainDataset.csv'
    df = pd.read_csv(file_path, encoding='latin1')
    df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
    df['shipping date (DateOrders)'] = pd.to_datetime(df['shipping date (DateOrders)'])
    return df

df = load_data()
st.title('DataCo Smart Supply Chain Analytics Dashboard')

# streamlit app
# sidebar
col1, col2, col3 = st.sidebar.columns([1,3,1])
with col2:
    st.image("world.gif", width=200)
    
st.sidebar.header("[AJ's GitHub Repo](https://github.com/andrew-jxhn/CMSE830_FDSProject)")

tabs = st.sidebar.radio("**Professional of ↓**", ["Business", "Technology"])
st.sidebar.markdown("---")

st.sidebar.markdown("**Navigate to ↓**")

if tabs == "Business":
    selected_page = st.sidebar.selectbox("", ["Overview", "User Guide, Documentation & References", "Customer Segmentation Analysis", "Interactable EDA (β-version)", "Modelling Results"])

    if selected_page == "Overview":
        st.header("Overview")
        st.markdown("---") 
        st.header("வணக்கம்! (Vanakkam!) - Hello There! 😊🏭🚚🏗️")
        st.subheader('Project Overview')
        st.write("""
            This project explores the **DataCo Smart Supply Chain** dataset to gain insights into the supply chain performance, focusing on key metrics such as sales, profit, and delivery time.
            
            **IoT and Data Growth**: The widespread adoption of IoT has generated vast amounts of data, valuable for uncovering insights and enhancing decision-making.
            
            **Customer Segmentation**: This project utilizes the DataCo dataset to conduct customer segmentation, enabling better customer understanding and revenue growth.
            
            **Model Selection Challenge**: With multiple data analysis methods and models available, choosing the right one is crucial as model performance varies with data parameters.
            
            **Comparative Studies**: Prior studies (e.g., Carbonneau, Hill, Vakili, Ahmed cited in references) have compared traditional forecasting methods with neural networks and machine learning models, revealing varying levels of performance across techniques.
            
            ***Project Objective***: This study plans to compare 10 machine learning classifiers and 7 regression models against neural networks (a classifer NN and a regressor NN). Key tasks include fraud detection, late delivery prediction, sales forecasting, and demand prediction.
            
            **Model Performance**:
            - *Classification models*: Logistic Regression, Gaussian Naive Bayes, SVM, k-NN, Linear Discriminant, Random Forest, Extra Trees, eXtreme Gradient Boosting, Decision Trees, Bagging models are evaluated for accuracy, recall, and F1 score for fraud and late delivery prediction.
            - *Regression models*: Lasso, Ridge, Light Gradient, Random Forest, eXtreme Gradient Boosting, Decision Tree, Linear Regression models are assessed with MAE and RMSE for sales and demand prediction.
        """)
        st.markdown("---") 
        st.write("***Note:*** If you are a Business User - please head over to Business Side Bar, and if you are a Technology User looking for technical implementation, please head over to Technology Side Bar.")
        st.markdown("---") 
        st.write("The original data is sourced from - [DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS](https://data.mendeley.com/datasets/8gx2fvg2k6/5)")
    
    elif selected_page == "User Guide, Documentation & References":
        st.header("User Guide, Documentation & References")   
        st.subheader("User Guide for ease of use of the dashboard -")
        st.write("""
                1. Feel free to start from the IDA to get the overall feel of the data, which gives important metrics such as summary, missingness, outliers etc.

                2. Next head over to EDA to see the analysis such as 
                    - Correlation Heatmap where features such as Product Price have high correlation Sales.
                    - Total Sales by Market
                    - and more...
                
                3. Now is the fun part, the Interactable EDA, where you can construct your own Univariate and Bivariate Analysis with many plot types to choose from. See what's the realtionship between all the 53 variables that are available (still in β-version).
                
                4. Moving on to the Missingness Analysis, given the almost-perfect nature of the dataset, we induce missingness. Once we induce missingness, we see how data gets imputed back into the dataset and we have imputation options as well. This allows to see how the data behaves with imputed data.
                
                5. Customer Segmentation Analysis is a supply chain management concept. Customers are divided into different groups based on specific criteria to tailor supply chain strategies and resource allocation to better meet their individual needs, 
                    thereby optimizing efficiency and customer satisfaction within the supply chain. The analysis is available under "Customer Segmentation Analysis" tab.

                6. Moving on to Modelling Metrics and Modelling Results, around 10 classification models, 1 classifier neural network, 7 regression models, and 1 regressor neural network are assessed, as per the objective of this dashboard.
                    The code for Modelling can be found here - [Assessment of ML and NN models for Supply Chain -TBD](tbd)
                
                7. If there are any questions, issues or concerns regarding the Dashboard, please don't hesitate to contact the [dev](mailto:johnprak@msu.edu).😊
                
                """)
        st.markdown("---") 
        st.subheader("Description of Models -")
        st.write("""
                ### Classification Models
                - **[Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)**: A statistical model for binary classification. It estimates probabilities using a logistic function. Simple, interpretable, and supports numerical and categorical features.
                - **[Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)**: A probabilistic classifier based on Bayes' theorem, assuming feature independence and normal distribution. Fast and effective for tasks like text classification and spam detection.
                - **[Linear SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)**: Finds the optimal hyperplane to separate classes. Suitable for high-dimensional data and complex decision boundaries in tasks like text categorization and image recognition.
                - **[K-Nearest Neighbors (KNN)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)**: An instance-based learning algorithm that classifies based on the class of k nearest neighbors. Versatile for linear and non-linear decision boundaries.
                - **[Linear Discriminant Analysis (LDA)](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)**: Combines features for maximum class separation. Works well for Gaussian-distributed data in applications like face recognition.
                - **[Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)**: An ensemble of decision trees trained on random subsets. Handles numerical and categorical features well and is robust against overfitting.
                - **[Extra Trees Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)**: A variant of Random Forest with additional randomness in split selection, leading to faster training and improved performance on high-dimensional data.
                - **[XGBoost](https://xgboost.readthedocs.io/en/stable/)**: A fast and scalable gradient boosting algorithm. Known for speed, robustness, and versatility across various tasks.
                - **[Decision Trees](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)**: Intuitive model using tree structures for predictions. Handles non-linear relationships and both numerical and categorical data effectively.
                - **[Bagging](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)**: An ensemble method that combines predictions from multiple weak learners (e.g., decision trees) to improve stability and accuracy.

                ### Regression Models
                - **[Lasso Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)**: Adds a penalty to coefficients for feature selection, ideal for high-dimensional data.
                - **[Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)**: Shrinks coefficients to address multicollinearity, reducing overfitting in models with many predictors.
                - **[LightGBM](https://lightgbm.readthedocs.io/en/latest/)**: A gradient boosting framework optimized for large-scale and high-dimensional data. Efficient and scalable.
                - **[Random Forest Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)**: An ensemble method averaging predictions from multiple decision trees to improve accuracy and reduce overfitting.
                - **[XGBoost Regression](https://xgboost.readthedocs.io/en/stable/)**: Combines weak models (e.g., decision trees) for strong predictive performance. Efficient and robust for diverse tasks.
                - **[Decision Tree Regression](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)**: Models data in a tree structure, finding splits that minimize prediction error. Simple and interpretable.
                - **[Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)**: Models the linear relationship between independent variables and a continuous target variable. Widely used for predicting continuous outcomes.
                """)


        st.markdown("---")
        st.subheader("Documentation and References -")
        st.write("""

        - Constante, Fabian; Silva, Fernando; Pereira, António (2019), “DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS”, Mendeley Data, V5, doi: 10.17632/8gx2fvg2k6.5
        - [GIF Image Credits](https://www.google.com/url?sa=i&url=https%3A%2F%2Fgifer.com%2Fen%2Fgifs%2Fwhite&psig=AOvVaw0Exqh5wzZ_glkkghB0M8QB&ust=1733268043013000&source=images&cd=vfe&opi=89978449&ved=0CBMQjRxqFwoTCIDMmPSciooDFQAAAAAdAAAAABAE)
        - Background Image Credits - https://pinkerton.com/programs/supply-chain-risk-management
        - Jupyter to Streamlit - https://github.com/ddobrinskiy/streamlit-jupyter
        - Ahmed, N. K., Atiya, A. F., Gayar, N. E., & El-Shishiny, H. (2010). An Empirical Comparison of Machine Learning Models for Time Series Forecasting. Econometric Reviews, 29(5-6), 594–621.
        - Carbonneau, R., Laframboise, K., & Vahidov, R. (2008). Application of machine learning techniques for supply chain demand forecasting. European Journal of Operational Research, 184(3), 1140–1154.
        - Ferreira, K.J., Lee, B.H.A., & Simchi-Levi, D. (2016). Analytics for an online retailer: Demand forecasting and price optimization. Manufacturing & Service Operations Management, 18(1), 69–88.
        - Martinez, A., Schmuck, C., Pereverzyev Jr, S., Pirker, C., & Haltmeier, M. (2020). A machine learning framework for customer purchase prediction in the non-contractual setting. European Journal of Operational Research, 281(3), 588–596.
        - Hassan, C.A., Khan, M.S., & Shah, M.A. (2018). Comparison of Machine Learning Algorithms in Data Classification. 24th International Conference on Automation and Computing (ICAC), Newcastle upon Tyne, United Kingdom, pp. 1–6.
        - Vakili, M., Ghamsari, M., & Rezaei, M. (2020). Performance Analysis and Comparison of Machine and Deep Learning Algorithms for IoT Data Classification. arXiv preprint, arXiv:2001.09636.
        - Constante, Fabian; Silva, Fernando; Pereira, António (2019). DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS. Mendeley Data, v5. Retrieved 25 March 2020, from http://dx.doi.org/10.17632/8gx2fvg2k6.5.
        - Building Neural Network Using Keras for Classification. (2020). Retrieved 18 April 2020, from https://medium.com/datadriveninvestor/building-neural-network-using-keras-for-classification-3a3656c726c1.
        - Explaining Feature Importance by Example of a Random Forest. (2020). Retrieved 15 April 2020, from https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e.
        - Find Your Best Customers with Customer Segmentation in Python. (2020). Retrieved 9 April 2020, from https://towardsdatascience.com/find-your-best-customers-with-customer-segmentation-in-python-61d602f9eee6.
        - Resampling Time Series Data with Pandas Ben Alex Keen. (2020). Retrieved 7 April 2020, from https://benalexkeen.com/resampling-time-series-data-with-pandas/.
        - Python matplotlib multiple bars. Bars, P., Smith, J., & Lyon, J. (2020). Retrieved 17 April 2020, from https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars.
        - sklearn.linear_model.LinearRegression — Scikit-learn Documentation. (2020). Retrieved 14 April 2020, from https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html.
        - RFM Segmentation | RFM Analysis, Model, Marketing & Software | Optimove. (2020). Retrieved 10 April 2020, from https://www.optimove.com/resources/learning-center/rfm-segmentation.
        - How to Label the Feature Importance with Forests of Trees? Trees?, H., & Dixit, H. (2020). Retrieved 10 April 2020, from https://stackoverflow.com/questions/37877542/how-to-label-the-feature-importance-with-forests-of-trees.
        """)
    
    elif selected_page == "Customer Segmentation Analysis":
        st.header("Customer Segmentation Analysis")
        st.header("Customer Segmentation Analysis using RFM Analysis")
        st.write("""Customer segmentation using RFM analysis is a widely adopted approach to understand and categorize customers based on their purchasing behavior. 
        By analyzing the recency, frequency, and monetary value of customer transactions, businesses can identify different customer segments 
        and develop targeted strategies to retain, engage, and acquire customers more effectively. This data-driven approach helps companies make informed decisions 
        to optimize their marketing efforts, improve customer loyalty, and ultimately enhance their overall business performance.""")
        st.write("""Targeting specific customer segments can help a supply chain company increase its customer base and profits. With purchase history available, 
        RFM analysis can be used for segmentation. It's favored for its numerical representation of customer recency, frequency, and monetary value, resulting in easily interpretable outcomes.""")
        st.markdown("---") 

        # Calculating total price for each order
        st.subheader("1. Data Preparation - Getting the Date of Max cost of Order and Total Price")
        df['TotalPrice'] = df['Order Item Quantity'] * df['Order Item Total']
        st.write("Date of Max cost of Order", df['order date (DateOrders)'].max())
        st.write("""The last order in the dataset was made on 2018-01-31 23:38:00. For the project, hypothetically, let us travel back in time to be in 'present time' and set 
        it slightly above than the last order time for more accuracy of recency value.""")

        # Determine the present date
        present = dt.datetime(2018,2,1)
        df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
        st.markdown("---") 

        # RFM Analysis
        st.subheader("2. RFM Analysis Calculation")
        st.write("Grouping data by Customer ID and calculating Recency, Frequency, and Monetary Value")
        Customer_seg = df.groupby('Order Customer Id').agg({
            'order date (DateOrders)': lambda x: (present - x.max()).days,
            'Order Id': lambda x: len(x),
            'TotalPrice': lambda x: x.sum()
        })

        # Rename columns
        Customer_seg.rename(columns={
            'order date (DateOrders)': 'R_Value', 
            'Order Id': 'F_Value', 
            'TotalPrice': 'M_Value'
        }, inplace=True)

        st.write("RFM Values Overview:")
        st.dataframe(Customer_seg.head())

        st.write("""
            - R_Value(Recency) indicates how much time elapsed since a customer last order, in hours.
            - F_Value(Frequency) indicates how many times a customer ordered.
            - M_Value(Monetary value) tells us how much a customer has spent purchasing items.
                """)
        st.markdown("---") 

        # Visualization of RFM Distributions
        st.subheader("3. RFM Value Distributions")
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        sns.histplot(Customer_seg['R_Value'], kde=True, color='skyblue', ax=axes[0])
        axes[0].set_title('Distribution of R_Value (Recency)', fontsize=16)
        axes[0].set_xlabel('R_Value', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)

        sns.histplot(Customer_seg['F_Value'], kde=True, color='salmon', ax=axes[1])
        axes[1].set_title('Distribution of F_Value (Frequency)', fontsize=16)
        axes[1].set_xlabel('F_Value', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)

        sns.histplot(Customer_seg['M_Value'], kde=True, color='lightgreen', ax=axes[2])
        axes[2].set_title('Distribution of M_Value (Monetary Value)', fontsize=16)
        axes[2].set_xlabel('M_Value', fontsize=12)
        axes[2].set_ylabel('Frequency', fontsize=12)

        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("---") 

        # Quantile Calculation
        st.subheader("4. Quantile Calculation")
        quantiles = Customer_seg.quantile(q=[0.25,0.5,0.75])
        quantiles = quantiles.to_dict()
        st.write("Quantiles for R, F, M Values:")
        st.write(quantiles)

        st.write("""The data is divided into four quantiles. A low R_Value reflects recent customer activity, 
        while high F_Value and M_Value indicate frequency and total purchase value. A function is defined to represent quantiles as numerical values.""")

        # Scoring Functions
        def R_Score(a,b,c):
            if a <= c[b][0.25]:
                return 1
            elif a <= c[b][0.50]:
                return 2
            elif a <= c[b][0.75]: 
                return 3
            else:
                return 4

        def FM_Score(x,y,z):
            if x <= z[y][0.25]:
                return 4
            elif x <= z[y][0.50]:
                return 3
            elif x <= z[y][0.75]: 
                return 2
            else:
                return 1
        st.markdown("---") 

        # Calculate Scores
        st.subheader("5. Calculating RFM Scores & Total RFM Score")
        Customer_seg['R_Score'] = Customer_seg['R_Value'].apply(R_Score, args=('R_Value',quantiles))
        Customer_seg['F_Score'] = Customer_seg['F_Value'].apply(FM_Score, args=('F_Value',quantiles))
        Customer_seg['M_Score'] = Customer_seg['M_Value'].apply(FM_Score, args=('M_Value',quantiles))

        # Combined RFM Score
        Customer_seg['RFM_Score'] = Customer_seg.R_Score.astype(str) + Customer_seg.F_Score.astype(str) + Customer_seg.M_Score.astype(str)

        st.write("RFM Scores Overview:")
        st.dataframe(Customer_seg.head())
        Customer_seg['RFM_Total_Score'] = Customer_seg[['R_Score','F_Score','M_Score']].sum(axis=1)
        st.write("Individual R, F, and M scores are calculated, and a new column for the combined RFM score is created.")
        
        # Customer Segmentation
        def RFM_Total_Score(df):
            if (df['RFM_Total_Score'] >= 11):
                return 'Champion Customers' 
            elif (df['RFM_Total_Score'] == 10):
                return 'Loyal Customers' 
            elif (df['RFM_Total_Score'] == 9):
                return 'Recent Customers'
            elif (df['RFM_Total_Score'] == 8):
                return 'Promising Customers'
            elif (df['RFM_Total_Score'] == 7):
                return 'Sporadic Customers'
            elif (df['RFM_Total_Score'] == 6):
                return 'Customers Needing Attention'
            elif (df['RFM_Total_Score'] == 5):
                return 'At Risk Customers'
            else:
                return 'Lost'

        Customer_seg['Customer_Segmentation'] = Customer_seg.apply(RFM_Total_Score, axis=1)

        count=Customer_seg['RFM_Score'].unique()
        print(count)# Printing all Unique values
        len(count)# Total count

        # Calculate RFM_Score
        Customer_seg['RFM_Total_Score'] = Customer_seg[['R_Score','F_Score','M_Score']].sum(axis=1)
        Customer_seg['RFM_Total_Score'].unique()
        st.markdown("---") 

        # Visualization of Customer Segments
        st.subheader("6. Customer Segmentation Distribution")
        st.write("""The total number of different customer segments can be determined using the .unique() method along with len. 
                    There are""", len(count), """distinct customer segments. To simplify segmentation, individual R, F, and M scores are combined.
                    There are""", len(Customer_seg['RFM_Total_Score'].unique()), """which are""", sorted(Customer_seg['RFM_Total_Score'].unique()), 
                    """values in total for customer segmentation. Appropriate names were assigned for each value seperately.
                    *P.S. - Category 11 and 12 are grouped together, so there will only be 8 pieces of pie.*
                """)
        fig, ax = plt.subplots(figsize=(10, 10))
        Customer_seg['Customer_Segmentation'].value_counts().plot.pie(
            ax=ax,
            startangle=135,
            explode=(0, 0, 0, 0.1, 0, 0, 0, 0),
            autopct='%.1f%%',
            shadow=True,
            colors=sns.color_palette("pastel")
        )
        ax.set_title("Customer Segmentation", fontsize=15, fontweight='bold')
        ax.set_ylabel("")
        st.pyplot(fig)
        st.write("""With a total of 9 customer segments, 11.4% are at risk of churn, while 11% require attention to prevent potential loss. 
        Additionally, 4.4% of customers have already churned. Here are our top 10 churned customers who have not made a purchase in a while.""")
        st.markdown("---") 

        # Detailed Insights
        st.subheader("7. Detailed Customer Insights")

        # Churned Customers
        st.write("Top 10 Churned Customers:")
        churned = Customer_seg[(Customer_seg['RFM_Score']=='411')].sort_values('M_Value', ascending=False).head(10)
        st.dataframe(churned)

        st.write("""These customers previously ordered frequently and in large amounts, but they haven't made a purchase in nearly a year, 
        indicating they're buying from competitors. This group should be targeted with offers to win them back. Meanwhile, Top 10 new best customers who place costly orders often.""")

        # Best Potential Customers
        st.write("Top 10 Best Potential Customers:")
        best_customers = Customer_seg[(Customer_seg['RFM_Score']=='144')|(Customer_seg['RFM_Score']=='143')].sort_values('M_Value', ascending=False).head(10)
        st.dataframe(best_customers)

        st.write("""The customers mentioned above have the potential to become loyal clients and should be targeted to foster their loyalty. Each customer segment should receive tailored advertisements and rewards to boost profits and enhance responsiveness.""")
        st.markdown("---") 

    elif selected_page == "Interactable EDA (β-version)":
        st.header("Customized EDA (β-version)")
        analysis_type = st.sidebar.selectbox('Select Analysis Type', 
                                            ['Univariate', 'Bivariate', 'Multivariate'])

        if analysis_type == 'Univariate':
            st.subheader('Univariate Analysis')
            column = st.selectbox('Select a column for analysis', df.columns)
            plot_type = st.selectbox('Select plot type', ['Box', 'Bar', 'Line', 'Histogram', 'Distribution'])
            
            st.write("""
            Univariate analysis examines one variable at a time. It's useful for understanding the distribution, central tendency, and spread of individual variables.
            """)
            
            if df[column].dtype in ['int64', 'float64']:
                if plot_type == 'Box':
                    fig = px.box(df, y=column)
                elif plot_type == 'Bar':
                    fig = px.bar(df[column].value_counts())
                elif plot_type == 'Line':
                    fig = px.line(df[column].sort_values())
                elif plot_type == 'Histogram':
                    fig = px.histogram(df, x=column)
                else:  # Distribution
                    fig = px.histogram(df, x=column, marginal='box')
            else:
                fig = px.bar(df[column].value_counts())
            
            st.plotly_chart(fig)

        elif analysis_type == 'Bivariate':
            st.subheader('Bivariate Analysis')
            col1 = st.selectbox('Select first variable', df.columns)
            col2 = st.selectbox('Select second variable', df.columns)
            
            st.write("""
            Bivariate analysis examines the relationship between two variables. It's useful for identifying correlations and patterns between pairs of variables.
            """)
            
            if df[col1].dtype in ['int64', 'float64'] and df[col2].dtype in ['int64', 'float64']:
                fig = px.scatter(df, x=col1, y=col2)
            else:
                fig = px.box(df, x=col1, y=col2)
            st.plotly_chart(fig)

        elif analysis_type == 'Multivariate':
            st.subheader('Multivariate Analysis')
            st.write("""
            Multivariate analysis examines relationships among multiple variables simultaneously. It's useful for uncovering complex patterns and interactions in the data.
            """)
            
            st.write("Pair plot of key numeric variables")
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # Limit to 5 for performance
            fig = sns.pairplot(df[numeric_cols])
            st.pyplot(fig)
            
            st.write("Parallel Coordinates Plot")
            categorical_cols = df.select_dtypes(include=['object']).columns[:3]  # Select top 3 categorical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]  # Select top 3 numeric columns
            selected_cols = list(categorical_cols) + list(numeric_cols)
            
            # Create a copy of the dataframe with encoded categorical variables
            df_encoded = df.copy()
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            
            fig = px.parallel_coordinates(df_encoded, dimensions=selected_cols, color=categorical_cols[0])
            st.plotly_chart(fig)

        
        elif selected_page == "Modelling Results":
            st.header("Modelling Results")
            st.write("This is the Modelling Results tab.")

    elif selected_page == "Modelling Results":
        st.header("Modelling Results")

        st.subheader("Results Analysis")

        # Classification Comparison
        classification_data = {
            'Classification Model': ['Logistic', 'Gaussian Naive Bayes', 'Support Vector Machines', 'K-nearest Neighbour', 'Linear Discriminant Analysis', 'Random Forest', 'Extra Trees', 'eXtreme Gradient Boosting', 'Decision Tree'],
            'Accuracy Score for Fraud Detection': [97.79, 87.85, 97.73, 97.18, 97.88, 98.67, 98.62, 98.91, 99.06],
            'Recall Score for Fraud Detection': [57.89, 16.23, 55.59, 36.61, 56.68, 97.74, 98.61, 91.02, 81.48],
            'F1 Score for Fraud Detection': [31.88, 27.93, 27.99, 31.20, 48.86, 61.29, 58.79, 72.07, 79.52],
            'Accuracy Score for Late Delivery': [98.85, 57.27, 98.85, 81.63, 97.91, 98.85, 99.03, 99.13, 99.25],
            'Recall Score for Late Delivery': [97.94, 56.20, 97.94, 83.95, 97.71, 97.94, 98.27, 98.45, 99.37],
            'F1 Score for Late Delivery': [98.96, 71.96, 98.96, 83.07, 98.10, 98.96, 99.12, 99.21, 99.32]
        }

        classification_comparision = pd.DataFrame(classification_data, columns=[
            'Classification Model', 'Accuracy Score for Fraud Detection', 'Recall Score for Fraud Detection', 'F1 Score for Fraud Detection',
            'Accuracy Score for Late Delivery', 'Recall Score for Late Delivery', 'F1 Score for Late Delivery'
        ])

        max_classification_values = classification_comparision.max()

        # Regression Comparison
        regression_data = {
            'Regression Model': ['Lasso', 'Ridge', 'Light Gradient Boosting', 'Random Forest', 'eXtreme Gradient Boosting', 'Decision Tree', 'Linear Regression'],
            'MAE Value for Sales': [1.33, 0.26, 0.55, 0.20, 0.154, 0.01, 0.0005],
            'RMSE Value for Sales': [2.09, 0.47, 4.34, 1.95, 0.54, 0.78, 0.0014],
            'MAE Value for Quantity': [1.25, 0.34, 0.0004, 0.00007, 0.0005, 0.00, 0.33],
            'RMSE Value for Quantity': [1.43, 0.52, 0.004, 0.006, 0.001, 0.00, 0.52]
        }

        regression_comparision = pd.DataFrame(
            regression_data, 
            columns=['Regression Model', 'MAE Value for Sales', 'RMSE Value for Sales', 'MAE Value for Quantity', 'RMSE Value for Quantity']
        )

        min_regression_values = regression_comparision.min()

        # Create two radio buttons for Classifier and Regressor
        classifier_or_regressor = st.radio("Select Model Type:", ["Classifiers", "Regressors"])
        st.markdown("---") 
        if classifier_or_regressor == "Classifiers":
            st.subheader("Classification Comparison")
            st.write(classification_comparision)
            st.write("")

            # Display the summary paragraph adjacent to the max values
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"* The maximum ***Accuracy Score for Fraud Detection*** is ***{max_classification_values['Accuracy Score for Fraud Detection']:.2f}***, achieved by the ***Decision Tree model***.")
                st.write(f"* The maximum ***Recall Score for Fraud Detection*** is ***{max_classification_values['Recall Score for Fraud Detection']:.2f}***, achieved by the ***Extra Trees model***.")
                st.write(f"* The maximum ***F1 Score for Fraud Detection*** is ***{max_classification_values['F1 Score for Fraud Detection']:.2f}***, achieved by the ***Decision Tree model***.")
                st.write(f"* The maximum ***Accuracy Score for Late Delivery*** is ***{max_classification_values['Accuracy Score for Late Delivery']:.2f}***, achieved by the ***Decision Tree model***.")
                st.write(f"* The maximum ***Recall Score for Late Delivery*** is ***{max_classification_values['Recall Score for Late Delivery']:.2f}***, achieved by the ***Decision Tree model***.")
                st.write(f"* The maximum ***F1 Score for Late Delivery*** is ***{max_classification_values['F1 Score for Late Delivery']:.2f}***, achieved by the ***Decision Tree model***.")
            with col2:
                st.write("""Based on the F1 score, it is evident that the Decision Tree classifier outperforms other models for classification tasks, achieving nearly 79.52% for fraud detection and 99.32% for late delivery. 
                To ensure the model's predictions are reliable, it has been cross-validated, and the results were compared against the model's accuracy.""")
                st.write("""The f1 score for neural network model is 96.48% which is pretty high and better when compared with decision tree f1 score which was 79.52%. 
                But comparing accuracy scores it can concluded that even machine learning models did pretty good for fraud detection and late delivery prediction.""")

        elif classifier_or_regressor == "Regressors":
            st.subheader("Regression Comparison")
            st.write(regression_comparision)
            st.write("")

            # Display the summary paragraph adjacent to the min values
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"* The minimum ***MAE Value for Sales*** is ***{min_regression_values['MAE Value for Sales']:.4f}***, achieved by the ***Linear Regression model***.")
                st.write(f"* The minimum ***RMSE Value for Sales*** is ***{min_regression_values['RMSE Value for Sales']:.4f}***, achieved by the ***Linear Regression model***.")
                st.write(f"* The minimum ***MAE Value for Quantity*** is ***{min_regression_values['MAE Value for Quantity']:.4f}***, achieved by the ***Decision Tree model***.")
                st.write(f"* The minimum ***RMSE Value for Quantity*** is ***{min_regression_values['RMSE Value for Quantity']:.4f}***, achieved by the ***Decision Tree model***.")
            with col2:
                st.write("""Surprisingly, the Linear Regression model outperformed other models in predicting sales, followed closely by the Decision Tree regression model. 
                For predicting order quantity, both the Random Forest and eXtreme Gradient Boosting models showed excellent performance. 
                How do these models compare to a neural network model in predicting order quantity?""")
                st.write("""As we saw from the Modelling Metrics - The MAE and RMSE scores for the neural network models are 0.005 and 0.01, which are quite favorable. 
                However, it is surprising that the Linear Regression and Decision Tree models achieved even lower MAE and RMSE scores.""")
    
    st.markdown("---")
    st.write("For Modelling Information and the Technical Aspects, check out -> Professional of Technology -> Modelling Metrics or also check the ***[official code](https://github.com/andrew-jxhn/CMSE830_FDSProject/blob/main/CMSE%20830%20-%20IDA-EDA-Model.ipynb)***.")

elif tabs == "Technology":
    selected_page = st.sidebar.selectbox("", ["IDA", "EDA", "Missingness Analysis", "Modelling Metrics"])

    if selected_page == "IDA":
        st.header('Initial Data Analysis (IDA)')

        # Data collection and importation
        st.subheader('1. Data Collection and Importation')
        st.write(f"Data source: DataCoSupplyChainDataset.csv")
        st.write(f"Number of records: {len(df)}")
        st.write(f"Number of features: {len(df.columns)}")
        st.markdown("---") 

        # Data cleaning and preprocessing
        st.subheader('2. Data Cleaning and Preprocessing')
        st.write("Preprocessing steps:")
        st.write("- Converted 'order date (DateOrders)' and 'shipping date (DateOrders)' to datetime")
        st.markdown("---") 

        # Variable identification and classification
        st.subheader('3. Variable Identification and Classification')
        variable_types = pd.DataFrame({
            'Variable': df.columns,
            'Type': df.dtypes,
            'Non-Null Count': df.notnull().sum(),
            'Null Count': df.isnull().sum()
        })
        st.dataframe(variable_types)
        st.markdown("---") 

        # Basic descriptive statistics
        st.subheader('4. Basic Descriptive Statistics')
        st.write(df.describe())
        st.markdown("---") 

        # Data quality assessment
        st.subheader('5. Data Quality Assessment')
        st.write("Checking for duplicates:")
        st.write(f"Number of duplicate rows: {df.duplicated().sum()}")
        st.markdown("---") 

        # Missing data analysis
        st.subheader('6. Missing Data Analysis')
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_data_pct = 100 * missing_data / len(df)
        missing_data_table = pd.concat([missing_data, missing_data_pct], axis=1, keys=['Total', 'Percent'])
        st.dataframe(missing_data_table[missing_data_table['Total'] > 0])
        st.markdown("---") 

        # Outlier detection
        st.subheader('7. Outlier Detection')
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns for brevity
            fig = go.Figure()
            fig.add_trace(go.Box(y=df[col], name=col))
            fig.update_layout(title=f'Box Plot for {col}')
            st.plotly_chart(fig)

    elif selected_page == "EDA":
        st.header('Exploratory Data Analysis (EDA)')
        
        # Sidebar for EDA options
        analysis_option = st.sidebar.selectbox('Select EDA Analysis', [
            'Correlation Analysis', 
            'Market & Region Sales', 
            'Product Category Analysis', 
            'Time Series Analysis', 
            'Payment Type Distribution', 
            'Loss and Fraud Analysis',
            'Investigating Fraud Analysis',
            'Late Delivery Analysis'
        ])

        if analysis_option == 'Correlation Analysis':
            st.subheader('Correlation Heatmap')
            data_numeric = df.select_dtypes(include=['number'])
            fig, ax = plt.subplots(figsize=(24, 12))
            sns.heatmap(data_numeric.corr(), annot=True, linewidths=.5, fmt='.1g', cmap='coolwarm')
            st.pyplot(fig)
            plt.close(fig)
            st.write("We can observe that product price price has high correlation with Sales, Order Item Total.")
            st.write("To analyze supply chain data, it's important to identify the region with the highest sales. This can be achieved using the groupby method to group similar market regions and the sum function to total their sales.")
            st.subheader('Please move onto Market & Region Sales Analysis...')

        elif analysis_option == 'Market & Region Sales':
            st.subheader('Sales by Market and Region')
            col1, col2 = st.columns(2)

            with col1:
                st.write('Total Sales by Market')
                market = df.groupby('Market', observed=False)
                market_sales = market['Sales per customer'].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(18, 8))
                sns.barplot(x=market_sales.index, y=market_sales.values, palette='viridis')
                plt.title("Total Sales for All Markets")
                plt.ylabel('Sales per Customer')
                plt.xlabel('Market')
                plt.xticks(rotation=30)
                st.pyplot(fig)
                plt.close(fig)


            with col2:
                st.write('Total Sales by Region')
                region = df.groupby('Order Region', observed=False)
                region_sales = region['Sales per customer'].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(18, 8))
                sns.barplot(x=region_sales.index, y=region_sales.values, palette='coolwarm')
                plt.title("Total Sales for All Regions")
                plt.ylabel('Sales per Customer')
                plt.xlabel('Order Region')
                plt.xticks(rotation=30)
                st.pyplot(fig)
                plt.close(fig)
            
            st.write("The graph shows that the European market has the highest sales, while Africa has the lowest. Within these markets, Western Europe and Central America reported the highest sales.")
            st.write("To determine which product category has the highest sales, the same method can be applied.")
            st.subheader("Please move onto Product Category Analysis Analysis...")

        elif analysis_option == 'Product Category Analysis':
            st.subheader('Product Category Performance')
            cat = df.groupby('Category Name')
            total_sales = cat['Sales per customer'].sum().sort_values(ascending=False)
            average_sales = cat['Sales per customer'].mean().sort_values(ascending=False)
            average_prices = cat['Product Price'].mean().sort_values(ascending=False)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write('Fig-1: Total Sales by Category')
                fig, ax = plt.subplots(figsize=(14, 8))
                sns.barplot(x=total_sales.index, y=total_sales.values, palette='viridis')
                plt.title("Total Sales")
                plt.xlabel("Category Name")
                plt.ylabel("Sales per Customer")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                st.write('Fig-2: Average Sales by Category')
                fig, ax = plt.subplots(figsize=(14, 8))
                sns.barplot(x=average_sales.index, y=average_sales.values, palette='coolwarm')
                plt.title("Average Sales")
                plt.xlabel("Category Name")
                plt.ylabel("Average Sales per Customer")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
                plt.close(fig)

            with col3:
                st.write('Fig-3: Average Price by Category')
                fig, ax = plt.subplots(figsize=(14, 8))
                sns.barplot(x=average_prices.index, y=average_prices.values, palette='magma')
                plt.title("Average Price")
                plt.xlabel("Category Name")
                plt.ylabel("Average Product Price")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
                plt.close(fig)
            
            st.write("Figure 1 shows that the fishing category had the highest sales, followed by cleats. Interestingly, the top seven products with the highest average prices are also the best-sellers, with computers achieving nearly 1,350 sales despite a $1,500 price tag. Given the strong correlation between price and sales, it will be insightful to analyze how price impacts sales trends across all products.")
            st.subheader("Please move onto Time Series Analysis...")

        elif analysis_option == 'Time Series Analysis':
            st.subheader('Sales Time Analysis')

            fig, ax = plt.subplots(figsize=(12, 6))  # Set figure size
            sns.scatterplot(data=df, x='Product Price', y='Sales per customer',
                            color='blue', marker='o', s=100, edgecolor='w', linewidth=1, ax=ax)

            # Add a regression line to visualize the trend
            sns.regplot(data=df, x='Product Price', y='Sales per customer', 
                        scatter=False, color='orange', line_kws={'linestyle': 'dotted'}, ax=ax)

            plt.title('Product Price vs Sales per Customer', fontsize=18)
            plt.xlabel('Product Price', fontsize=14)
            plt.ylabel('Sales per Customer', fontsize=14)

            st.pyplot(fig)
            plt.close(fig)
            
            st.write("Prices show a linear relationship with sales. To identify which quarter had the highest sales, we can analyze order timestamps by breaking them down into years, months, weekdays, and hours to better observe the trends.")

            # Create datetime-based features
            df['order_year'] = pd.DatetimeIndex(df['order date (DateOrders)']).year
            df['order_month'] = pd.DatetimeIndex(df['order date (DateOrders)']).month
            df['order_week_day'] = pd.DatetimeIndex(df['order date (DateOrders)']).day_name()
            df['order_hour'] = pd.DatetimeIndex(df['order date (DateOrders)']).hour
            df['order_month_year'] = pd.to_datetime(df['order date (DateOrders)']).dt.to_period('M')
            
            # Calculate quarterly sales
            quarter_sales = df.groupby('order_month_year')['Sales'].sum().to_timestamp().resample('QE').mean()
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(15, 6))
            sns.lineplot(x=quarter_sales.index, y=quarter_sales.values, marker='o', color='blue', linewidth=2, ax=ax)
            
            plt.title('Average Quarterly Sales', fontsize=20, fontweight='bold')
            plt.xlabel('Quarter', fontsize=14)
            plt.ylabel('Average Sales', fontsize=14)
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Display in Streamlit
            st.pyplot(fig)
            plt.close(fig)
            
            # Optional: Add some insights
            st.write("The graph shows consistent sales from Q1 2015 to Q3 2017, followed by a sudden decline in Q1 2018. What are the purchasing trends across weekdays, hours, and months?")

            df['order_year'] = pd.DatetimeIndex(df['order date (DateOrders)']).year
            df['order_month'] = pd.DatetimeIndex(df['order date (DateOrders)']).month
            df['order_week_day'] = pd.DatetimeIndex(df['order date (DateOrders)']).day_name()
            df['order_hour'] = pd.DatetimeIndex(df['order date (DateOrders)']).hour

            fig, axs = plt.subplots(2, 2, figsize=(20, 15))
            
            # Sales by Year
            df.groupby('order_year')['Sales'].mean().plot(kind='line', color='skyblue', linewidth=2, ax=axs[0,0])
            axs[0,0].set_title('Average Sales per Year')
            axs[0,0].set_ylabel('Average Sales')
            
            # Sales by Weekday
            df.groupby("order_week_day")['Sales'].mean().plot(kind='line', color='lightgreen', linewidth=2, ax=axs[0,1])
            axs[0,1].set_title('Average Sales per Weekday')
            axs[0,1].set_ylabel('Average Sales')
            
            # Sales by Hour
            df.groupby("order_hour")['Sales'].mean().plot(kind='line', color='coral', linewidth=2, ax=axs[1,0])
            axs[1,0].set_title('Average Sales per Hour')
            axs[1,0].set_ylabel('Average Sales')
            
            # Sales by Month
            df.groupby("order_month")['Sales'].mean().plot(kind='line', color='lightblue', linewidth=2, ax=axs[1,1])
            axs[1,1].set_title('Average Sales per Month')
            axs[1,1].set_ylabel('Average Sales')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.write("The analysis reveals how price affects sales, as well as when and which products see the most sales. October had the highest number of orders, followed by November, with other months showing consistent sales. The peak year for orders was 2017. Saturdays had the highest average sales, while Wednesdays had the lowest. Average sales remained stable throughout the day, with a standard deviation of 3.")
            st.write("It's also essential to understand the preferred payment methods for purchasing products across regions. This can be determined by using the .unique() method to identify the different payment options.")
            st.subheader("Please move onto Payment Type Distribution Analysis...")

        elif analysis_option == 'Payment Type Distribution':
            st.subheader('Payment Type Across Regions')
            st.write("['DEBIT', 'TRANSFER', 'CASH', 'PAYMENT'] are the payment types")
            payment_types = df['Type'].unique()
            counts = {ptype: df[df['Type'] == ptype]['Order Region'].value_counts() for ptype in payment_types}
            
            fig, ax = plt.subplots(figsize=(20, 8))
            n_groups = len(df['Order Region'].value_counts().index)
            bar_width = 0.2
            
            for i, ptype in enumerate(payment_types):
                ax.bar(np.arange(n_groups) + i * bar_width, counts[ptype], bar_width, label=ptype)
            
            ax.set_xlabel('Order Regions')
            ax.set_ylabel('Number of Payments')
            ax.set_title('Distribution of Payment Types Across Regions')
            ax.legend(title='Payment Type')
            ax.set_xticks(np.arange(n_groups) + bar_width * (len(payment_types) - 1) / 2)
            ax.set_xticklabels(df['Order Region'].value_counts().index, rotation=45, ha='right')
            st.pyplot(fig)
            plt.close(fig)
            st.write("Debit is the most preferred payment method across all regions, while cash is the least preferred. Some products show a negative benefit per order, indicating revenue loss for the company. Which products are experiencing this?")
            st.subheader("Please move onto Loss and Fraud Analysis...")

        elif analysis_option == 'Loss and Fraud Analysis':
            st.subheader('Loss and Fraud Analysis')
            loss = df[df['Benefit per order'] < 0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write('Top 10 Products with Most Loss')
                fig, ax = plt.subplots(figsize=(20, 8))
                loss['Category Name'].value_counts().nlargest(10).plot.bar(color='coral', ax=ax)
                plt.title("Top 10 Products with Most Loss")
                plt.xlabel("Product Categories")
                plt.ylabel("Loss Amount")
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                st.write('Top 10 Regions with Most Loss')
                fig, ax = plt.subplots(figsize=(20, 8))
                loss['Order Region'].value_counts().nlargest(10).plot.bar(color='skyblue', ax=ax)
                plt.title("Top 10 Regions with Most Loss")
                plt.xlabel("Order Regions")
                plt.ylabel("Loss Amount")
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close(fig)

            st.write(f'Total revenue lost with orders: {loss["Benefit per order"].sum()}')
            st.write("Total lost sales are around 3.9 million, a significant amount. The Cleats category has the highest losses, followed by Men's Footwear. Most losses occur in Central America and Western Europe, possibly due to suspected fraud or late deliveries. Identifying which payment methods are associated with fraud could help prevent future incidents.")
            st.write("It’s evident that no fraud is associated with DEBIT, CASH, or PAYMENT methods; all suspected fraudulent orders likely involve wire transfers from abroad. Which region and product are most frequently linked to these suspected frauds?")

            # Fraud Analysis
            high_fraud = df[(df['Order Status'] == 'SUSPECTED_FRAUD') & (df['Type'] == 'TRANSFER')]
            
            fig, ax = plt.subplots(figsize=(24, 12))
            high_fraud['Order Region'].value_counts().plot.pie(
                startangle=180, 
                explode=[0.1] + [0]*(len(high_fraud['Order Region'].value_counts())-1),
                autopct='%.1f%%',
                shadow=True,
                ax=ax
            )
            plt.title("Regions with Highest Fraud")
            st.pyplot(fig)
            plt.close(fig)
            st.write("The majority of suspected fraud orders originate from Western Europe, accounting for about 17.4% of total orders, followed by Central America at 15.5%. Which product is most frequently associated with these suspected frauds?")
            st.subheader("Please move onto Investigating Fraud Analysis...")

        elif analysis_option == 'Investigating Fraud Analysis':
            st.subheader("Investigating Fraud Analysis")
            # Filter for suspected fraud orders
            high_fraud1 = df[(df['Order Status'] == 'SUSPECTED_FRAUD')] 
            high_fraud2 = df[(df['Order Status'] == 'SUSPECTED_FRAUD') & 
                            (df['Order Region'] == 'Western Europe')]
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(20, 8))
            
            # Plotting bar chart for top 10 most suspected fraud categories in all regions
            fraud1 = high_fraud1['Category Name'].value_counts().nlargest(10).plot.bar(
                ax=ax, position=0, width=0.4, color='orange', label="All regions"
            )
            
            # Plotting bar chart for top 10 most suspected fraud categories in Western Europe
            fraud2 = high_fraud2['Category Name'].value_counts().nlargest(10).plot.bar(
                ax=ax, position=1, width=0.4, color='green', label="Western Europe"
            )
            
            plt.title("Top 10 Products with Highest Fraud Detections", size=15)
            plt.xlabel("Products", size=13)
            plt.ylabel("Number of Suspected Fraud Cases", size=13)
            plt.legend()
            plt.ylim(0, 600)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Display in Streamlit
            st.pyplot(fig)
            plt.close(fig)
            
            # Add insights
            st.write("The bar chart compares suspected fraud cases across different product categories.")
            st.write("The orange bars represent fraud cases across all regions, while green bars show cases specific to Western Europe.")
            st.write("It's surprising that the cleats department has the highest suspected fraud, followed by men's footwear, in all regions and specifically in Western Europe. Which customers are responsible for these frauds?")

            # Filtering out suspected fraud orders
            df['Customer Full Name'] = df['Customer Fname'].astype(str)+df['Customer Lname'].astype(str)
            cus = df[df['Order Status'] == 'SUSPECTED_FRAUD']
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(20, 8))
            
            # Plotting top 10 customers with most fraud
            top_customers = cus['Customer Full Name'].value_counts().nlargest(10)
            bars = ax.bar(top_customers.index, top_customers.values, color='skyblue', edgecolor='black')
            
            # Adding title and labels
            ax.set_title("Top 10 Customers with Highest Fraud Incidents", fontsize=18, fontweight='bold')
            ax.set_xlabel("Customer Names", fontsize=14)
            ax.set_ylabel("Number of Fraudulent Orders", fontsize=14)
            
            # Adding value annotations on top of the bars
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', fontsize=12)
            
            # Improving x-axis clarity
            plt.xticks(rotation=45, ha='right', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)  # Adding horizontal grid lines
            plt.tight_layout()
            
            # Display in Streamlit
            st.pyplot(fig)
            plt.close(fig)
            
            # Add insights
            st.write("Mary Smith attempted fraud 528 times, which is astonishing. What was the total amount involved in her fraudulent orders?")

            # Filtering orders of Mary Smith with suspected fraud
            amount = df[(df['Customer Full Name'] == 'MarySmith')&(df['Order Status'] == 'SUSPECTED_FRAUD')]
            total_fraud_amount = amount['Sales'].sum()

            st.write(f"The total fraudulent amount reached nearly ${total_fraud_amount:,.2f}, a substantial figure. Mary used different addresses for each order, resulting in a new customer ID each time, complicating her identification and banning. These factors must be considered to enhance fraud detection algorithms for more accurate identification.")
            st.write("Timely product delivery is crucial for customer satisfaction in a supply chain. Which product categories are experiencing the most late deliveries?")
            st.subheader("Please move onto Late Delivery Analysis...")

        elif analysis_option == 'Late Delivery Analysis':
            # First Visualization: Top 10 Products with Most Late Deliveries
            st.subheader('Top 10 Products with Most Late Deliveries')
            fig1, ax1 = plt.subplots(figsize=(20, 8))
            late_delivery = df[df['Delivery Status'] == 'Late delivery']
            top_late_products = late_delivery['Category Name'].value_counts().nlargest(10)
            
            bars1 = ax1.bar(top_late_products.index, top_late_products.values, color='skyblue', edgecolor='black')
            ax1.set_title("Top 10 Products with Most Late Deliveries", fontsize=18, fontweight='bold')
            ax1.set_xlabel("Product Categories", fontsize=14)
            ax1.set_ylabel("Number of Late Deliveries", fontsize=14)
            ax1.set_xticklabels(top_late_products.index, rotation=45, ha='right', fontsize=12)
            ax1.set_ylim(0, top_late_products.values.max() * 1.1)
            
            for bar in bars1:
                yval = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontsize=12)
            
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)
            st.write("It can be seen that orders with Cleats department is getting delayed the most followed by Men's Footwear.For some orders risk of late delivery is given in data.The products with late delivery risk are compared with late delivered products.")
            
            # Second Visualization: Late Delivery Risks vs. Late Deliveries by Region
            st.subheader('Late Delivery Risks vs. Late Deliveries by Region')
            fig2, ax2 = plt.subplots(figsize=(20, 8))
            
            xyz1 = df[df['Late_delivery_risk'] == 1]
            xyz2 = df[df['Delivery Status'] == 'Late delivery']
            count1 = xyz1['Order Region'].value_counts()
            count2 = xyz2['Order Region'].value_counts()
            names = df['Order Region'].value_counts().keys()
            n_groups = len(names)
            
            index = np.arange(n_groups)
            bar_width = 0.35
            opacity = 0.8
            
            bars21 = ax2.bar(index, count1, bar_width, alpha=opacity, color='coral', label='Risk of Late Delivery')
            bars22 = ax2.bar(index + bar_width, count2, bar_width, alpha=opacity, color='gold', label='Late Deliveries')
            
            ax2.set_xlabel('Order Regions', fontsize=14)
            ax2.set_ylabel('Number of Shipments', fontsize=14)
            ax2.set_title('Late Delivery Risks vs. Late Deliveries by Region', fontsize=18, fontweight='bold')
            ax2.legend(fontsize=12)
            ax2.set_xticks(index + bar_width / 2)
            ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=12)
            
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
            st.write("In conclusion, products at risk of late delivery are consistently being delivered late across all regions. To mitigate this, the company could enhance shipping methods or allow more time for deliveries, keeping customers informed of expected arrival dates. It would also be valuable to analyze the number of late deliveries by shipment method across different regions.")
            
            # Third Visualization: Late Deliveries by Shipping Method Across Regions
            st.subheader('Late Deliveries by Shipping Method Across Regions')
            fig3, ax3 = plt.subplots(figsize=(20, 8))
            
            xyz1 = df[(df['Delivery Status'] == 'Late delivery') & (df['Shipping Mode'] == 'Standard Class')]
            xyz2 = df[(df['Delivery Status'] == 'Late delivery') & (df['Shipping Mode'] == 'First Class')]
            xyz3 = df[(df['Delivery Status'] == 'Late delivery') & (df['Shipping Mode'] == 'Second Class')]
            xyz4 = df[(df['Delivery Status'] == 'Late delivery') & (df['Shipping Mode'] == 'Same Day')]
            
            count1 = xyz1['Order Region'].value_counts()
            count2 = xyz2['Order Region'].value_counts()
            count3 = xyz3['Order Region'].value_counts()
            count4 = xyz4['Order Region'].value_counts()
            
            names = df['Order Region'].value_counts().keys()
            n_groups = len(names)
            
            index = np.arange(n_groups)
            bar_width = 0.15
            opacity = 0.8
            
            ax3.bar(index, count1, bar_width, alpha=opacity, color='royalblue', label='Standard Class')
            ax3.bar(index + bar_width, count2, bar_width, alpha=opacity, color='tomato', label='First Class')
            ax3.bar(index + 2 * bar_width, count3, bar_width, alpha=opacity, color='mediumseagreen', label='Second Class')
            ax3.bar(index + 3 * bar_width, count4, bar_width, alpha=opacity, color='gold', label='Same Day')
            
            ax3.set_xlabel('Order Regions', fontsize=14)
            ax3.set_ylabel('Number of Shipments', fontsize=14)
            ax3.set_title('Late Deliveries by Shipping Method Across Regions', fontsize=16)
            ax3.legend(title='Shipping Methods', fontsize=12)
            ax3.set_xticks(index + 1.5 * bar_width)
            ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=12)
            
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
            
            st.write("As expected the most number of late deliveries for all regions occured with standard class shipping,with same day shipping being the one with least number of late deliveries.Both the first class and second class shipping have almost equal number of late deliveries.")

    elif selected_page == "Missingness Analysis":
        st.header('Missingness Analysis and Imputation')
        
        # Function to induce missingness
        def induce_missingness(df, columns, percentage):
            df_copy = df.copy()
            for column in columns:
                mask = np.random.rand(len(df_copy)) < percentage/100
                df_copy.loc[mask, column] = np.nan
            return df_copy
        
        # Function to impute missing values
        def impute_missing(df, strategy='mean'):
            imputer = SimpleImputer(strategy=strategy)
            df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            return df_imputed
        
        # Select columns for missingness induction
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_cols = st.multiselect('Select columns to induce missingness', numeric_cols)
        
        # Slider for missingness percentage
        missingness_pct = st.slider('Select percentage of missingness to induce', 0, 50, 10)
        
        if st.button('Induce Missingness'):
            df_missing = induce_missingness(df, selected_cols, missingness_pct)
            st.write(f"Missingness induced. New missing value counts:")
            st.write(df_missing[selected_cols].isnull().sum())
            
            # Imputation
            imputation_strategy = st.selectbox('Select imputation strategy', ['mean', 'median', 'most_frequent'])
            df_imputed = impute_missing(df_missing[selected_cols], strategy=imputation_strategy)
            
            st.write(f"Data imputed using {imputation_strategy} strategy. Comparison of original vs imputed data:")
            for col in selected_cols:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df[col], name='Original'))
                fig.add_trace(go.Histogram(x=df_imputed[col], name='Imputed'))
                fig.update_layout(barmode='overlay', title=f'Distribution of {col} - Original vs Imputed')
                st.plotly_chart(fig)

    elif selected_page == "Modelling Metrics":
        st.subheader("Machine Learning Model Performance Metrics")
        
        # # Add a radio button to select between Regression and Classification
        # model_type = st.radio("Select Model Type", ["Regression", "Classification"])
        
        # if model_type == "Regression":
        #     # Regression Model Metrics
        #     regression_model_metrics = {
        #         "Lasso": {
        #             "Sales Prediction": {
        #                 "MAE": 1.337426,
        #                 "RMSE": 2.095162
        #             },
        #             "Order Quantity Prediction": {
        #                 "MAE": 1.253982, 
        #                 "RMSE": 1.435849
        #             }
        #         },
        #         "Ridge": {
        #             "Sales Prediction": {
        #                 "MAE": 0.269265,
        #                 "RMSE": 0.473015
        #             },
        #             "Order Quantity Prediction": {
        #                 "MAE": 0.340329,
        #                 "RMSE": 0.525778
        #             }
        #         },
        #         "LightGBM": {
        #             "Sales Prediction": {
        #                 "MAE": 0.550048,
        #                 "RMSE": 4.348850
        #             },
        #             "Order Quantity Prediction": {
        #                 "MAE": 0.000410,
        #                 "RMSE": 0.004389
        #             }
        #         },
        #         "Random Forest": {
        #             "Sales Prediction": {
        #                 "MAE": 0.202989,
        #                 "RMSE": 1.951588
        #             },
        #             "Order Quantity Prediction": {
        #                 "MAE": 0.000076,
        #                 "RMSE": 0.006126
        #             }
        #         },
        #         "XGBoost": {
        #             "Sales Prediction": {
        #                 "MAE": 0.157674,
        #                 "RMSE": 0.548000
        #             },
        #             "Order Quantity Prediction": {
        #                 "MAE": 0.000052,
        #                 "RMSE": 0.001299
        #             }
        #         },
        #         "Decision Tree": {
        #             "Sales Prediction": {
        #                 "MAE": 0.010847,
        #                 "RMSE": 0.784462
        #             },
        #             "Order Quantity Prediction": {
        #                 "MAE": 0.000000,
        #                 "RMSE": 0.000000
        #             }
        #         },
        #         "Linear Regression": {
        #             "Sales Prediction": {
        #                 "MAE": 0.000590,
        #                 "RMSE": 0.001484
        #             },
        #             "Order Quantity Prediction": {
        #                 "MAE": 0.338359,
        #                 "RMSE": 0.525365
        #             }
        #         }
        #     }
            
        #     # Dropdowns for Regression Model
        #     selected_regression_model = st.selectbox("Select Regression Model", list(regression_model_metrics.keys()))
        #     prediction_type = st.selectbox("Select Prediction Type", ["Sales Prediction", "Order Quantity Prediction"])
            
        #     # Get selected model's metrics
        #     metrics = regression_model_metrics[selected_regression_model][prediction_type]
            
        #     # Display metrics
        #     st.markdown(f"### {selected_regression_model} - {prediction_type} Performance")
            
        #     # Create columns for metrics
        #     col1, col2 = st.columns(2)
            
        #     with col1:
        #         st.metric(label="Mean Absolute Error (MAE)", value=f"{metrics['MAE']:.6f}")
            
        #     with col2:
        #         st.metric(label="Root Mean Squared Error (RMSE)", value=f"{metrics['RMSE']:.6f}")
        
        # else:
            # Classification Model Metrics
        #     classification_model_metrics = {
        #     "Logistic Regression": {
        #         "Fraud Detection": {
        #             "Accuracy": "97.79%",
        #             "Recall": "57.89%",
        #             "F1 Score": "31.88%",
        #             "Confusion Matrix": [[35118, 136], [663, 187]]
        #         },
        #         "Late Delivery Prediction": {
        #             "Accuracy": "98.85%",
        #             "Recall": "97.94%",
        #             "F1 Score": "98.96%",
        #             "Confusion Matrix": [[15891, 416], [0, 19797]]
        #         }
        #     },
        #     "Gaussian Naive Bayes": {
        #     "Fraud Detection": {
        #         "Accuracy": "87.85%",
        #         "Recall": "16.23%",
        #         "F1 Score": "27.93%",
        #         "Confusion Matrix": [[30867, 4387], [0, 850]]
        #     },
        #     "Late Delivery Prediction": {
        #         "Accuracy": "57.27%",
        #         "Recall": "56.20%",
        #         "F1 Score": "71.96%",
        #         "Confusion Matrix": [[882, 15425], [3, 19794]]
        #     }
        # },
        #     "Linear SVC": {
        #     "Fraud Detection": {
        #         "Accuracy": "97.73%",
        #         "Recall": "55.59%",
        #         "F1 Score": "27.99%",
        #         "Confusion Matrix": [[35127, 127], [691, 159]]
        #     },
        #     "Late Delivery Prediction": {
        #         "Accuracy": "98.85%",
        #         "Recall": "97.94%",
        #         "F1 Score": "98.96%",
        #         "Confusion Matrix": [[15891, 416], [0, 19797]]
        #     }
        # },
        #     "K-Nearest Neighbors": {
        #     "Fraud Detection": {
        #         "Accuracy": "97.18%",
        #         "Recall": "36.61%",
        #         "F1 Score": "31.20%",
        #         "Confusion Matrix": [[34854, 400], [619, 231]]
        #     },
        #     "Late Delivery Prediction": {
        #         "Accuracy": "81.63%",
        #         "Recall": "83.95%",
        #         "F1 Score": "83.07%",
        #         "Confusion Matrix": [[13196, 3111], [3522, 16275]]
        #     }
        # },
        #     "Linear Discriminant Analysis": {
        #     "Fraud Detection": {
        #         "Accuracy": "97.88%",
        #         "Recall": "56.68%",
        #         "F1 Score": "48.86%",
        #         "Confusion Matrix": [[34975, 279], [485, 365]]
        #     },
        #     "Late Delivery Prediction": {
        #         "Accuracy": "97.91%",
        #         "Recall": "97.71%",
        #         "F1 Score": "98.10%",
        #         "Confusion Matrix": [[15851, 456], [300, 19497]]
        #     }
        # },
        #     "Random Forest": {
        #     "Fraud Detection": {
        #         "Accuracy": "98.67%",
        #         "Recall": "97.44%",
        #         "F1 Score": "61.29%",
        #         "Confusion Matrix": [[35244, 10], [470, 380]]
        #     },
        #     "Late Delivery Prediction": {
        #         "Accuracy": "98.85%",
        #         "Recall": "97.94%",
        #         "F1 Score": "98.96%",
        #         "Confusion Matrix": [[15891, 416], [0, 19797]]
        #     }
        # },
        #     "Extra Trees Classifier": {
        #     "Fraud Detection": {
        #         "Accuracy": "98.62%",
        #         "Recall": "98.61%",
        #         "F1 Score": "58.79%",
        #         "Confusion Matrix": [[35249, 5], [494, 356]]
        #     },
        #     "Late Delivery Prediction": {
        #         "Accuracy": "99.03%",
        #         "Recall": "98.27%",
        #         "F1 Score": "99.12%",
        #         "Confusion Matrix": [[15958, 349], [1, 19796]]
        #     }
        # },
        #     "XGBoost Classifier": {
        #     "Fraud Detection": {
        #         "Accuracy": "98.91%",
        #         "Recall": "91.02%",
        #         "F1 Score": "72.07%",
        #         "Confusion Matrix": [[35204, 50], [343, 507]]
        #     },
        #     "Late Delivery Prediction": {
        #         "Accuracy": "99.13%",
        #         "Recall": "98.45%",
        #         "F1 Score": "99.21%",
        #         "Confusion Matrix": [[15996, 311], [3, 19794]]
        #     }
        # },
        #     "Decision Tree Classifier": {
        #     "Fraud Detection": {
        #         "Accuracy": "99.06%",
        #         "Recall": "81.48%",
        #         "F1 Score": "79.52%",
        #         "Confusion Matrix": [[35104, 150], [190, 660]]
        #     },
        #     "Late Delivery Prediction": {
        #         "Accuracy": "99.25%",
        #         "Recall": "99.37%",
        #         "F1 Score": "99.32%",
        #         "Confusion Matrix": [[16182, 125], [145, 19652]]
        #     }
        # },
        #     "Bagging Classifier": {
        #         "Fraud Detection": {
        #             "Accuracy": "99.12%",
        #             "Recall": "94.05%",
        #             "F1 Score": "78.21%",
        #             "Confusion Matrix": [[35218, 36], [281, 569]]
        #         },
        #         "Late Delivery Prediction": {
        #             "Accuracy": "99.48%",
        #             "Recall": "99.13%",
        #             "F1 Score": "99.53%",
        #             "Confusion Matrix": [[16134, 173], [13, 19784]]
        #         }
        #     }
        # }
            
        #     # Dropdowns for Classification Model
        #     selected_classification_model = st.selectbox("Select Classification Model", list(classification_model_metrics.keys()))
        #     prediction_type = st.selectbox("Select Prediction Type", ["Fraud Detection", "Late Delivery Prediction"])
            
        #     # Get selected model's metrics
        #     metrics = classification_model_metrics[selected_classification_model][prediction_type]
            
        #     # Display metrics
        #     st.markdown(f"### {selected_classification_model} - {prediction_type} Performance")
            
        #     # Create columns for metrics
        #     col1, col2, col3 = st.columns(3)
            
        #     with col1:
        #         st.metric(label="Accuracy", value=metrics["Accuracy"])
            
        #     with col2:
        #         st.metric(label="Recall", value=metrics["Recall"])
            
        #     with col3:
        #         st.metric(label="F1 Score", value=metrics["F1 Score"])
            
        #     # Confusion Matrix
        #     st.write("Confusion Matrix:")
        #     matrix_df = pd.DataFrame(metrics["Confusion Matrix"], 
        #                               columns=['Predicted 0', 'Predicted 1'], 
        #                               index=['Actual 0', 'Actual 1'])
        #     st.table(matrix_df)

        def display_regression_metrics(model, prediction_type):
            metrics = regression_model_metrics[model][prediction_type]
            st.markdown(f"### {model} - {prediction_type} Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Mean Absolute Error (MAE)", value=f"{metrics['MAE']:.6f}")
            with col2:
                st.metric(label="Root Mean Squared Error (RMSE)", value=f"{metrics['RMSE']:.6f}")

        def display_classification_metrics(model, prediction_type):
            metrics = classification_model_metrics[model][prediction_type]
            st.markdown(f"### {model} - {prediction_type} Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Accuracy", value=metrics["Accuracy"])
            with col2:
                st.metric(label="Recall", value=metrics["Recall"])
            with col3:
                st.metric(label="F1 Score", value=metrics["F1 Score"])
            st.write("Confusion Matrix:")
            matrix_df = pd.DataFrame(metrics["Confusion Matrix"], 
                                    columns=['Predicted 0', 'Predicted 1'], 
                                    index=['Actual 0', 'Actual 1'])
            st.table(matrix_df)

        def display_neural_network_metrics(model_type, prediction_type):
            if model_type == "Classification":
                metrics = classification_nn_metrics[prediction_type]
                st.markdown(f"### Classification Neural Network - {prediction_type} Performance")
                col1, col2 = st.columns(2)
                #, col3
                with col1:
                    st.metric(label="Accuracy", value=metrics["Accuracy"])
                # with col2:
                    # st.metric(label="Recall", value=metrics["Recall"])
                with col2:
                    st.metric(label="F1 Score", value=metrics["F1 Score"])
                # st.write("Confusion Matrix:")
                # matrix_df = pd.DataFrame(metrics["Confusion Matrix"], 
                                        # columns=['Predicted 0', 'Predicted 1'], 
                                        # index=['Actual 0', 'Actual 1'])
                # st.table(matrix_df)
            else:
                metrics = regression_nn_metrics
                st.markdown(f"### Regression Neural Network - {prediction_type} Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Mean Absolute Error (MAE)", value=f"{metrics['MAE']:.6f}")
                with col2:
                    st.metric(label="Root Mean Squared Error (RMSE)", value=f"{metrics['RMSE']:.6f}")

        # Regression Model Metrics
        regression_model_metrics = {
                "Lasso": {
                    "Sales Prediction": {
                        "MAE": 1.337426,
                        "RMSE": 2.095162
                    },
                    "Order Quantity Prediction": {
                        "MAE": 1.253982, 
                        "RMSE": 1.435849
                    }
                },
                "Ridge": {
                    "Sales Prediction": {
                        "MAE": 0.269265,
                        "RMSE": 0.473015
                    },
                    "Order Quantity Prediction": {
                        "MAE": 0.340329,
                        "RMSE": 0.525778
                    }
                },
                "LightGBM": {
                    "Sales Prediction": {
                        "MAE": 0.550048,
                        "RMSE": 4.348850
                    },
                    "Order Quantity Prediction": {
                        "MAE": 0.000410,
                        "RMSE": 0.004389
                    }
                },
                "Random Forest": {
                    "Sales Prediction": {
                        "MAE": 0.202989,
                        "RMSE": 1.951588
                    },
                    "Order Quantity Prediction": {
                        "MAE": 0.000076,
                        "RMSE": 0.006126
                    }
                },
                "XGBoost": {
                    "Sales Prediction": {
                        "MAE": 0.157674,
                        "RMSE": 0.548000
                    },
                    "Order Quantity Prediction": {
                        "MAE": 0.000052,
                        "RMSE": 0.001299
                    }
                },
                "Decision Tree": {
                    "Sales Prediction": {
                        "MAE": 0.010847,
                        "RMSE": 0.784462
                    },
                    "Order Quantity Prediction": {
                        "MAE": 0.000000,
                        "RMSE": 0.000000
                    }
                },
                "Linear Regression": {
                    "Sales Prediction": {
                        "MAE": 0.000590,
                        "RMSE": 0.001484
                    },
                    "Order Quantity Prediction": {
                        "MAE": 0.338359,
                        "RMSE": 0.525365
                    }
                }
            }

        # Classification Model Metrics
        classification_model_metrics = {
            "Logistic Regression": {
                "Fraud Detection": {
                    "Accuracy": "97.79%",
                    "Recall": "57.89%",
                    "F1 Score": "31.88%",
                    "Confusion Matrix": [[35118, 136], [663, 187]]
                },
                "Late Delivery Prediction": {
                    "Accuracy": "98.85%",
                    "Recall": "97.94%",
                    "F1 Score": "98.96%",
                    "Confusion Matrix": [[15891, 416], [0, 19797]]
                }
            },
            "Gaussian Naive Bayes": {
            "Fraud Detection": {
                "Accuracy": "87.85%",
                "Recall": "16.23%",
                "F1 Score": "27.93%",
                "Confusion Matrix": [[30867, 4387], [0, 850]]
            },
            "Late Delivery Prediction": {
                "Accuracy": "57.27%",
                "Recall": "56.20%",
                "F1 Score": "71.96%",
                "Confusion Matrix": [[882, 15425], [3, 19794]]
            }
        },
            "Linear SVC": {
            "Fraud Detection": {
                "Accuracy": "97.73%",
                "Recall": "55.59%",
                "F1 Score": "27.99%",
                "Confusion Matrix": [[35127, 127], [691, 159]]
            },
            "Late Delivery Prediction": {
                "Accuracy": "98.85%",
                "Recall": "97.94%",
                "F1 Score": "98.96%",
                "Confusion Matrix": [[15891, 416], [0, 19797]]
            }
        },
            "K-Nearest Neighbors": {
            "Fraud Detection": {
                "Accuracy": "97.18%",
                "Recall": "36.61%",
                "F1 Score": "31.20%",
                "Confusion Matrix": [[34854, 400], [619, 231]]
            },
            "Late Delivery Prediction": {
                "Accuracy": "81.63%",
                "Recall": "83.95%",
                "F1 Score": "83.07%",
                "Confusion Matrix": [[13196, 3111], [3522, 16275]]
            }
        },
            "Linear Discriminant Analysis": {
            "Fraud Detection": {
                "Accuracy": "97.88%",
                "Recall": "56.68%",
                "F1 Score": "48.86%",
                "Confusion Matrix": [[34975, 279], [485, 365]]
            },
            "Late Delivery Prediction": {
                "Accuracy": "97.91%",
                "Recall": "97.71%",
                "F1 Score": "98.10%",
                "Confusion Matrix": [[15851, 456], [300, 19497]]
            }
        },
            "Random Forest": {
            "Fraud Detection": {
                "Accuracy": "98.67%",
                "Recall": "97.44%",
                "F1 Score": "61.29%",
                "Confusion Matrix": [[35244, 10], [470, 380]]
            },
            "Late Delivery Prediction": {
                "Accuracy": "98.85%",
                "Recall": "97.94%",
                "F1 Score": "98.96%",
                "Confusion Matrix": [[15891, 416], [0, 19797]]
            }
        },
            "Extra Trees Classifier": {
            "Fraud Detection": {
                "Accuracy": "98.62%",
                "Recall": "98.61%",
                "F1 Score": "58.79%",
                "Confusion Matrix": [[35249, 5], [494, 356]]
            },
            "Late Delivery Prediction": {
                "Accuracy": "99.03%",
                "Recall": "98.27%",
                "F1 Score": "99.12%",
                "Confusion Matrix": [[15958, 349], [1, 19796]]
            }
        },
            "XGBoost Classifier": {
            "Fraud Detection": {
                "Accuracy": "98.91%",
                "Recall": "91.02%",
                "F1 Score": "72.07%",
                "Confusion Matrix": [[35204, 50], [343, 507]]
            },
            "Late Delivery Prediction": {
                "Accuracy": "99.13%",
                "Recall": "98.45%",
                "F1 Score": "99.21%",
                "Confusion Matrix": [[15996, 311], [3, 19794]]
            }
        },
            "Decision Tree Classifier": {
            "Fraud Detection": {
                "Accuracy": "99.06%",
                "Recall": "81.48%",
                "F1 Score": "79.52%",
                "Confusion Matrix": [[35104, 150], [190, 660]]
            },
            "Late Delivery Prediction": {
                "Accuracy": "99.25%",
                "Recall": "99.37%",
                "F1 Score": "99.32%",
                "Confusion Matrix": [[16182, 125], [145, 19652]]
            }
        },
            "Bagging Classifier": {
                "Fraud Detection": {
                    "Accuracy": "99.12%",
                    "Recall": "94.05%",
                    "F1 Score": "78.21%",
                    "Confusion Matrix": [[35218, 36], [281, 569]]
                },
                "Late Delivery Prediction": {
                    "Accuracy": "99.48%",
                    "Recall": "99.13%",
                    "F1 Score": "99.53%",
                    "Confusion Matrix": [[16134, 173], [13, 19784]]
                }
            }
        }

        # Neural Network Metrics
        classification_nn_metrics = {
            "Fraud Detection": {
                "Accuracy": "97.99%",
                # "Recall": "N/A",
                "F1 Score": "96.48%",
                # "Confusion Matrix": [[35179, 75], [589, 261]]
            }
        }

        regression_nn_metrics = {
            "MAE": 0.0064,
            "RMSE": 0.02
        }

        model_type = st.radio("Select Model Type", ["Regression", "Classification", "Neural Network"])
        st.markdown("---") 
        if model_type == "Regression":
            selected_model = st.selectbox("Select Regression Model", list(regression_model_metrics.keys()))
            prediction_type = st.selectbox("Select Prediction Type", ["Sales Prediction", "Order Quantity Prediction"])
            display_regression_metrics(selected_model, prediction_type)
        elif model_type == "Classification":
            selected_model = st.selectbox("Select Classification Model", list(classification_model_metrics.keys()))
            prediction_type = st.selectbox("Select Prediction Type", ["Fraud Detection", "Late Delivery Prediction"])
            display_classification_metrics(selected_model, prediction_type)
        else:
            selected_model_type = st.selectbox("Select Neural Network Model Type", ["Classification", "Regression"])
            if selected_model_type == "Classification":
                prediction_type = st.selectbox("Select Prediction Type", ["Fraud Detection"])
            else:
                prediction_type = st.selectbox("Select Prediction Type", ["Order Quantity Prediction"])
            display_neural_network_metrics(selected_model_type, prediction_type)
        st.markdown("---")
        st.write("Check out the ***[official code](https://github.com/andrew-jxhn/CMSE830_FDSProject/blob/main/CMSE%20830%20-%20IDA-EDA-Model.ipynb)*** implementation, where you will see the complete IDA, EDA, Customer Segmentation Analysis, Modelling, and Feature Importance aspects.")

        st.markdown("---") 
        st.write("***P.S - Not taking Bagging Classifier into consideration, since it has no Feature Importance extracting properties. But it performs really well in comparison to other models.***")

# Add notes for sales prediction, late delivery prediction, demand forecasting, and fraud detection
st.sidebar.markdown("---")
st.sidebar.subheader("*Key variables for Modeling:*")
st.sidebar.write("- Fraud Detection: ***'Order Status'***")
st.sidebar.write("- Late Delivery Prediction: ***'Delivery Status'***")
st.sidebar.write("- Sales Prediction: ***'Sales'***")
st.sidebar.write("- Demand Forecasting: ***'Order Item Quantity'***")

st.sidebar.markdown("---") 
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/andrew-jxhn/) --- [E-Mail](mailto:johnprak@msu.edu) --- [GitHub](https://github.com/andrew-jxhn/CMSE830_FDSProject)")
st.sidebar.markdown("---")
st.sidebar.markdown("***by Andrew John J for CMSE 830.***")
