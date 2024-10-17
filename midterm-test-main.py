import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the data
@st.cache_data
def load_data():
    file_path = r'/workspaces/blank-app/DataCoSupplyChainDataset.csv'
    df = pd.read_csv(file_path, encoding='latin1')
    df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
    df['shipping date (DateOrders)'] = pd.to_datetime(df['shipping date (DateOrders)'])
    return df

df = load_data()

# Streamlit app
st.title('🚚🏭🏗️ DataCo SMART SUPPLY CHAIN Analysis Dashboard')

# Sidebar
st.sidebar.header('Navigation')
page = st.sidebar.radio('Go to', ['Overview', 'IDA', 'EDA', 'Missingness Analysis'])

if page == 'Overview':
    st.header('Project Overview')
    st.write("""
        This project explores the **DataCo Smart Supply Chain** dataset to gain insights 
        into the supply chain performance, focusing on key metrics such as sales, 
        profit, and delivery time.
        
        **IoT and Data Growth**: The widespread adoption of IoT has generated vast amounts of data, valuable for uncovering insights and enhancing decision-making.
        
        **Customer Segmentation**: This project utilizes the DataCo dataset to conduct customer segmentation, enabling better customer understanding and revenue growth.
        
        **Model Selection Challenge**: With multiple data analysis methods and models available, choosing the right one is crucial as model performance varies with data parameters.
        
        **Comparative Studies**: Prior studies (e.g., Carbonneau, Hill, Vakili, Ahmed) have compared traditional forecasting methods with neural networks and machine learning models, revealing varying levels of performance across techniques.
        
        **Project Objective**: This study plans to compare 9 machine learning classifiers and 7 regression models against neural networks. Key tasks include fraud detection, late delivery prediction, sales forecasting, and demand prediction.
        
        **Model Performance (Subject to Change)**:
        - *Classification models*: Logistic Regression, SVM, k-NN, Random Forest, etc., evaluated for accuracy, recall, and F1 score.
        - *Regression models*: Lasso, Ridge, Random Forest, and others, assessed with MAE and RMSE for sales and demand prediction.
    """)

elif page == 'IDA':
    st.header('Initial Data Analysis (IDA)')

    # Data collection and importation
    st.subheader('1. Data Collection and Importation')
    st.write(f"Data source: DataCoSupplyChainDataset.csv")
    st.write(f"Number of records: {len(df)}")
    st.write(f"Number of features: {len(df.columns)}")

    # Data cleaning and preprocessing
    st.subheader('2. Data Cleaning and Preprocessing')
    st.write("Preprocessing steps:")
    st.write("- Converted 'order date (DateOrders)' and 'shipping date (DateOrders)' to datetime")

    # Variable identification and classification
    st.subheader('3. Variable Identification and Classification')
    variable_types = pd.DataFrame({
        'Variable': df.columns,
        'Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Null Count': df.isnull().sum()
    })
    st.dataframe(variable_types)

    # Basic descriptive statistics
    st.subheader('4. Basic Descriptive Statistics')
    st.write(df.describe())

    # Data quality assessment
    st.subheader('5. Data Quality Assessment')
    st.write("Checking for duplicates:")
    st.write(f"Number of duplicate rows: {df.duplicated().sum()}")

    # Missing data analysis
    st.subheader('6. Missing Data Analysis')
    missing_data = df.isnull().sum().sort_values(ascending=False)
    missing_data_pct = 100 * missing_data / len(df)
    missing_data_table = pd.concat([missing_data, missing_data_pct], axis=1, keys=['Total', 'Percent'])
    st.dataframe(missing_data_table[missing_data_table['Total'] > 0])

    # Outlier detection
    st.subheader('7. Outlier Detection')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:5]:  # Limit to first 5 numeric columns for brevity
        fig = go.Figure()
        fig.add_trace(go.Box(y=df[col], name=col))
        fig.update_layout(title=f'Box Plot for {col}')
        st.plotly_chart(fig)

elif page == 'EDA':
    st.header('Exploratory Data Analysis (EDA)')

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
        
        st.write("Suggested distributions to explore:")
        st.write("- Normal distribution")
        st.write("- Poisson distribution")
        st.write("- Exponential distribution")
        st.write("- Log-normal distribution")

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

elif page == 'Missingness Analysis':
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

# Add notes for sales prediction, late delivery prediction, demand forecasting, and fraud detection
st.sidebar.markdown("---")
st.sidebar.subheader("Notes for Predictive Modeling")
st.sidebar.write("Key variables for:")
st.sidebar.write("- Sales Prediction: 'Order Item Quantity', 'Product Price', 'Category Name'")
st.sidebar.write("- Late Delivery Prediction: 'shipping_days', 'Shipping Mode', 'Distance'")
st.sidebar.write("- Demand Forecasting: 'order date (DateOrders)', 'Category Name', 'Department Name'")
st.sidebar.write("- Fraud Detection: 'Order Status', 'Late_delivery_risk', 'Customer Segment'")