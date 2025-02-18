import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error  # Import this line
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from twilio.rest import Client
import datetime
import matplotlib.pyplot as plt 
import time
import warnings

st.header("Welcome to Sales Data Analytics Project")

# List of the first 4 images for the slideshow
image_paths_3 = [
    "salesapp3.jpg",
    "salesapp1.jpg",
    "salesapp2.jpg",
]

# Create a slideshow for the first 3 images with a 2-second interval
placeholder_1 = st.empty()  # Empty placeholder to refresh the image
for i in range(len(image_paths_3)):
    placeholder_1.image(image_paths_3[i], caption="Sales Analytics by Somnath", use_container_width=True)
    time.sleep(8)  # Wait for 8 seconds before displaying the next image


st.write('''What is this? 
            Hi, Myself Somnath Banerjee, Mail ID : somnathbanerjee342000@gmail.com 
            I have created this for sales analytics purpose, I assume that I have multiple trading retail Tech Stores in various cities across PAN India.
            For any important decision making purpose I have to analyse the data for calculating the business performance.
            I have a dummy sales data of my store (assume not real) & I am putting the link of it''')

st.write('''How to use? 
            To use this you have to click on this link to download the sample dummy data which is just 2MB, 
            Download link (Excel) : https://docs.google.com/spreadsheets/d/12h_yB87LM94vXw6kFBbe_I3tGkq-GSRy/edit?usp=sharing&ouid=110430024153326232778&rtpof=true&sd=true 
            Download link (CSV) : https://drive.google.com/file/d/1l3ixYZ3YtFV9AfcXSqgWNYcrATskqfRD/view?usp=sharing
            After downloading the Somnath_Techstore_Dummy_Data_(2023-24).xlsx file, click on the "Browse Files" button. This will open the "File Explorer/Manager," then go to the download folder and select the downloaded dummy data.''')

st.write('''My other project link : https://movieexplorationsuggestion-somnath.streamlit.app/''')

# Default Dataset URL
DEFAULT_DATA_URL = "https://github.com/Somnath342000/Sales_Analytics/blob/main/Somnath_techstore_Dummy_Data_(2023-24).csv"

# Try to load the default dataset
try:
    df = pd.read_csv(DEFAULT_DATA_URL)
    st.write("Using the default dataset for analysis.")
except Exception as e:
    st.write(f"Could not load the default dataset. Error: {e}")

# File Uploader for CSV file
uploaded_file2 = st.file_uploader("Upload Sales Data CSV File", type=["csv"], key="file2")

# If the file is uploaded, read it
if uploaded_file2:
    try:
        df = pd.read_csv(uploaded_file2)
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading the file: {e}")

# If no file is uploaded, the app will use the default dataset
if df is not None:
    # Data Overview
    st.subheader("📌 Data Preview")
    st.write(df.head())

    # Convert 'Date' column to datetime if exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        st.warning("⚠ No 'Date' column found in dataset.")

    # Customer Segmentation
    st.subheader("🔍 Customer Segmentation")
    if "Customer_ID" in df.columns:
        customer_data = df.groupby("Customer_ID").agg({"Sales": "sum", "Price": "mean"}).reset_index()
        customer_data.columns = ["Customer_ID", "Total_Spending", "Avg_Price"]

        # Apply K-Means Clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        customer_data["Segment"] = kmeans.fit_predict(customer_data[["Total_Spending", "Avg_Price"]])

        # Display Segmented Customers
        st.write(customer_data.head())

        # Visualize the Clusters
        st.subheader("📈 Cluster Visualization")
        fig, ax = plt.subplots()
        sns.scatterplot(x="Total_Spending", y="Avg_Price", hue="Segment", palette="viridis", data=customer_data, ax=ax)
        st.pyplot(fig)

    else:
        st.warning("⚠ No Customer ID column found in dataset.")

        # Customer Churn Prediction
        st.subheader("📉 Customer Churn Prediction")
        df["Last_Purchase"] = pd.to_datetime(df["Date"])
        df["Days_Since_Last_Purchase"] = (df["Last_Purchase"].max() - df["Last_Purchase"]).dt.days
        df["Churn"] = df["Days_Since_Last_Purchase"].apply(lambda x: 1 if x > 90 else 0)

        if "Customer_ID" in df.columns:
            churn_data = df.groupby("Customer_ID").agg({"Days_Since_Last_Purchase": "max", "Price": "mean", "Churn": "max"}).reset_index()
            X = churn_data[["Days_Since_Last_Purchase", "Price"]]
            y = churn_data["Churn"]

            # Train a Churn Prediction Model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)

            # Predict Churn Risk for a New Customer
            days_since_last_purchase = st.number_input("Days since last purchase", min_value=0)
            avg_purchase_price = st.number_input("Average purchase price", min_value=0)

            if st.button("Predict Churn Risk"):
                churn_risk = model.predict([[days_since_last_purchase, avg_purchase_price]])[0]
                churn_message = "❌ High Risk of Churn" if churn_risk == 1 else "✅ Low Risk of Churn"
                st.metric(label="Churn Prediction", value=churn_message)

            st.info("📌 Offer discounts to high-churn-risk customers to retain them!")
        else:
            st.warning("⚠ No Customer ID column found in dataset.")


        # Predictive Analytics: Future Sales Prediction
        st.subheader("📊 Predicting Future Sales")

        X = df[["Price", "Delivery_Time"]]
        y = df["Sales"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        future_price = st.number_input("Enter Future Price (₹)", min_value=0)
        future_delivery_time = st.number_input("Enter Future Delivery Time (Days)", min_value=1)

        if st.button("Predict Sales"):
            predicted_sales = model.predict([[future_price, future_delivery_time]])[0]
            st.metric(label="📊 Predicted Sales", value=round(predicted_sales))

        # Prescriptive Analytics: Suggested Actions
        st.subheader("🛠 Suggested Business Improvements")
        if future_price and future_delivery_time:
            if future_price > df["Price"].mean():
                st.warning("💡 Consider lowering prices to boost sales.")
            if future_delivery_time > df["Delivery_Time"].mean():
                st.warning("🚚 Faster delivery can increase conversions.")
           


        st.success("📌 Data-driven decisions help improve business performance!")

        #-----------------------------------

        # Sidebar Filters
        st.sidebar.header("🔍 Filter Data")
        selected_category = st.sidebar.selectbox("Select Product Category", ["All"] + df["Category"].unique().tolist())
        selected_region = st.sidebar.selectbox("Select Region", ["All"] + df["Region"].unique().tolist())
        selected_product = st.sidebar.selectbox("Select Product", ["All"] + df["Product"].unique().tolist())
        selected_month = st.sidebar.selectbox("Select Month", ["All"] + list(df['Date'].dt.strftime('%B').unique()))
        selected_year = st.sidebar.selectbox("Select Year", ["All"] + list(df['Date'].dt.year.unique()))
        selected_day = st.sidebar.selectbox("Select Day of the Week", ["All"] + list(df['Date'].dt.strftime('%A').unique()))

        # Apply Filters
        if selected_category != "All":
            df = df[df["Category"] == selected_category]
        if selected_region != "All":
            df = df[df["Region"] == selected_region]
        if selected_product != "All":
            df = df[df["Product"] == selected_product]
        if selected_month != "All":
            df['Month'] = df['Date'].dt.strftime('%B')  # Extract month name from the Date column
            df = df[df['Month'] == selected_month]
        if selected_year != "All":
            df['Year'] = df['Date'].dt.year  # Extract year from the Date column
            df = df[df['Year'] == selected_year]
        # Apply Day Filter
        if selected_day != "All":
            df['Day_of_Week'] = df['Date'].dt.strftime('%A')  # Extract the day of the week from the Date column
            df = df[df['Day_of_Week'] == selected_day]


        # Compute Metrics
        total_sales = df["Sales"].sum()
        total_customers = df["Customer_ID"].nunique()
        top_product = df.groupby("Product")["Sales"].sum().idxmax()
        low_product = df.groupby("Product")["Sales"].sum().idxmin()
        top_region = df.groupby("Region")["Sales"].sum().idxmax()
        low_region = df.groupby("Region")["Sales"].sum().idxmin()
        avg_delivery_time = df["Delivery_Time"].mean()
        correlation = df[["Price", "Sales"]].corr().iloc[0, 1]
        #--------------------------------------------------------
        
        # Group by State and sum sales
        state_sales = df.groupby('State')['Sales'].sum().reset_index()
        # Calculate percentage of total sales for each state
        state_sales['Sales_Percentage'] = (state_sales['Sales'] / total_sales) * 100
        top_5_state_sorted = state_sales.sort_values('Sales', ascending=False).head(5)

        # Group by City and sum sales
        city_sales = df.groupby('City')['Sales'].sum().reset_index()
        # Calculate percentage of total sales for each city
        city_sales['Sales_Percentage'] = (city_sales['Sales'] / total_sales) * 100
        top_5_cities_sorted = city_sales.sort_values('Sales', ascending=False).head(5)

        # Group by Product and sum sales
        product_sales = df.groupby('Product')['Sales'].sum().reset_index()
        # Calculate percentage of total sales for each product
        product_sales['Sales_Percentage'] = (product_sales['Sales'] / total_sales) * 100


        # Group by Customer_ID and sum the sales
        customer_sales = df.groupby('Customer_ID')['Sales'].sum().reset_index()
        # Sort by Sales in descending order
        customer_sales_sorted = customer_sales.sort_values('Sales', ascending=False)
        # Get the top 3 customers
        top_customers = customer_sales_sorted.head(5)

        #--------------------------------------------------------

        # Train Predictive Model
        X = df[["Price", "Delivery_Time"]]
        y = df["Sales"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predicted_sales = model.predict([[50000, 6]])[0]

        # Streamlit UI
        st.title("📊 Sales Analytics Dashboard")
        st.subheader(f"📌 Category: {selected_category} | Region: {selected_region} | Product: {selected_product} | Day : {selected_day} | Month : {selected_month} | Year : {selected_year}")

        # Display Analytics
        st.header("🔹 Descriptive Analytics")
        st.write(f"**Total Sales:** ₹{total_sales:,.2f}")
        st.write(f"**Total Customers:** {total_customers:,.2f}")
        st.write(f"🏆 **Best-Selling Product:** {top_product}")
        st.write(f"⚠ **Lowest-Selling Product:** {low_product}")
        st.write(f"🏆 **Best-Selling Region:** {top_region}")
        st.write(f"⚠ **Lowest-Selling Region:** {low_region}")


        st.header("Top 5 states wise sales : ")
        for index, row in top_5_state_sorted.iterrows():
            st.write(f"**{row['State']}:** ₹{row['Sales']:,.2f} ({row['Sales_Percentage']:.2f}%)")
        

        st.header("Top 5 cities wise sales : ")
        for index, row in top_5_cities_sorted.iterrows():
            st.write(f"**{row['City']}:** ₹{row['Sales']:,.2f} ({row['Sales_Percentage']:.2f}%)")

        st.header("Product wise sales : ")    
        for index, row in product_sales.iterrows():
            st.write(f"**{row['Product']}:** ₹{row['Sales']:,.2f} ({row['Sales_Percentage']:.2f}%)")

        st.header("Top 5 Customer based on Sales : ")
        for index, row in top_customers.iterrows():
            st.write(f"**Customer {row['Customer_ID']}:** ₹{row['Sales']:,.2f}")

        
        st.header("📉 Diagnostic Analytics")
        st.write(f"📊 **Price-Sales Correlation:** {correlation:.2f}")
        st.write(f"🚚 **Avg. Delivery Time:** {avg_delivery_time:.1f} days")

        st.header("📈 Predictive Analytics")
        st.write(f"📊 **Predicted Next Month's Sales:** ₹{predicted_sales:,.2f}")

        st.header("✅ Prescriptive Analytics")
        if predicted_sales < total_sales * 0.85:
            st.write("⚠ **Lower prices by 10%.**")
            st.write("⚠ **Improve delivery by 2 days.**")
            st.write("📢 **Offer promotions to high-risk customers.**")

        # Adding the Sales by Region Pie Chart
        st.header("📊 Sales by Region")
        region_sales = df.groupby("Region")["Sales"].sum()

        # Pie chart
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.pie(region_sales, labels=region_sales.index, autopct='%1.1f%%', colors=["red", "lightblue","pink", "yellow", "lightgreen"], startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Title and displaying the plot
        plt.title("Sales by Region")
        st.pyplot(plt)
        plt.clf()  # Clear the plot for the next one

        # Adding the Sales by Category Pie Chart
        st.header("📊 Sales by Category")
        category_sales = df.groupby("Category")["Sales"].sum()

        # Plotting the pie chart
        fig, ax = plt.subplots(figsize=(8, 8))  # Adjusted the size for a more circular pie chart
        ax.pie(category_sales, labels=category_sales.index, autopct='%1.1f%%', colors=["lightgreen", "orange", "pink", "blue"], startangle=90, wedgeprops={'edgecolor': 'black'})

        # Title
        plt.title("Sales by Category")

        # Display the plot in Streamlit
        st.pyplot(plt)
        plt.clf()  # Clear the plot for the next one

        # Adding the Sales by Product Pie Chart
        st.header("📊 Sales by Product")
        product_sales = df.groupby("Product")["Sales"].sum()

        # Plotting the pie chart
        fig, ax = plt.subplots(figsize=(8, 8))  # Adjusted size for a more circular pie chart
        ax.pie(product_sales, 
            labels=product_sales.index, 
            autopct='%1.1f%%',  # Display percentages on the slices
            colors=["yellow","orange", "grey", "lightgreen", "red", "pink","skyblue",], 
            startangle=90, 
            wedgeprops={'edgecolor': 'black'})

        # Title
        plt.title("Sales by Product")

        # Display the plot in Streamlit
        st.pyplot(plt)
        plt.clf()  # Clear the plot for the next one

        #-----------------------------

        # Selecting features and target variable
        X = df[["Price", "Delivery_Time"]]  # Features used during training
        y = df["Sales"]

        # Splitting data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training Linear Regression Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Simulating price drop by 10%
        df["New_Price"] = df["Price"] * 0.9

        # Predicting sales using the original Price (not New_Price)
        df["Predicted_Sales"] = model.predict(df[["Price", "Delivery_Time"]])

        # Predicting Sales for the Test Data
        y_pred = model.predict(X_test)

        # Evaluating Model Accuracy
        mae = mean_absolute_error(y_test, y_pred)

        # Streamlit Output
        st.header("📊 Sales Analysis")
        st.write(f"**Mean Absolute Error (MAE) on Test Data:** {mae:.2f}")

        # Scatter plot for Delivery Time vs Sales using Seaborn
        st.header("📊 Delivery Time vs Sales (Scatter Plot)")

        # Create the scatter plot
        plt.figure(figsize=(5, 3))
        sns.scatterplot(x=df["Delivery_Time"], y=df["Sales"])

        # Adding titles and labels
        plt.title("Delivery Time vs Sales")
        plt.xlabel("Delivery Time (Days)")
        plt.ylabel("Sales")

        # Display the plot in Streamlit
        st.pyplot(plt)
        
        # Descriptive Analytics: Sales Trend
        st.subheader("📈 Sales Trend Over Time")
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day  # Day of the month
        df['Weekday'] = df['Date'].dt.weekday  # Weekday (0=Monday, 6=Sunday)

        # Plot average sales by the day of the week (Weekday)
        df.groupby('Weekday')['Sales'].mean().plot(kind="line", marker='o', figsize=(7, 3))

        # Add titles and labels
        plt.title("Average Sales by Weekday")
        plt.xlabel("Weekday")
        plt.ylabel("Average Sales")
        plt.xticks(ticks=range(7), labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        st.pyplot(plt)

        # Diagnostic Analytics: Price vs Sales
        st.subheader("🔍 Price vs Sales Analysis")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(x=df["Price"], y=df["Sales"], hue=df["Product"], ax=ax)
        st.pyplot(fig)

        # Visualizing Customer Segmentation
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.scatterplot(x="Avg_Price", y="Total_Spending", hue="Segment", data=customer_data, palette="coolwarm", ax=ax)
        st.pyplot(fig)
        st.info("📌 High-spending customers should be targeted with exclusive offers!")


        from statsmodels.tsa.statespace.sarimax import SARIMAX
        import warnings
        warnings.filterwarnings('ignore')

        # Assuming df is your DataFrame with a 'Date' column
        # Define the function to extract month and year
        def Function_get_month(date):
            return date.month

        def Function_get_year(date):
            return date.year

        # Sample DataFrame setup (Make sure to replace this with your actual DataFrame)
        # df = pd.read_csv("your_data.csv")  # Load your dataset here
        # For demonstration, let's assume `df` has 'Date' and 'Quantity' columns.

        # Example DataFrame for demonstration:
        data = {
            'Date': pd.date_range(start="2015-01-01", periods=48, freq='M'),
            'Quantity': np.random.randint(100, 500, 48)
        }
        df = pd.DataFrame(data)

        # Extract Month and Year columns
        df['Month'] = df['Date'].apply(Function_get_month)
        df['Year'] = df['Date'].apply(Function_get_year)

        # Aggregating the sales quantity for each month for all categories
        SalesQuantity = pd.crosstab(columns=df['Year'],
                                    index=df['Month'],
                                    values=df['Quantity'],
                                    aggfunc='sum').melt()['value']

        # Create month labels for the x-axis
        MonthNames = ['Jan','Feb','Mar','Apr','May', 'Jun', 'Jul', 'Aug', 'Sep','Oct','Nov','Dec'] * 4

        # Sidebar for the user to input model parameters
        st.sidebar.header("Sales Forecasting Parameters")
        start_month = st.sidebar.selectbox("Select start month for forecast", options=MonthNames)
        num_months = st.sidebar.slider("Select the number of months to forecast", min_value=1, max_value=12, value=6)

        # Plotting the sales data and forecasting
        st.header("📊 Sales Quantity Forecasting")

        # Plot the time series data of sales quantity
        fig, ax = plt.subplots(figsize=(16, 8))
        SalesQuantity.plot(kind='line', ax=ax, title='Total Sales Quantity per month')

        # Set x-axis labels
        plotLabels = plt.xticks(np.arange(0, len(SalesQuantity), 1), MonthNames, rotation=30)

        # Train the SARIMAX model on the full dataset
        SarimaxModel = SARIMAX(SalesQuantity, order=(0, 1, 10), seasonal_order=(1, 0, 0, 12))
        SalesModel = SarimaxModel.fit()

        # Forecast the next 'num_months' months
        forecast = SalesModel.predict(start=0,
                                    end=len(SalesQuantity) + num_months - 1,  # Adjust for total forecast length
                                    typ='levels').rename('Forecast')

        # Show forecasted values in Streamlit
        st.write("Forecast for the next months:")
        forecast_values = forecast[-num_months:]
        st.write(forecast_values)

        # Plot the forecast
        fig, ax = plt.subplots(figsize=(20, 7))
        SalesQuantity.plot(ax=ax, legend=True, title='Time Series Sales Forecasts')
        forecast.plot(ax=ax, legend=True)

        # Extend the MonthNames to match the length of the full data (original + forecasted)
        MonthNames_extended = MonthNames + MonthNames[:num_months]  # Adjust the number of months for the forecast

        # Adjust the x-ticks to include all months (original + forecasted)
        plt.xticks(np.arange(0, len(forecast), 1), MonthNames_extended, rotation=30)

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Measure the Training accuracy of the model
        MAPE = np.mean(abs(SalesQuantity - forecast[:len(SalesQuantity)]) / SalesQuantity) * 100
        st.write('#### Accuracy of model:', round(100 - MAPE, 2), '% ####')

        # Add month names on the x-axis for the forecast
        MonthNames_extended = MonthNames + MonthNames[:num_months]
        plotLabels = plt.xticks(np.arange(0, len(forecast), 1), MonthNames_extended, rotation=30)




        # PDF Report Function
        def generate_pdf_report():
            filename = f"Sales_Report_{datetime.date.today()}.pdf"
            c = canvas.Canvas(filename, pagesize=letter)

            c.drawString(100, 750, "📊 Somnath Techstore Sales Analytics Report")
            c.drawString(100, 730, f"Date: {datetime.date.today()} - Category: {selected_category}, Region: {selected_region},  Product: {selected_product},  Day : {selected_day} ,Month : {selected_month} , Year : {selected_year}")

            c.drawString(100, 700, "🔹 Descriptive Analytics")
            c.drawString(120, 680, f"Total Sales: ₹{total_sales:,.2f}")
            c.drawString(120, 660, f"Best-Selling Product: {top_product}")
            c.drawString(120, 640, f"Lowest-Selling Product: {low_product}")

            c.drawString(100, 610, "📉 Diagnostic Analytics")
            c.drawString(120, 590, f"Price-Sales Correlation: {correlation:.2f}")
            c.drawString(120, 570, f"Avg. Delivery Time: {avg_delivery_time:.1f} days")

            c.drawString(100, 540, "📈 Predictive Analytics")
            c.drawString(120, 520, f"Predicted Next Month's Sales: ₹{predicted_sales:,.2f}")

            c.drawString(100, 490, "✅ Prescriptive Analytics")
            c.drawString(120, 470, "- Lower prices by 10%.")
            c.drawString(120, 450, "- Improve delivery by 2 days.")
            c.drawString(120, 430, "- Offer promotions to high-risk customers.")

            c.save()

            return filename

        st.subheader("📄 Download Sales Report")
        if st.button("Generate PDF Report"):
            report_file = generate_pdf_report()
            # Provide the PDF as a downloadable file
            with open(report_file, "rb") as file:
                st.download_button("Download PDF", file, file_name=report_file)

        # Twilio Setup
        TWILIO_SID = "your_account_sid"
        TWILIO_AUTH_TOKEN = "your_auth_token"
        TWILIO_PHONE = "your_twilio_number"
        ADMIN_PHONE = "your_admin_phone_number"

        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

        def send_alert_sms(message):
            client.messages.create(body=message, from_=TWILIO_PHONE, to=ADMIN_PHONE)

        def send_alert_whatsapp(message):
            client.messages.create(
                body=message,
                from_="whatsapp:" + TWILIO_PHONE,
                to="whatsapp:" + ADMIN_PHONE
            )

        # Trigger Alerts
        #if total_sales < df["Sales"].sum() * 0.85:
        #    alert_msg = "⚠ URGENT: Sales have dropped by more than 15%! Immediate action needed."
         #   st.warning(alert_msg)
         #   send_alert_sms(alert_msg)
         #   send_alert_whatsapp(alert_msg)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("⚠ Please upload a file to get started!")
