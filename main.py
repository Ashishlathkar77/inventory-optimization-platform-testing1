import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
import plotly.express as px
from fpdf import FPDF
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Access the API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if the API key is loaded
if OPENAI_API_KEY is None:
    st.error("API Key is missing. Please check your .env file.")

# Initialize OpenAI
llm = OpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY)

# Streamlit UI
st.title("AI-Powered Inventory Optimization")
st.header("Inventory Dashboard")

# Session state to hold uploaded data
if 'data' not in st.session_state:
    st.session_state.data = None

# File uploader for CSV files
uploaded_file = st.file_uploader("Upload CSV with inventory data", type="csv")
if uploaded_file is not None:
    try:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.write("Data loaded:")
        st.write(st.session_state.data)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.write("Please check if the file is a valid CSV.")

def generate_pdf(data, recommendations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add company logo as watermark
    pdf.image("/workspaces/inventory-optimization-platform-testing1/885kxu3w_.png", x=10, y=8, w=100, h=20)  # Adjust path and size

    # Add insights
    pdf.cell(200, 10, txt="Inventory Insights", ln=True)
    pdf.multi_cell(0, 10, txt=data.describe().to_string(index=True))
    
    # Add recommendations
    pdf.cell(200, 10, txt="Recommendations", ln=True)
    pdf.multi_cell(0, 10, txt=recommendations)

    # Save to BytesIO
    pdf_output = BytesIO()
    pdf.output(dest='S').encode('latin1')  # 'S' means return the document as a string
    pdf_output.write(pdf.output(dest='S').encode('latin1'))  # Write the PDF bytes to BytesIO
    pdf_output.seek(0)  # Move to the beginning of the BytesIO buffer
    return pdf_output.getvalue()  # Return bytes

# Forecasting and Recommendations
st.header("Forecasting and Recommendations")
if st.button("Get Recommendations"):
    if st.session_state.data is None:
        st.warning("Please upload inventory data before getting recommendations.")
    else:
        chunk_size = 50  # Define the size of each chunk
        recommendations_list = []  # List to store recommendations
        
        # Process the DataFrame in chunks
        for start in range(0, len(st.session_state.data), chunk_size):
            chunk = st.session_state.data.iloc[start:start + chunk_size]
            chunk_summary = chunk.describe(include='all').to_string()  # Summarize chunk
            
            prompt = (
                "Based on the following inventory data summary, provide restocking recommendations:\n"
                + chunk_summary
            )
            
            try:
                recommendations = llm(prompt)
                recommendations_list.append(recommendations)
            except Exception as e:
                st.error(f"Error getting recommendations for chunk starting at row {start}: {e}")

        # Combine all recommendations
        all_recommendations = "\n".join(recommendations_list)
        st.write("Recommendations:")
        st.write(all_recommendations)

        # Generate PDF
        pdf_output = generate_pdf(st.session_state.data, all_recommendations)

        # Download PDF
        st.download_button("Download PDF Report", pdf_output, "inventory_report.pdf", mime='application/pdf')



# Q&A Section
st.header("Ask Questions About Your Inventory Data")
user_question = st.text_input("Enter your question:")

if user_question:
    if st.session_state.data is None:
        st.warning("Please upload inventory data before asking questions.")
    else:
        try:
            # Summarize the inventory data
            data_summary = st.session_state.data.describe(include='all').to_string()
            
            # Construct the prompt
            prompt = (
                "Based on the following inventory data summary, answer the question:\n"
                + data_summary + "\n\nQuestion: " + user_question
            )

            # Call the language model
            answer = llm(prompt)
            st.write("Answer:")
            st.write(answer)
        
        except Exception as e:
            st.error(f"Error while processing the question: {e}")


# Visualization Section
st.header("Visualize Inventory Data")
if st.session_state.data is not None:
    columns = st.session_state.data.columns.tolist()
    selected_columns = st.multiselect("Select columns to visualize", columns)

    if selected_columns:
        # Separate selected columns into numeric and categorical
        numeric_columns = st.session_state.data[selected_columns].select_dtypes(include=['number']).columns.tolist()
        categorical_columns = st.session_state.data[selected_columns].select_dtypes(include=['object']).columns.tolist()

        if len(selected_columns) == 1:
            # Single numeric column histogram
            col = selected_columns[0]
            if col in numeric_columns:
                fig = px.histogram(st.session_state.data, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig)
            elif col in categorical_columns:
                # Single categorical column count plot
                fig = px.histogram(st.session_state.data, x=col, title=f"Count of {col}")
                st.plotly_chart(fig)

        elif len(selected_columns) == 2:
            # Two selected features
            if len(numeric_columns) == 2:
                # 2D scatter plot
                fig = px.scatter(st.session_state.data, 
                                 x=numeric_columns[0], 
                                 y=numeric_columns[1], 
                                 title=f"{numeric_columns[0]} vs {numeric_columns[1]}")
                st.plotly_chart(fig)
            elif len(categorical_columns) == 2:
                # Count plot for two categorical columns
                fig = px.histogram(st.session_state.data, x=categorical_columns[0], color=categorical_columns[1],
                                   title=f"Count of {categorical_columns[0]} by {categorical_columns[1]}")
                st.plotly_chart(fig)
            elif len(numeric_columns) == 1 and len(categorical_columns) == 1:
                # Bar plot for one numeric and one categorical column
                fig = px.bar(st.session_state.data, 
                             x=categorical_columns[0], 
                             y=numeric_columns[0], 
                             title=f"{numeric_columns[0]} by {categorical_columns[0]}")
                st.plotly_chart(fig)

        elif len(selected_columns) >= 3:
            # Three selected features
            cat_col = [col for col in selected_columns if col in categorical_columns]
            num_cols = [col for col in selected_columns if col in numeric_columns]

            if cat_col and len(num_cols) == 2:
                # Create a bar plot with product_name as categories and two numeric features as bars
                fig = px.bar(st.session_state.data, 
                             x=cat_col[0],  # Categorical column
                             y=num_cols,    # Numeric columns
                             title="Quantity and Sales Last Month by Product",
                             barmode='group')
                st.plotly_chart(fig)

            else:
                st.warning("Please select one categorical column and two numeric columns.")
        else:
            st.warning("Please select at least one column to visualize.")

# Debugging: Check API Key and uploaded data
if OPENAI_API_KEY:
    st.write("API Key loaded successfully.")
else:
    st.error("API Key is missing.")