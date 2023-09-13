import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title='Python Data Science Cheat Sheet',
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main function
def main():
    st.title('Python Machine Learning and Data Science Cheat Sheet')
    
    st.sidebar.header('Select a Library')
    selected_library = st.sidebar.radio('Choose a library:', ['Pandas', 'NumPy', 'Matplotlib','Plotly','Tensorflow','Scikit','Seaborn'], key='my_unique_key')

    if selected_library == 'Pandas':
        pandas_cheat_sheet()
    elif selected_library == 'NumPy':
        numpy_cheat_sheet()
    # elif selected_library == 'Matplotlib':
    #     matplotlib_cheat_sheet()
    # elif selected_library == 'Scikit-learn':
    #     scikit_learn_cheat_sheet()
    # elif selected_library == 'Seaborn':
    #     seaborn_cheat_sheet()
    # elif selected_library == 'Plotly':
    #     plotly_cheat_sheet()
    # elif selected_library == 'Tensorflow':
    #     tensorflow_cheat_sheet()
    




# Matplotlib Cheat Sheet

def matplotlib_cheat_sheet():
    st.header('Matplotlib Cheat Sheet')

    # Add Matplotlib content here
    st.write("Matplotlib is a popular library for data visualization.")
    st.code("import matplotlib.pyplot as plt")
    st.write("Create a simple plot:")
    st.code("plt.plot([1, 2, 3, 4])\nplt.show()")

    # Customizing plots
    st.subheader('Customizing Plots')
    st.write("Customize your plots with labels, titles, and styles.")
    st.code("plt.xlabel('X-axis')\nplt.ylabel('Y-axis')\nplt.title('My Plot')")
    st.code("plt.plot(x, y, label='Line 1', linestyle='--', color='b')\nplt.legend()")

    # Creating different types of plots
    st.subheader('Creating Different Types of Plots')
    st.write("Explore different types of plots like bar charts, histograms, and scatter plots.")
    st.code("plt.bar(x, y)")
    st.code("plt.hist(data, bins=10)")
    st.code("plt.scatter(x, y)")

# Continue the Cheat Sheet for Machine Learning and Data Science Libraries

# Add a section for Pandas Cheat Sheet
def pandas_cheat_sheet():
    st.header('Pandas Cheat Sheet')

    # Add Pandas content here
    st.write("Pandas is a powerful library for data manipulation and analysis.")
    st.code("import pandas as pd")
    st.write("Read data from a CSV file:")
    st.code("df = pd.read_csv('data.csv')")

    # Common DataFrame operations
    st.subheader('Common DataFrame Operations')
    st.write("Perform common operations on DataFrames.")
    st.code("df.head(5)  # Display the first 5 rows")
    st.code("df.describe()  # Summary statistics")
    st.code("df['column_name'].unique()  # Unique values")
    st.code("df.groupby('category').mean()  # Group by and aggregate")

# Add a section for NumPy Cheat Sheet
def numpy_cheat_sheet():
    st.header('NumPy Cheat Sheet')

    # Add NumPy content here
    st.write("NumPy is the fundamental package for scientific computing with Python.")
    st.code("import numpy as np")
    st.write("Create a NumPy array:")
    st.code("arr = np.array([1, 2, 3, 4, 5])")

    # Array operations
    st.subheader('Array Operations')
    st.write("Perform array operations and computations.")
    st.code("np.mean(arr)  # Mean of the array")
    st.code("np.max(arr)  # Maximum value")
    st.code("np.min(arr)  # Minimum value")
    st.code("np.sum(arr)  # Sum of elements")

# # Add a section for scikit-learn Cheat Sheet
# def scikit_learn_cheat_sheet():
#     st.header('scikit-learn Cheat Sheet')

#     # Add scikit-learn content here
#     st.write("scikit-learn is a machine learning library for Python.")
#     st.code("from sklearn.model_selection import train_test_split")
#     st.write("Split data into training and testing sets:")
#     st.code("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)")

#     # Common machine learning tasks
#     st.subheader('Common Machine Learning Tasks')
#     st.write("Explore common machine learning tasks and techniques.")
#     st.code("from sklearn.linear_model import LinearRegression")
#     st.code("model = LinearRegression()")
#     st.code("model.fit(X_train, y_train)  # Train the model")
#     st.code("y_pred = model.predict(X_test)  # Make predictions")

# # Add a section for seaborn Cheat Sheet
# def seaborn_cheat_sheet():
#     st.header('Seaborn Cheat Sheet')

#     # Add Seaborn content here
#     st.write("Seaborn is a data visualization library based on Matplotlib.")
#     st.code("import seaborn as sns")
#     st.write("Create a scatter plot:")
#     st.code("sns.scatterplot(x='feature1', y='feature2', data=df)")

#     # Customizing plots
#     st.subheader('Customizing Plots')
#     st.write("Customize Seaborn plots with various options.")
#     st.code("sns.set(style='whitegrid')")
#     st.code("sns.pairplot(df, hue='target')")


# # Add a section for Plotly Cheat Sheet
# def plotly_cheat_sheet():
#     st.header('Plotly Cheat Sheet')

#     # Add Plotly content here
#     st.write("Plotly is an interactive data visualization library.")
#     st.code("import plotly.express as px")
#     st.write("Create an interactive scatter plot:")
#     st.code("fig = px.scatter(df, x='feature1', y='feature2')")
#     st.code("fig.update_layout(title='Scatter Plot')")

#     # Interactive features
#     st.subheader('Interactive Features')
#     st.write("Add interactive elements to Plotly plots.")
#     st.code("fig.add_trace(go.Box(y=data))")
#     st.code("fig.update_xaxes(type='log')")

# # Add a section for TensorFlow Cheat Sheet
# def tensorflow_cheat_sheet():
#     st.header('TensorFlow Cheat Sheet')

#     # Add TensorFlow content here
#     st.write("TensorFlow is an open-source machine learning framework.")
#     st.code("import tensorflow as tf")
#     st.write("Create a neural network model:")
#     st.code("model = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)), tf.keras.layers.Dense(1)])")
#     st.write("Compile and train the model:")
#     st.code("model.compile(optimizer='adam', loss='mean_squared_error')")
#     st.code("model.fit(X_train, y_train, epochs=10)")

#     # TensorBoard integration
#     st.subheader('TensorBoard Integration')
#     st.write("Visualize TensorFlow model training with TensorBoard.")
#     st.code("tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')")
#     st.code("model.fit(X_train, y_train, epochs=10, callbacks=[tensorboard_callback])")

# Call the cheat sheet functions for each library
# matplotlib_cheat_sheet()
# plotly_cheat_sheet()
# tensorflow_cheat_sheet()

# # Call the cheat sheet functions for each library
# pandas_cheat_sheet()
# numpy_cheat_sheet()
# scikit_learn_cheat_sheet()
# seaborn_cheat_sheet()

# Run the app
if __name__ == '__main__':
    main()
