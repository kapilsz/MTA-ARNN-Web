from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
# import tensorflow as tf
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.graph_objs import *
import time 
import sqlite3

##### for tf model #####################
import sys
import tensorflow as tf
import pickle 
from Criteo.arnn_config import config1
# load it
with open(f'config_class_arnn_criteo.pickle', 'rb') as file2:
    Config = pickle.load(file2)

from Criteo.arnn_tf_function import *
from Criteo.arnn_function import *

app = Flask(__name__)

medicine_type = 'Product1'

df = pd.DataFrame()
df['Channel'] = ['Facebook', 'Google','IG','MAIL','TV']
df['Score'] = [0.8, 0.6, 0.4, 0.2, 0.1]
df['Cost'] = [100, 200, 300, 400, 500]
df.to_csv('data.csv',index=False)

layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

def check_userpass(username, password):
    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # Execute the query to check if the username and password are present in the 'userpass' table
    c.execute("SELECT COUNT(*) FROM userpass WHERE username=? AND password=?", (username, password))
    
    # Fetch the result of the query
    result = c.fetchone()[0]
    # Close the database connection
    conn.close()
    
    # Return True if the query result is greater than 0, else return False
    return True if result > 0 else False


def process_file(file=None):
    # Read CSV file into a Pandas dataframe
    # df = pd.read_csv(file)

    # TODO: Process the dataframe using TensorFlow model to calculate attribution
    print(2)
    X, y, key_ids, costs = data_preprocess_criteo(file)
    print(3)
    # Load the TensorFlow model
    model_attn = tf.keras.models.load_model('Model/model_attn_weights_c.h5', custom_objects={'CustomAttention':CustomAttention,'XMI_lay':XMI_lay})
    print(4)
    # Make predictions using the model
    predictions = attribution_criteo(X, y, costs, model_attn)
    print(5)
    predictions.to_csv('prediction.csv',index=False)
    print(6)
    # For now, just return the original dataframe with three additional columns
    df = pd.DataFrame()
    df['Channel'] = ['Facebook', 'Google','IG','MAIL','TV']
    df['Score'] = [0.8, 0.6, 0.4, 0.2, 0.1]
    df['Cost'] = [100, 200, 300, 400, 500]
    return df

def create_graph(df):
    # Create a Plotly graph
    data = [go.Bar(x=df['Channel'], y=df['Score'], name='Score'),
            go.Scatter(x=df['Channel'], y=df['Cost'], name='Cost')]
    layout = go.Layout(title='Attribution', xaxis_title='Channel', yaxis_title='Score/Cost')
    fig = go.Figure(data=data, layout=layout)
    graph = plot(fig, output_type='div')
    return graph

@app.route('/', methods=['GET', 'POST'])
def login():
    message = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        usercheck = check_userpass(username,password)
        if usercheck:
            return redirect(url_for('home'))
        else:
            error = 'Please Check Your Credentials'
            return render_template('login.html', error=error)
    return render_template('login.html', error=None)

@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        global medicine_type
        medicine_type = request.form['medicine_type']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        region = '(' + str(request.form.getlist('region'))[1:-1] + ')'
        user_type = '(' + str(request.form.getlist('user_type'))[1:-1] + ')'
        sample_size = request.form['sample_size']

        # read from database
        conn = sqlite3.connect('database.db')
        query = f"select * from {medicine_type} where first_interaction >= {start_date} AND region in {region} AND user_type in {user_type} limit {sample_size};"         # first_interaction < {end_date} AND 
        sampled_data = pd.read_sql(query,conn)
        conn.close()
        if sampled_data.shape[0] > 9:
            print(1)
            df = process_file(sampled_data)
            return redirect(url_for('attr_table'))
        else:
            error = 'Please Select Sufficient Data'
            return render_template('home.html', error=error)
    elif request.method == 'GET':
        return render_template('home.html', error=None)
    return render_template('home.html', error=None)

@app.route('/requestt', methods=['GET', 'POST'])
def requestt():    
    return render_template('requestt.html')

@app.route('/attr_table')
def attr_table():
    # Read CSV file into a Pandas dataframe
    df = pd.read_csv('prediction.csv')
    df = df[df.frequency>1].iloc[:15]
    
    rows = df.to_numpy().tolist()
    return render_template('results.html', rows=rows, medicine_type=medicine_type)

@app.route('/bar_chart')
def bar_chart():
    # Read CSV file into a Pandas dataframe
    df = pd.read_csv('data.csv')
    df = pd.read_csv('prediction.csv')
    df = df[df.frequency>1].iloc[:15]

    # Create a bar chart
    data = [go.Bar(x=df['Channel'], y=df['Score'], name='Score'),
            go.Scatter(x=df['Channel'], y=df['Cost'], name='Cost')]
    layout = go.Layout(title='Attribution - Bar Chart', xaxis_title='Channel', yaxis_title='Score/Cost')#,paper_bgcolor='rgba(0.1,0.05,0.011,0.3)')
    fig = go.Figure(data=data, layout=layout)
    chart = plot(fig, output_type='div')

    return render_template('bar_chart.html', chart=chart, medicine_type=medicine_type)

@app.route('/pie_chart')
def pie_chart():
    # Read CSV file into a Pandas dataframe
    df = pd.read_csv('data.csv')
    df = pd.read_csv('prediction.csv')
    df = df[df.frequency>1].iloc[:15]

    # Create a pie chart
    data = [go.Pie(labels=df['Channel'], values=df['Score'], hole=0.3)]
    layout = go.Layout(title='Attribution - Pie Chart')
    fig = go.Figure(data=data, layout=layout)
    chart = plot(fig, output_type='div')

    return render_template('pie_chart.html', chart=chart, medicine_type=medicine_type)

@app.route('/heatmap')
def heatmap():
    # Read CSV file into a Pandas dataframe
    df = pd.read_csv('data.csv')
    df = pd.read_csv('prediction.csv')
    df = df[df.frequency>1].iloc[:15]

    # Create a heatmap
    data = [go.Heatmap(x=df['Channel'], y=df['Score'], z=df['Cost'])]
    layout = go.Layout(title='Attribution - Heatmap', xaxis_title='Channel', yaxis_title='Score')
    fig = go.Figure(data=data, layout=layout)
    chart = plot(fig, output_type='div')

    return render_template('heatmap.html', chart=chart, medicine_type=medicine_type)

from flask import send_file

@app.route('/download')
def download_file():
    # Get the path to the file you want to download
    file_path = 'data.csv'
    # Return the file for download using the send_file() function
    return send_file(file_path, as_attachment=True)

@app.route('/download/<graph_name>')
def download_graph(graph_name):
    graph_path = os.path.join(app.root_path, 'static', graph_name)
    return send_file(graph_path, as_attachment=True)



if __name__ == '__main__':
    app.run(debug=True, port=8000)


# import sys
# import tensorflow as tf
# import pickle 
# from Criteo.arnn_config import config1
# # load it
# with open(f'config_class_arnn_criteo.pickle', 'rb') as file2:
#     Config = pickle.load(file2)

# from Criteo.arnn_tf_function import *
# from Criteo.arnn_function import *

# # sys.path.insert(0, 'C:/Users/hp/Downloads/MTP/web_app/Criteo')


# if file is not None:
#     # Load the CSV file into a pandas dataframe
#     data = pd.read_csv(file)

#     # Preprocess the data for the TensorFlow model
#     # ...
#     X, y, key_ids = data_preprocess_criteo(data)

#     # Load the TensorFlow model
#     model_attn = tf.keras.models.load_model('Model/model_attn_weights_c.h5', custom_objects={'CustomAttention':CustomAttention,'XMI_lay':XMI_lay})

#     # Make predictions using the model
#     predictions = attribution_criteo(X, y, model_attn)
#     predictions.sort_values(by=['mean_weight'],ascending=False,inplace=True)
#     # Display the attribution results in a table
#     st.write("Attribution Results:")
#     # attribution_df = pd.DataFrame(predictions, columns=["Channel 1", "Channel 2", "Channel 3", "Channel 4"])
#     st.dataframe(predictions)#.iloc[:20])
