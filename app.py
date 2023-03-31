from flask import Flask, render_template, request, redirect, url_for, session, send_file
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
app.secret_key = 'kapil_shyam_zadpe_21IM60R16_IITKgp23'

medicine_type = 'Product1'

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

    X, y, key_ids, costs = data_preprocess_criteo(file)
    model_attn = tf.keras.models.load_model('Model/model_attn_weights_c.h5', custom_objects={'CustomAttention':CustomAttention,'XMI_lay':XMI_lay})
    predictions = attribution_criteo(X, y, costs, model_attn)
    predictions.to_csv('prediction.csv',index=False)
    return True

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
            session['username'] = username
            return redirect(url_for('home'))
        else:
            error = 'Please Check Your Credentials'
            return render_template('login.html', error=error)
    return render_template('login.html', error=None)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'username' not in session:
        return redirect(url_for('login'))

    elif request.method == 'POST':
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
    if 'username' not in session:
        return redirect(url_for('login'))
    # Read CSV file into a Pandas dataframe
    df = pd.read_csv('prediction.csv')
    df = df[df.frequency>1].iloc[:15]
    
    rows = df.to_numpy().tolist()
    return render_template('results.html', rows=rows, medicine_type=medicine_type)

@app.route('/bar_chart')
def bar_chart():
    if 'username' not in session:
        return redirect(url_for('login'))
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
    if 'username' not in session:
        return redirect(url_for('login'))
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
    if 'username' not in session:
        return redirect(url_for('login'))
    df = pd.read_csv('prediction.csv')
    df = df[df.frequency>1].iloc[:15]

    # Create a heatmap
    data = [go.Heatmap(x=df['Channel'], y=df['Score'], z=df['Cost'])]
    layout = go.Layout(title='Attribution - Heatmap', xaxis_title='Channel', yaxis_title='Score')
    fig = go.Figure(data=data, layout=layout)
    chart = plot(fig, output_type='div')

    return render_template('heatmap.html', chart=chart, medicine_type=medicine_type)

@app.route('/download')
def download_file():
    # Get the path to the file you want to download
    file_path = 'prediction.csv'
    # Return the file for download using the send_file() function
    return send_file(file_path, as_attachment=True)

@app.route('/download/<graph_name>')
def download_graph(graph_name):
    graph_path = os.path.join(app.root_path, 'static', graph_name)
    return send_file(graph_path, as_attachment=True)



if __name__ == '__main__':
    app.run(debug=True, port=8000)