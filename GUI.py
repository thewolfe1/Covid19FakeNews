import time

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Output, Input
from Utils import *
import dash_bootstrap_components as dbc




index=-1
num=0
load_data()
data = read_data()
lstm=predict_lstm(data)
gru=predict_gru(data)
lr=predict_LR(data)
svc=predict_svc(data)
nb=predict_NB(data)
mnb=predict_MNB(data)
gnb=predict_GNB(data)
rf=predict_RF(data)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',dbc.themes.GRID]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)




colors = {
    'background': '#111111',
    'text': '#d1d1e0',
}
df = pd.DataFrame({"Model": ['svc', 'Logistic Regression', 'Bernoulli Naive Bayes', 'Multinomial Naive Bayes', 'Gaussian Naive Bayes','random forest','LSTM','GRU'],
                   "Accuracy": [0.9936102236421726, 0.9297124600638977, 0.8865814696485623, 0.8769968051118211, 0.7012779552715654,0.8753993610223643,0.8530351519584656,0.7451757001876831]
                      })



fig = px.bar(df, x="Model", y='Accuracy', barmode="group",width=600, height=400)

fig.update_layout(plot_bgcolor=colors['background'],
                  paper_bgcolor=colors['background'],
                  font_color=colors['text']
                  )

df2 = pd.DataFrame({"Model": ['svc', 'Logistic Regression', 'Bernoulli Naive Bayes', 'Multinomial Naive Bayes', 'Gaussian Naive Bayes','random forest','LSTM','GRU'],
                   "F1-score": [0.99, 0.93,  0.89 , 0.87, 0.70,0.88,0.85 ,0.74]
                      })

fig2 = px.bar(df2, x="Model", y='F1-score', barmode="group",width=600, height=400)

fig2.update_layout(plot_bgcolor=colors['background'],
                  paper_bgcolor=colors['background'],
                  font_color=colors['text']
                  )

df3 = pd.DataFrame({"Model": ['svc', 'Logistic Regression', 'Bernoulli Naive Bayes', 'Multinomial Naive Bayes', 'Gaussian Naive Bayes','random forest','LSTM','GRU'],
                   "mse": [0.006389776357827476, 0.07028753993610223,  0.1134185303514377 , 0.12300319488817892, 0.2987220447284345,0.12460063897763578,0.3692074716091156 ,0.3442157208919525]
                      })

fig3 = px.bar(df3, x="Model", y='mse', barmode="group",width=600, height=400)

fig3.update_layout(plot_bgcolor=colors['background'],
                  paper_bgcolor=colors['background'],
                  font_color=colors['text']
                  )

app.layout = html.Div(children=[
    html.H1(children='Covid-19 Fake News Detection', style={
            'textAlign': 'center',
            'color': colors['text']
        }),


    html.Div(
        [
            html.P(id="update_p"),
            dcc.Interval(id="refresh", interval=1 * 1000, n_intervals=0),
        ],style={
            'textAlign': 'center',
            'color': colors['text'],
            'family':'Courier New, monospace',
        }
    ),

    dbc.Row(
        [
            dbc.Col(dcc.Graph(
        id='accuracy',
        figure=fig,

    )),
            dbc.Col(dcc.Graph(
        id='f1-score',
        figure=fig2,

    )),
            dbc.Col(dcc.Graph(
        id='mse',
        figure=fig3,

    )),
        ]

    )
])


@app.callback(Output("update_p", "children"), [Input("refresh", "n_intervals")])
def update(n_intervals):
    global num
    global index
    global data
    time.sleep(0.8)
    if num == n_intervals:
        num+=5
        index+=1
    return dcc.Markdown(
        "{}\n \n \n  svc: {}\n  Logistic Regression: {}\n  Naive Bayes: {}\n  Multinomial Naive Bayes: {}\n  Gaussian Naive Bayes: {}\n  random forest: {}\n  LSTM: {}\n  GRU: {} ".format
            (data[index],num_to_bool(svc[index]),num_to_bool(lr[index]),num_to_bool(nb[index]),num_to_bool(mnb[index]),num_to_bool(gnb[index]),num_to_bool(rf[index])
             ,num_to_bool(lstm[index]),num_to_bool(gru[index])),
        style={"white-space": "pre"}
    )
    #return "tweet: {} \n lstm: {}".format(data[index],predictions[index])



if __name__ == '__main__':
    app.run_server(debug=True)
