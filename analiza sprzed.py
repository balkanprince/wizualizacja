# app.py - Wersja pełna z autoryzacją i 3 zakładkami
import pandas as pd
import datetime as dt
import os
from dash import Dash, dcc, html, Input, Output
import dash_auth
import plotly.graph_objs as go

# ====== KONFIGURACJA LOGOWANIA ======
USERNAME_PASSWORD = [['user', 'pass']]
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD)

# ====== WCZYTYWANIE I ŁĄCZENIE DANYCH ======
def load_data():
    folder = r'C:\Users\mpiesio\Desktop\KODILLA\wizualizacja'
    files = [f'transactions-{year}.csv' for year in range(2016, 2020)]
    
    transactions = pd.concat([
        pd.read_csv(os.path.join(folder, f), index_col=0) for f in files
    ], ignore_index=True)

    def convert_dates(x):
        try:
            return dt.datetime.strptime(x, '%d-%m-%Y')
        except:
            return dt.datetime.strptime(x, '%d/%m/%Y')

    transactions['tran_date'] = transactions['tran_date'].apply(convert_dates)

    cc = pd.read_csv(os.path.join(folder, 'country_codes.csv'), index_col=0)
    customers = pd.read_csv(os.path.join(folder, 'customers.csv'), index_col=0)
    prod_info = pd.read_csv(os.path.join(folder, 'prod_cat_info.csv'))

    df = transactions.join(
        prod_info.drop_duplicates(subset=['prod_cat_code']).set_index('prod_cat_code')['prod_cat'],
        on='prod_cat_code', how='left')
    df = df.join(
        prod_info.drop_duplicates(subset=['prod_sub_cat_code']).set_index('prod_sub_cat_code')['prod_subcat'],
        on='prod_subcat_code', how='left')
    df = df.join(
        customers.join(cc, on='country_code').set_index('customer_Id'),
        on='cust_id')

    return df

df = load_data()

# ====== LAYOUT ======
app.layout = html.Div([
    html.Div([
        dcc.Tabs(id='tabs', value='tab-1', children=[
            dcc.Tab(label='Sprzedaż globalna', value='tab-1'),
            dcc.Tab(label='Produkty', value='tab-2'),
            dcc.Tab(label='Kanały sprzedaży', value='tab-3')
        ]),
        html.Div(id='tabs-content')
    ], style={'width': '80%', 'margin': 'auto'})
], style={'height': '100%'})

# ====== CALLBACK DO ZAKŁADEK ======
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return render_tab1(df)
    elif tab == 'tab-2':
        return render_tab2(df)
    elif tab == 'tab-3':
        return render_tab3(df)

# ====== ZAKŁADKA 1: Sprzedaż globalna ======
def render_tab1(df):
    return html.Div([
        html.H1('Sprzedaż globalna', style={'text-align': 'center'}),
        html.Div([
            dcc.DatePickerRange(
                id='sales-range',
                start_date=df['tran_date'].min(),
                end_date=df['tran_date'].max(),
                display_format='YYYY-MM-DD')
        ], style={'text-align': 'center'}),
        html.Div([
            html.Div([dcc.Graph(id='bar-sales')], style={'width': '50%'}),
            html.Div([dcc.Graph(id='choropleth-sales')], style={'width': '50%'})
        ], style={'display': 'flex'})
    ])

@app.callback(Output('bar-sales', 'figure'),
              [Input('sales-range', 'start_date'),
               Input('sales-range', 'end_date')])
def tab1_bar_sales(start_date, end_date):
    truncated = df[(df['tran_date'] >= start_date) & (df['tran_date'] <= end_date)]
    grouped = truncated[truncated['total_amt'] > 0].groupby(
        [pd.Grouper(key='tran_date', freq='M'), 'Store_type'])['total_amt'].sum().unstack()

    traces = [go.Bar(x=grouped.index, y=grouped[col], name=col) for col in grouped.columns]
    return go.Figure(data=traces, layout=go.Layout(title='Przychody', barmode='stack'))

@app.callback(Output('choropleth-sales', 'figure'),
              [Input('sales-range', 'start_date'),
               Input('sales-range', 'end_date')])
def tab1_choropleth_sales(start_date, end_date):
    truncated = df[(df['tran_date'] >= start_date) & (df['tran_date'] <= end_date)]
    grouped = truncated[truncated['total_amt'] > 0].groupby('country')['total_amt'].sum()

    trace = go.Choropleth(locations=grouped.index, locationmode='country names', z=grouped.values,
                          colorscale='Viridis', reversescale=True, colorbar=dict(title='Sales'))
    return go.Figure(data=[trace], layout=go.Layout(title='Mapa', geo=dict(projection={'type': 'natural earth'})))

# ====== ZAKŁADKA 2: Produkty ======
def render_tab2(df):
    grouped = df[df['total_amt'] > 0].groupby('prod_cat')['total_amt'].sum()
    fig = go.Figure(data=[go.Pie(labels=grouped.index, values=grouped.values)],
                   layout=go.Layout(title='Udział grup produktów'))
    return html.Div([
        html.H1('Produkty', style={'text-align': 'center'}),
        html.Div([
            html.Div([dcc.Graph(id='pie-prod-cat', figure=fig)], style={'width': '50%'}),
            html.Div([
                dcc.Dropdown(id='prod_dropdown',
                             options=[{'label': cat, 'value': cat} for cat in df['prod_cat'].unique()],
                             value=df['prod_cat'].unique()[0]),
                dcc.Graph(id='barh-prod-subcat')
            ], style={'width': '50%'})
        ], style={'display': 'flex'})
    ])

@app.callback(Output('barh-prod-subcat', 'figure'),
              Input('prod_dropdown', 'value'))
def tab2_barh_prod_subcat(chosen_cat):
    grouped = df[(df['total_amt'] > 0) & (df['prod_cat'] == chosen_cat)]
    pivot = grouped.pivot_table(index='prod_subcat', columns='Gender', values='total_amt', aggfunc='sum').fillna(0)
    pivot['sum'] = pivot.sum(axis=1)
    pivot = pivot.sort_values('sum')
    traces = [go.Bar(x=pivot[sex], y=pivot.index, orientation='h', name=sex) for sex in ['F', 'M']]
    return go.Figure(data=traces, layout=go.Layout(barmode='stack'))

# ====== ZAKŁADKA 3: Kanały sprzedaży ======
def render_tab3(df):
    return html.Div([
        html.H1('Kanały sprzedaży', style={'text-align': 'center'}),
        html.Div([
            dcc.Graph(id='bar-weekday-sales')
        ]),
        html.Div([
            dcc.Dropdown(id='store_dropdown',
                         options=[{'label': s, 'value': s} for s in df['Store_type'].unique()],
                         value=df['Store_type'].unique()[0]),
            dcc.Graph(id='store-demographics')
        ])
    ])

@app.callback(Output('bar-weekday-sales', 'figure'),
              Input('tabs', 'value'))
def tab3_weekday_sales(_):
    df2 = df.copy()
    df2['weekday'] = df2['tran_date'].dt.day_name()
    grouped = df2[df2['total_amt'] > 0].groupby(['weekday', 'Store_type'])['total_amt'].sum().unstack()
    grouped = grouped.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    traces = [go.Bar(x=grouped.index, y=grouped[col], name=col) for col in grouped.columns]
    return go.Figure(data=traces, layout=go.Layout(title='Sprzedaż wg dni tygodnia', barmode='group'))

@app.callback(Output('store-demographics', 'figure'),
              Input('store_dropdown', 'value'))
def tab3_store_demographics(store):
    df2 = df[(df['Store_type'] == store) & (df['total_amt'] > 0)]
    grouped = df2['Gender'].value_counts()
    return go.Figure(data=[go.Pie(labels=grouped.index, values=grouped.values)],
                     layout=go.Layout(title='Płeć klientów dla: ' + store))

# ====== START SERWERA ======
if __name__ == '__main__':
    app.run(debug=True)
