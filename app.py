from flask import Flask, request, render_template
import pandas as pd
import joblib
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

# Load the trained model and necessary data
model = joblib.load('random_forest_model_expanded.pkl')
df_encoded = pd.read_csv('Datasets\preprocessed_encoded.csv')
type_mean = df_encoded.groupby('type')['rate'].mean()
rest_type_mean = df_encoded.groupby('rest_type')['rate'].mean()
cuisines_mean = df_encoded.groupby('cuisines')['rate'].mean()

def expand_dataframe(df, column):
    df_expanded = df.copy()
    df_expanded[column] = df_expanded[column].str.split(', ')
    df_expanded = df_expanded.explode(column)
    return df_expanded

def predict_restaurant_rating(online_order, book_table, votes, rest_type, cuisines, cost, type):
    input_data = pd.DataFrame({
        'online_order': [online_order],
        'book_table': [book_table],
        'votes': [votes],
        'rest_type': [rest_type],
        'cuisines': [cuisines],
        'cost': [cost],
        'type': [type]
    })

    input_data = expand_dataframe(input_data, 'rest_type')
    input_data = expand_dataframe(input_data, 'cuisines')

    input_data['type'] = input_data['type'].map(type_mean)
    input_data['rest_type'] = input_data['rest_type'].map(rest_type_mean)
    input_data['cuisines'] = input_data['cuisines'].map(cuisines_mean)

    input_data = input_data.fillna(df_encoded.mean())

    predicted_rating = model.predict(input_data).mean()
    return predicted_rating.round()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    online_order = int(request.form['online_order'])
    book_table = int(request.form['book_table'])
    votes = int(request.form['votes'])
    rest_type = request.form['rest_type']
    cuisines = request.form['cuisines']
    cost = float(request.form['cost'])
    type = request.form['type']

    predicted_rating = predict_restaurant_rating(online_order, book_table, votes, rest_type, cuisines, cost, type)
    return render_template('result.html', rating=predicted_rating)

if __name__ == '__main__':
    app.run(debug=True)
