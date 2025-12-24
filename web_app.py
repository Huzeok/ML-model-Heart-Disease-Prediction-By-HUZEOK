from flask import Flask, render_template, request
import joblib
import pandas as pd
from pathlib import Path
import itertools

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'KNN.pkl'
SCALER_PATH = BASE_DIR / 'scaler.pkl'
COLUMNS_PATH = BASE_DIR / 'columns.pkl'

def _load_or_exit(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"Required file '{name}' not found at: {path}")
    return joblib.load(path)

model = _load_or_exit(MODEL_PATH, 'KNN.pkl')
scaler = _load_or_exit(SCALER_PATH, 'scaler.pkl')
expected_columns = _load_or_exit(COLUMNS_PATH, 'columns.pkl')

app = Flask(__name__)

COLOR_MAP = {
    'success': {'text': '#155724', 'bg': '#d4edda'},
    'danger': {'text': '#721c24', 'bg': '#f8d7da'},
    'info': {'text': '#0c5460', 'bg': '#d1ecf1'},
    'warning': {'text': '#856404', 'bg': '#fff3cd'},
}
INPUT_PALETTE = [
    '#FFB6C1', '#FFA07A', '#FFD700', '#90EE90', '#87CEFA',
    '#BA55D3', '#FFE4B5', '#98FB98', '#AFEEEE', '#F08080'
]

# add title
TITLE = "Hearth Disease Prediction by HUZEOK"


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', title=TITLE)


@app.route('/predict', methods=['POST'])
def predict():
    form = request.form
    # collect inputs
    raw_input = {
        'age': int(form.get('age', 40)),
        'trestbps': float(form.get('trestbps', 120)),
        'chol': float(form.get('chol', 200)),
        'fbs': int(form.get('fbs', 0)),
        'thalach': float(form.get('thalach', 150)),
        'oldpeak': float(form.get('oldpeak', 1.0)),
        'cp': int(form.get('cp', 0)),
        'sex': 0 if form.get('sex', 'M') == 'M' else 1,
        'restecg': int(form.get('restecg', 0)),
        'exang': int(form.get('exang', 0)),
        'slope': int(form.get('slope', 0)),
        'ca': int(form.get('ca', 0)),
        'thal': int(form.get('thal', 0)),
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    scaled = scaler.transform(input_df)
    pred = model.predict(scaled)[0]
    if int(pred) == 1:
        message = 'High risk of Heart Disease — consult a doctor.'
        result_type = 'danger'
    else:
        message = 'Low risk of Heart Disease.'
        result_type = 'success'

    color_info = COLOR_MAP.get(result_type, COLOR_MAP['info'])
    text_color = color_info['text']
    bg_color = color_info['bg']

    # assign colors to each input for display in the template
    palette_cycle = itertools.cycle(INPUT_PALETTE)
    input_colors = {k: next(palette_cycle) for k in raw_input.keys()}

    return render_template(
        'result.html',
        message=message,
        result_type=result_type,
        inputs=raw_input,
        text_color=text_color,
        bg_color=bg_color,
        input_colors=input_colors,
        title=TITLE
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8501)
