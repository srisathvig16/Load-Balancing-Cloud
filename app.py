from flask import Flask, request, render_template

app = Flask(__name__)

# Sample function for load balancing prediction (replace with your actual prediction logic)
def predict_load_balancing(features):
    # Example hardcoded prediction based on feature values
    source_port = int(features['source_port'])
    destination_port = int(features['destination_port'])
    
    if source_port % 2 == 0:
        if destination_port % 2 == 0:
            return "Server1"
        else:
            return "Server2"
    else:
        if destination_port % 2 == 0:
            return "Server3"
        else:
            return "Server4"

# Route to render the HTML form
@app.route('/')
def home():
    return render_template('webpage.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve input features from the form
        features = {
            'source_port': request.form['source_port'],
            'destination_ip': request.form['destination_ip'],
            'destination_port': request.form['destination_port'],
            'protocol': request.form['protocol'],
            'flow_duration': request.form['flow_duration'],
            'total_fwd_packets': request.form['total_fwd_packets'],
            'total_backward_packets': request.form['total_backward_packets'],
            'total_length_fwd_packets': request.form['total_length_fwd_packets'],
            'total_length_bwd_packets': request.form['total_length_bwd_packets'],
            'fwd_packet_length_max': request.form['fwd_packet_length_max']
        }

        # Simulate prediction based on hardcoded logic
        prediction = predict_load_balancing(features)

        # Render result page with prediction
        return render_template('result.html', prediction_result=prediction)

if __name__ == '__main__':
    app.run(debug=True)