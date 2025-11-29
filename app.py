from flask import Flask, render_template, request
import numpy as np
import pickle
import tensorflow as tf

app = Flask(__name__)

# -------------------------------------------
# LOAD TRAINED MODEL
# -------------------------------------------
model = tf.keras.models.load_model("EncDec.h5", compile=False)

latent_dim = model.layers[2].units

# EXTRACT TRAINING INPUTS
encoder_inputs = model.input[0]   # OK to reuse
training_decoder_inputs = model.input[1]  # DO NOT reuse

# EXTRACT LAYERS
encoder_lstm = model.layers[2]
decoder_lstm = model.layers[3]
decoder_dense = model.layers[4]

# -------------------------------------------
# BUILD INFERENCE ENCODER
# -------------------------------------------
encoder_outputs, state_h_enc, state_c_enc = encoder_lstm(encoder_inputs)
encoder_states = [state_h_enc, state_c_enc]
encoder_model = tf.keras.Model(encoder_inputs, encoder_states)

# -------------------------------------------
# BUILD INFERENCE DECODER (SAFE VERSION)
# -------------------------------------------
decoder_state_input_h = tf.keras.Input(shape=(latent_dim,), name="dec_h")
decoder_state_input_c = tf.keras.Input(shape=(latent_dim,), name="dec_c")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# NEW decoder input layer (MUST be new Input)
decoder_inputs_inf = tf.keras.Input(
    shape=(1, model.output_shape[-1]),
    name="decoder_input_inf"
)

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs_inf, initial_state=decoder_states_inputs
)

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = tf.keras.Model(
    [decoder_inputs_inf] + decoder_states_inputs,
    [decoder_outputs, state_h, state_c],
)

# -------------------------------------------
# LOAD TOKEN DICTIONARIES
# -------------------------------------------
with open("input_token_index.pkl", "rb") as f:
    input_token_index = pickle.load(f)

with open("target_token_index.pkl", "rb") as f:
    target_token_index = pickle.load(f)

with open("reverse_input_char_index.pkl", "rb") as f:
    reverse_input_char_index = pickle.load(f)

with open("reverse_target_char_index.pkl", "rb") as f:
    reverse_target_char_index = pickle.load(f)

num_encoder_tokens = len(input_token_index)
num_decoder_tokens = len(target_token_index)
max_encoder_seq_length = 20
max_decoder_seq_length = 20

# -------------------------------------------
# ENCODE INPUT
# -------------------------------------------
def encode_input_text(text):
    x = np.zeros((1, max_encoder_seq_length, num_encoder_tokens))
    for t, char in enumerate(text):
        if t < max_encoder_seq_length:
            x[0, t, input_token_index.get(char, input_token_index[" "])] = 1.0
    return x

# -------------------------------------------
# DECODE SEQUENCE
# -------------------------------------------
def decode_sequence(input_seq):
    states = encoder_model.predict(input_seq, verbose=0)

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    decoded = ""

    while True:
        output_tokens, h, c = decoder_model.predict([target_seq] + states, verbose=0)

        sampled_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_index]
        decoded += sampled_char

        if sampled_char == "\n" or len(decoded) > max_decoder_seq_length:
            break

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_index] = 1.0

        states = [h, c]

    return decoded

# -------------------------------------------
# FLASK ROUTES
# -------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    translation = ""
    english = ""

    if request.method == "POST":
        english = request.form["english_text"].strip().lower()
        encoded = encode_input_text(english)
        translation = decode_sequence(encoded)

    return render_template("index.html",
                           translated_text=translation,
                           english_text=english)

if __name__ == "__main__":
    app.run(debug=True)
