import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import pywt
import mne
import os
import joblib
import tempfile

from PIL import Image
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler

# Model Paths
model_paths = {
    "SVM": "model/SVM.joblib",
    "KNN": "model/KNN.joblib",
    "Decision Tree": "model/IamGroot.joblib",
    "Random Forest": "model/Forest.joblib",
    "Gradient Boosting": "model/GradientBoosting.joblib",
    "Logistic Regression": "model/LogisticRegression.joblib",
}

# Load Models
try:
    models = {name: joblib.load(path) for name, path in model_paths.items()}
except Exception as e:
    st.error(f"Error loading models: {e}")
    models = {}

# Load Scaler
scaler_path = "scaler.pkl"
try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    scaler = None
    st.error(f"Error loading scaler: {e}")


# Wavelet Features
def wavelet_features(signal, wavelet="db4", level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = []
    for coeff in coeffs:
        features.append(np.mean(coeff))
        features.append(np.std(coeff))
        features.append(np.var(coeff))
        features.append(np.sum(np.abs(coeff)))
    return features


# Hjorth Parameters
def hjorth_parameters(signal):
    activity = np.var(signal)
    mobility = np.sqrt(np.var(np.diff(signal)) / activity)
    complexity = np.sqrt(np.var(np.diff(np.diff(signal))) / np.var(np.diff(signal)))
    return activity, mobility, complexity


# Bandpower Ratios
def bandpower_ratio(fft, freqs):
    delta_power = np.sum(np.abs(fft[(freqs >= 0.5) & (freqs < 4)]) ** 2)
    theta_power = np.sum(np.abs(fft[(freqs >= 4) & (freqs < 8)]) ** 2)
    alpha_power = np.sum(np.abs(fft[(freqs >= 8) & (freqs < 13)]) ** 2)
    beta_power = np.sum(np.abs(fft[(freqs >= 13) & (freqs < 30)]) ** 2)
    theta_alpha_ratio = theta_power / alpha_power if alpha_power != 0 else 0
    delta_theta_ratio = delta_power / theta_power if theta_power != 0 else 0
    return (
        delta_power,
        theta_power,
        alpha_power,
        beta_power,
        theta_alpha_ratio,
        delta_theta_ratio,
    )


# Process EDF Features
def process_edf_features_return(file):
    # Save the UploadedFile to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    # Now read the EDF file using the temporary file path
    raw = mne.io.read_raw_edf(tmp_file_path, preload=True)
    raw.filter(1, 50.0)
    first_4_channels = raw.get_data()[0::2, :]
    sum_signal = np.sum(first_4_channels, axis=0)
    avg_signal = sum_signal / first_4_channels.shape[0]

    fft = np.fft.fft(avg_signal)
    freqs = np.fft.fftfreq(len(avg_signal), d=1 / raw.info["sfreq"])

    (
        delta_power,
        theta_power,
        alpha_power,
        beta_power,
        theta_alpha_ratio,
        delta_theta_ratio,
    ) = bandpower_ratio(fft, freqs)
    activity, mobility, complexity = hjorth_parameters(avg_signal)

    variance = np.var(avg_signal)
    power = np.mean(avg_signal**2)
    median = np.median(avg_signal)
    signal_range = np.max(avg_signal) - np.min(avg_signal)
    wavelet_feats = wavelet_features(avg_signal, wavelet="db4", level=4)
    duration = raw.times[-1]

    features = [
        variance,
        power,
        median,
        signal_range,
        delta_power,
        theta_power,
        alpha_power,
        beta_power,
        theta_alpha_ratio,
        delta_theta_ratio,
        activity,
        mobility,
        complexity,
        duration,
    ] + wavelet_feats

    return features


# GUI Title
st.markdown(
    "<div class='centered-menu-title'>Epileptic Classify</div>", unsafe_allow_html=True
)

# Option Menu
selected = option_menu(
    menu_title=None,
    options=["Image", "Features"],
    default_index=0,
    orientation="horizontal",
)

# Image Tab
if selected == "Image":
    st.markdown(
    """<div class='centered-menu-title'>Image Processing</div>""",
    unsafe_allow_html=True,
    )
    st.image(
        "images/image_processing.png",
        caption="Image Processing",
        use_container_width=True,
    )
    st.markdown(
        """
    <div style="text-align: justify;">
        The original data have noise, which make makes it difficult to use the original data for training. Therefore, we need to apply a technique called Independent Component Analysis (ICA) to filter channels with most information and reduce noises. This step is like isolating individual voices in a noisy room.The cleaned data is averaged to create a single signal representing overall brain activity.
        </div>
                """,
        unsafe_allow_html=True,
    )
    st.markdown(" ")
    st.markdown(
        """
    <div style="text-align: justify;">
        To enhance the signal quality, filters are applied. A band-pass filter is used to isolate specific frequency bands, such as those associated with sweating and blinking. Additionally, a notch filter removes any noise at a particular frequency, further refining the signal.        </div>
                """,
        unsafe_allow_html=True,
    )
    st.markdown(" ")
    st.markdown(
        """
    <div style="text-align: justify;">
                Finally, the processed signal is visualized as a graph. The x-axis represents time, while the y-axis shows the intensity of brain activity.        
                </div>
                """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    col1, col2, col3 = st.columns([1, 3, 3])  # Adjust the ratio as needed

    with col2:
        st.image(
            "images/aaaaaawu_s001_t001.png",
            caption="Average Signal",
            width=500,  # Adjust the width as desired
        )
    
    st.markdown("### Performance")
    st.image(
        "images/241123_bestmodel.png",
        caption="ROC Curve",
        use_container_width=True,
    )

    # Load Pre-trained Model
    model = tf.keras.models.load_model("241123_my_model.keras")
    uploaded_file = st.file_uploader("Choose an image...")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        if img.mode != "RGB":
            img = img.convert("RGB")
        st.image(img, caption="Uploaded Image")
        img = img.resize((456, 608))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array) + 0.076
        progress_bar = st.progress(0, text="Operation in progress...")
        for perc_completed in range(100):
            time.sleep(0.005)
            progress_bar.progress(perc_completed + 1, text="Operation in progress...")
        time.sleep(0.01)

        if predictions[0] < 0.5:
            st.write("Seizure detected")
        else:
            st.write("Normal signal")
    # st.write(predictions)

# Features Tab
elif selected == "Features":
    st.markdown(
    """<div class='centered-menu-title'>Feature Extraction</div>""",
    unsafe_allow_html=True,
    )
    st.image(
        "images/feature_extraction.png",
        caption="Feature Extraction Process",
        use_container_width=True,
    )
    st.markdown(
        """
    <div style="text-align: justify;">
                The first approach of executing the model requires the collection of EEG data
    through the help of electroencephalogram devices. These devices measure electrical
    signals from the scalp and give information concerning the brains’ activity. The
    collected data is normally stored in the form of European Data Format files, ab-
    breviated as EDF files. This data is first preprocessed in order to check the quality
    and appropriateness of this data for the analysis stage. This involves several steps,
    including:       
                </div>
                """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown("**Time-Frequency Analysis:** Quantifying the EEG spectra together with thetime course in order for specific signatures and/or frequencies to be pinpointedto different mental states.")
    st.markdown("**Denoising:** Filtering out artifacts from the EEG signal including muscle activity, eye blinks as well as power line interference. This is usually done by asising method such as filtering and Independent Component Analysis (ICA).")
    st.markdown(
        """
    <div style="text-align: justify;">
    Next, after the preprocessing, features are selected from EEG data. These features can be of time domain features, frequency domain features or time-frequency domain features. These features are then used to train a certain machine learning model and then classify different brain states or different certain events, say seizures.    
                </div>
                """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(
        """
    <div style="text-align: justify;">
    It means the training process reduces the dataset into the training set and the testing one. The training file is used to to build the model and the test file is used to measure accuracy of the built model. Different techniques in machine learning like Support Vector Machines, Random Forest or even deep learning methods can be used for it.    
                </div>
                """,
        unsafe_allow_html=True,
    )
    st.image("images/feature.png", caption="Part of features container file", use_container_width =True)
    st.markdown("### Performance")
    data_length = {
        "Model": ["SVM", "Random Forest", "KNN", "Logistic Regression", "Gradient Boosting", "Decision Tree"],
        "Accuracy": [0.915, 0.833, 0.875, 0.875, 0.9375, 0.75],
    }
    df_length = pd.DataFrame(data_length)
    df_length['Accuracy'] = df_length['Accuracy'].map('{:.3f}'.format) 
    st.markdown("<h3 style='text-align: center;'>Table: Performance of models</h3>", unsafe_allow_html=True)
    st.dataframe(
    df_length.style.set_properties(
        **{"text-align": "center",}
    ).set_table_styles(
        [{"selector": "th", "props": [("text-align", "center")]}]
    ),
    use_container_width=True,)
    if not models:
        st.error("No models are loaded!")
    else:
        selected_model = st.selectbox("Choose a Model:", list(models.keys()))
        ML_model = models[selected_model]
        uploaded_file = st.file_uploader("Upload an EDF file", type=["edf"])
        if uploaded_file is not None:
            try:
                st.write(
                    f"Processing the uploaded EDF file using {selected_model} model..."
                )
                features = process_edf_features_return(uploaded_file)
                features = np.array(features).reshape(1, -1)

                # if scaler:
                #     features = scaler.transform(features)

                prediction = ML_model.predict(features)
                prediction_proba = None
                if hasattr(ML_model, "predict_proba"):
                    prediction_proba = ML_model.predict_proba(features)

                st.write(
                    f"Prediction: {'Epileptic' if prediction[0] == 1 else 'Non-Epileptic'}"
                )
                if prediction_proba is not None:
                    st.write(f"Probability (Epileptic): {prediction_proba[0][1]:.2f}")
            except Exception as e:
                st.error(f"Error processing file: {e}")


# Styling
st.markdown(
    """
    <style>
    .paragraph {
        text-align: justify;
        font-size: 1.1em;
        margin-top: 10px;
    }

    .centered-menu-title {
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .centered-title {
        text-align: center;
        font-size: 2.5em;  /* Adjust font size if needed */
        margin-top: 20px;  /* Adjust margin for spacing */
    }
    .centered-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
