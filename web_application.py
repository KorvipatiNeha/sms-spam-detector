import streamlit as st
import pickle
from pathlib import Path

# 1. Configure the webpage
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="🛡️",
    layout="centered"
)

# 2. Load the trained model and classification threshold
MODEL_PATH = Path(__file__).parent / "spam_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        saved_data = pickle.load(file)

    # The pickle file contains a dictionary
    # with the trained model and the best threshold.
    model = saved_data["model"]
    threshold = saved_data["threshold"]

    return model, threshold


model, SPAM_THRESHOLD = load_model()

# 3. Display the webpage heading
st.title("🛡️ Spam Message Detector")

st.write(
    "Enter a message below to check whether it is "
    "spam or a legitimate message."
)

st.info(
    "This application uses TF-IDF and Naive Bayes "
    "to classify SMS messages."
)

# 4. Get message input from the user
message = st.text_area(
    "Enter your message",
    placeholder=(
        "Example: Congratulations! You have won "
        "a free prize. Call now to claim your reward!"
    ),
    height=150
)

# 5. Predict whether the message is spam or ham
if st.button(
    "Check Message",
    type="primary",
    use_container_width=True
):

    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        # Get the probability of each class
        probabilities = model.predict_proba([message])[0]

        # Get the class names learned during training
        classes = list(
            model.named_steps["classifier"].classes_
        )

        # Find the correct probability positions
        spam_index = classes.index("spam")
        ham_index = classes.index("ham")

        spam_probability = float(
            probabilities[spam_index]
        )

        ham_probability = float(
            probabilities[ham_index]
        )

        # Use the saved threshold instead of the default
        # prediction threshold used by model.predict().
        if spam_probability >= SPAM_THRESHOLD:
            prediction = "spam"
        else:
            prediction = "ham"

        # 6. Display the prediction
        st.subheader("Prediction Result")

        if prediction == "spam":
            st.error(
                "🚨 This message is classified as SPAM!"
            )
        else:
            st.success(
                "✅ This message is classified as HAM (Not Spam)."
            )

        # 7. Display probability details
        st.subheader("Model Prediction Details")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Spam Probability",
                value=f"{spam_probability:.2%}"
            )

        with col2:
            st.metric(
                label="Ham Probability",
                value=f"{ham_probability:.2%}"
            )

        st.progress(spam_probability)

        st.caption(
            f"Classification threshold: {SPAM_THRESHOLD:.2f}"
        )

        # 8. Check which text features the model recognizes
        vectorizer = model.named_steps["tfidf"]

        transformed_message = vectorizer.transform(
            [message]
        )

        recognized_words = transformed_message.nnz

        st.subheader("Text Analysis")

        st.write(
            f"**Non-zero TF-IDF features:** {recognized_words}"
        )

        if recognized_words == 0:
            st.warning(
                "The model could not recognize any vocabulary "
                "features in this message. Try entering a message "
                "with words commonly used in SMS messages."
            )
        else:
            st.write(
                "The model recognized vocabulary features "
                "from your message."
            )

        # 9. Warn when a ham prediction still has notable
        # spam probability
        if (
            prediction == "ham"
            and spam_probability >= 0.10
        ):
            st.warning(
                "This message was classified as HAM, but it has "
                "some estimated spam probability. Review the "
                "message carefully."
            )

        # Explain the limitations of the model
        st.caption(
            "Note: Probabilities are model estimates. "
            "They do not guarantee that a message is safe or spam."
        )

# 10. Footer
st.divider()

st.caption(
    "Built with Python, Streamlit, TF-IDF and Naive Bayes."
)