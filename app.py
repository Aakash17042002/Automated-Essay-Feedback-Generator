import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import textstat
import json

from scipy.sparse import hstack
import ollama


# -----------------------------
# Load trained ML components
# -----------------------------

vectorizer = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("feature_scaler.pkl")
all_models = joblib.load("all_models.pkl")


# -----------------------------
# Feature definitions
# -----------------------------

numeric_features = [
    'word_count',
    'sentence_count',
    'avg_sentence_length',
    'unique_word_count',
    'vocabulary_richness',
    'readability_score',
    'comma_count',
    'question_count',
    'exclamation_count',
    'long_sentence_count',
    'repeated_word_count'
]


# -----------------------------
# Text cleaning
# -----------------------------

def clean_essay(text):

    text = re.sub(r'@\w+\d*', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


# -----------------------------
# Feature extraction
# -----------------------------

def analyze_essay(text):

    sentences = re.split(r'[.!?]+', text)

    sentences = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    words = text.split()

    word_list = [
        word.lower().strip(".,!?;:\"'()")
        for word in words
    ]

    return {
        'word_count': len(words),

        'sentence_count': len(sentences),

        'avg_sentence_length':
            len(words) / len(sentences)
            if sentences else 0,

        'unique_word_count':
            len(set(word_list)),

        'vocabulary_richness':
            len(set(word_list)) / len(words)
            if words else 0,

        'comma_count':
            text.count(','),

        'question_count':
            text.count('?'),

        'exclamation_count':
            text.count('!'),

        'long_sentence_count':
            sum(
                len(sentence.split()) > 30
                for sentence in sentences
            ),

        'repeated_word_count':
            sum(
                word_list[i] == word_list[i - 1]
                for i in range(1, len(word_list))
                if word_list[i]
            )
    }


# -----------------------------
# Prepare essay features
# -----------------------------

def prepare_features(essay_text):

    clean_text = clean_essay(essay_text)

    features = analyze_essay(clean_text)

    readability = textstat.flesch_reading_ease(
        clean_text
    )

    # Create numeric features
    numeric_values = pd.DataFrame([[
        features['word_count'],
        features['sentence_count'],
        features['avg_sentence_length'],
        features['unique_word_count'],
        features['vocabulary_richness'],
        readability,
        features['comma_count'],
        features['question_count'],
        features['exclamation_count'],
        features['long_sentence_count'],
        features['repeated_word_count']
    ]], columns=numeric_features)

    # Scale numeric features
    numeric_values_scaled = scaler.transform(
        numeric_values
    )

    # TF-IDF features
    essay_tfidf = vectorizer.transform(
        [clean_text]
    )

    # TF-IDF + NLP features
    essay_combined = hstack([
        essay_tfidf,
        numeric_values_scaled
    ])

    return (
        clean_text,
        essay_tfidf,
        numeric_values_scaled,
        essay_combined
    )


# -----------------------------
# ML prediction
# -----------------------------

def predict_score(
    essay_text,
    model_name,
    essay_set
):

    (
        clean_text,
        essay_tfidf,
        numeric_values,
        essay_combined
    ) = prepare_features(essay_text)


    # --------------------------------
    # Ridge - TF-IDF
    # --------------------------------

    if model_name == "Ridge - TF-IDF":

        model = all_models[
            "Ridge - TF-IDF"
        ]

        predicted_score = model.predict(
            essay_tfidf
        )[0]


    # --------------------------------
    # Ridge - TF-IDF + NLP Features
    # --------------------------------

    elif model_name == "Ridge - TF-IDF + NLP Features":

        model = all_models[
            "Ridge - TF-IDF + NLP Features"
        ]

        predicted_score = model.predict(
            essay_combined
        )[0]


    # --------------------------------
    # Random Forest
    # --------------------------------

    elif model_name == "Random Forest":

        model = all_models[
            "Random Forest"
        ]

        predicted_score = model.predict(
            numeric_values
        )[0]


    # --------------------------------
    # Gradient Boosting
    # --------------------------------

    elif model_name == "Gradient Boosting":

        model = all_models[
            "Gradient Boosting"
        ]

        predicted_score = model.predict(
            numeric_values
        )[0]


    # --------------------------------
    # Ridge - Separate Essay Set
    # --------------------------------

    elif model_name == "Ridge - Separate Essay Set":

        selected_models = all_models[
            "Ridge - Separate Essay Set"
        ]

        model = selected_models[essay_set]

        predicted_score = model.predict(
            essay_combined
        )[0]


    else:

        raise ValueError(
            "Invalid model selected."
        )


    # Keep score between 0 and 1
    predicted_score = max(
        0,
        min(1, predicted_score)
    )

    # Convert to 0-100
    return round(
        predicted_score * 100,
        2
    )


# -----------------------------
# AI Feedback
# -----------------------------

def generate_feedback(essay_text):

    response = ollama.chat(

        model='llama3.2:3b',

        messages=[

            {
                'role': 'user',

                'content': f"""
You are an expert English writing evaluator.

Analyze the student essay and return ONLY valid JSON.

Use exactly this structure:

{{
  "grammar": {{
    "score": 0,
    "feedback": ""
  }},

  "sentence_structure": {{
    "score": 0,
    "feedback": ""
  }},

  "coherence": {{
    "score": 0,
    "feedback": ""
  }},

  "argument_strength": {{
    "score": 0,
    "feedback": ""
  }},

  "vocabulary": {{
    "score": 0,
    "feedback": ""
  }},

  "overall": {{
    "score": 0,
    "feedback": ""
  }},

  "suggestions": [
    "Suggestion 1",
    "Suggestion 2",
    "Suggestion 3"
  ]
}}

Rules:

- Scores must be between 1 and 10.
- Give concise and constructive feedback.
- Give exactly 3 specific and actionable suggestions.
- Suggestions must NOT be empty.
- Suggestions must be related specifically to the student's essay.
- Do not invent grammar errors.
- Only report a grammar error if there is a clear grammatical mistake.
- Distinguish grammar errors from style suggestions.
- Do not include Markdown.
- Do not include anything outside the JSON.

Student Essay:

{essay_text}
"""
            }
        ]
    )


    result = response['message']['content']

    return json.loads(result)


# -----------------------------
# Streamlit Page Configuration
# -----------------------------

st.set_page_config(

    page_title="Automated Essay Feedback Generator",

    page_icon="📝",

    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title(
    "📝 Automated Essay Feedback Generator"
)

st.write(
    "Analyze your essay using Machine Learning and Generative AI."
)


# -----------------------------
# Model Selection
# -----------------------------

st.subheader(
    "⚙️ Model Selection"
)

model_options = [

    "Ridge - TF-IDF",

    "Ridge - TF-IDF + NLP Features",

    "Random Forest",

    "Gradient Boosting",

    "Ridge - Separate Essay Set"

]

selected_model = st.selectbox(

    "Select Model",

    model_options
)


# -----------------------------
# Essay Set
# -----------------------------

st.subheader(
    "📚 Essay Settings"
)

essay_set = st.selectbox(

    "Select Essay Set",

    [1, 2, 3, 4, 5, 6, 7, 8]
)


# -----------------------------
# Essay Input
# -----------------------------

st.subheader(
    "✍️ Enter Your Essay"
)

essay_text = st.text_area(

    "Paste your essay here",

    height=350,

    placeholder="Enter your essay..."
)


# -----------------------------
# Evaluate Button
# -----------------------------

if st.button(
    "🚀 Evaluate Essay"
):

    if not essay_text.strip():

        st.warning(
            "Please enter an essay first."
        )

    else:

        with st.spinner(
            "Analyzing your essay..."
        ):

            try:

                # ML prediction
                ml_score = predict_score(

                    essay_text,

                    selected_model,

                    essay_set

                )

                # AI feedback
                feedback = generate_feedback(

                    essay_text

                )


            except Exception as e:

                st.error(
                    f"Error while evaluating essay: {e}"
                )

                st.stop()


        # -----------------------------
        # ML Score
        # -----------------------------

        st.subheader(
            "📊 ML Predicted Quality Score"
        )

        st.metric(

            "Score",

            f"{ml_score}/100"
        )

        st.caption(
            f"Model used: {selected_model}"
        )


        # -----------------------------
        # AI Writing Feedback
        # -----------------------------

        st.subheader(
            "✍️ AI Writing Feedback"
        )


        col1, col2 = st.columns(2)


        # Left column
        with col1:

            st.write(
                f"**Grammar:** "
                f"{feedback['grammar']['score']}/10"
            )

            st.write(
                feedback['grammar']['feedback']
            )


            st.write(
                f"**Sentence Structure:** "
                f"{feedback['sentence_structure']['score']}/10"
            )

            st.write(
                feedback['sentence_structure']['feedback']
            )


            st.write(
                f"**Coherence:** "
                f"{feedback['coherence']['score']}/10"
            )

            st.write(
                feedback['coherence']['feedback']
            )


        # Right column
        with col2:

            st.write(
                f"**Argument Strength:** "
                f"{feedback['argument_strength']['score']}/10"
            )

            st.write(
                feedback['argument_strength']['feedback']
            )


            st.write(
                f"**Vocabulary:** "
                f"{feedback['vocabulary']['score']}/10"
            )

            st.write(
                feedback['vocabulary']['feedback']
            )


            st.write(
                f"**Overall:** "
                f"{feedback['overall']['score']}/10"
            )

            st.write(
                feedback['overall']['feedback']
            )


        # -----------------------------
        # Personalized Suggestions
        # -----------------------------

        st.subheader(
            "💡 Personalized Suggestions"
        )


        suggestions = feedback.get(

            "suggestions",

            []
        )


        # Remove empty suggestions
        valid_suggestions = [

            suggestion.strip()

            for suggestion in suggestions

            if isinstance(
                suggestion,
                str
            )

            and suggestion.strip()
        ]


        if len(valid_suggestions) >= 3:

            for suggestion in valid_suggestions[:3]:

                st.write(
                    "•",
                    suggestion
                )


        else:

            st.write(
                "• Add specific examples or evidence "
                "to strengthen your main arguments."
            )

            st.write(
                "• Improve transitions between paragraphs "
                "to make the essay flow more naturally."
            )

            st.write(
                "• Strengthen the conclusion by clearly "
                "summarizing your main argument."
            )
