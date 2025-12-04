import streamlit as st
import chromadb
import os

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Page config
st.set_page_config(
    page_title="Philadelphia Places Finder",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .place-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    .theme-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        margin: 10px 0;
        border: none;
        width: 100%;
        cursor: pointer;
    }
    .sentiment-great {
        color: #4CAF50;
        font-weight: bold;
    }
    .sentiment-good {
        color: #8BC34A;
        font-weight: bold;
    }
    .sentiment-mixed {
        color: #FF9800;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #9E9E9E;
        font-weight: bold;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


class PhillyPlacesRAG:
    def __init__(self):
        self.client = None
        self.collection = None
        self.llm_pipeline = None
        self.tokenizer = None

    @staticmethod
    @st.cache_resource
    def load_transformers():
        """Lazy load transformers to avoid import issues"""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
            return pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError as e:
            st.error("""
            ⚠️ **Missing Dependencies**

            Please install required packages:
            ```
            pip install transformers torch sentencepiece
            ```
            """)
            return None, None, None

    def load_models(self):
        """Load LLM models"""
        pipeline_func, AutoTokenizer, AutoModelForSeq2SeqLM = self.load_transformers()

        if pipeline_func is None:
            st.error("Cannot load AI model. Please install dependencies.")
            return None, None

        with st.spinner("Loading FLAN-T5 model... This may take a minute..."):
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            llm_pipeline = pipeline_func(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=50
            )
        return llm_pipeline, tokenizer

    def connect_to_db(self):
        """Connect to ChromaDB"""
        if self.client is None:
            self.client = chromadb.CloudClient(
                api_key="ck-Bz6vWE8LH4mrmKx2dfreGSzgiikFBKhm7BHNHivkDvvw",
                tenant="6db89e03-6466-4af4-ad1c-d8237a75efa7",
                database="places"
            )
            self.collection = self.client.get_collection(name="cleaned_posts")
        return True

    def extract_intent_from_query(self, query):
        """Use LLM to extract location and theme from user query"""
        prompt = f"""Extract the location and theme from this query. 
Return ONLY in this format: Location: [location], Theme: [theme]

Query: {query}
Answer:"""

        response = self.llm_pipeline(prompt)[0]['generated_text'].strip()

        # Parse response
        location = "philadelphia"  # default
        theme = None

        try:
            if "Location:" in response:
                location = response.split("Location:")[1].split(",")[0].strip().lower()
            if "Theme:" in response:
                theme = response.split("Theme:")[1].strip().lower()
        except:
            pass

        return location, theme, response

    def simple_extract(self, query):
        """Simple keyword-based extraction (fallback)"""
        query_lower = query.lower()

        # Extract location
        location = "philadelphia"
        if "philly" in query_lower or "philadelphia" in query_lower:
            location = "philadelphia"

        # Extract theme
        theme = None
        theme_keywords = {
            'restaurants': ['restaurant', 'food', 'eat', 'dining', 'steak'],
            'hidden_gems': ['hidden', 'gem', 'secret', 'underrated'],
            'bars': ['bar', 'pub', 'drink', 'cocktail'],
            'cafes': ['cafe', 'coffee', 'tea'],
        }

        for theme_name, keywords in theme_keywords.items():
            if any(kw in query_lower for kw in keywords):
                theme = theme_name
                break

        return location, theme

    def filter_by_location(self, location):
        """Filter all data by location and return available themes"""
        all_data = self.collection.get()

        filtered_items = []
        themes_dict = {}

        for i, metadata in enumerate(all_data['metadatas']):
            item_location = metadata.get('location', '').lower()

            if location in item_location:
                theme = metadata.get('theme', 'other')

                item = {
                    'id': all_data['ids'][i],
                    'place': metadata.get('places', 'N/A'),
                    'location': metadata.get('location', 'N/A'),
                    'theme': theme,
                    'notes': metadata.get('notes', 'N/A'),
                    'sentiment': metadata.get('sentiment', 'N/A'),
                }

                filtered_items.append(item)

                # Group by theme
                if theme not in themes_dict:
                    themes_dict[theme] = []
                themes_dict[theme].append(item)

        return filtered_items, themes_dict

    def get_theme_icon(self, theme):
        """Get emoji icon for theme"""
        icons = {
            'restaurants': '🍽️',
            'hidden_gems': '💎',
            'attractions': '🎡',
            'bars': '🍺',
            'cafes': '☕',
            'shopping': '🛍️',
            'parks': '🌳',
            'museums': '🏛️',
            'nightlife': '🌃',
        }
        return icons.get(theme, '📍')

    def get_sentiment_icon(self, sentiment):
        """Get emoji for sentiment"""
        icons = {
            'great': '⭐⭐⭐',
            'good': '⭐⭐',
            'mixed': '⭐',
            'neutral': '➖',
        }
        return icons.get(sentiment, '')


# Initialize session state
if 'app' not in st.session_state:
    st.session_state.app = PhillyPlacesRAG()
    st.session_state.connected = False
    st.session_state.models_loaded = False
    st.session_state.current_themes = None
    st.session_state.current_location = None
    st.session_state.llm_response = None
    st.session_state.use_llm = True

app = st.session_state.app

# Header
st.title("🔍 Philadelphia Places Finder")
st.markdown("### Discover restaurants, hidden gems, and more using AI-powered search")

# Sidebar for setup
with st.sidebar:
    st.header("⚙️ Setup")

    if st.button("🔌 Connect to Database", type="primary"):
        with st.spinner("Connecting..."):
            try:
                app.connect_to_db()
                st.session_state.connected = True
                st.success("✅ Connected to ChromaDB!")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")

    if st.session_state.connected:
        st.markdown("---")
        st.subheader("🤖 AI Options")

        use_ai = st.checkbox("Use AI for query understanding", value=False)
        st.session_state.use_llm = use_ai

        if use_ai and not st.session_state.models_loaded:
            if st.button("Load AI Model", type="primary"):
                try:
                    result = app.load_models()
                    if result[0] is not None:
                        app.llm_pipeline, app.tokenizer = result
                        st.session_state.models_loaded = True
                        st.success("✅ Model loaded!")
                    else:
                        st.error("Failed to load model")
                except Exception as e:
                    st.error(f"❌ Model loading failed: {e}")
                    st.info("💡 Tip: You can still search without AI using keyword matching")

    st.markdown("---")

    if st.session_state.connected:
        st.metric("📊 Status", "Connected ✓")
        try:
            total = app.collection.count()
            st.metric("📍 Total Places", total)
        except:
            pass

    if st.session_state.models_loaded:
        st.metric("🤖 AI Model", "Ready ✓")
    elif st.session_state.use_llm:
        st.metric("🤖 AI Model", "Not Loaded")
    else:
        st.metric("🔍 Search Mode", "Keyword-based")

    st.markdown("---")
    st.markdown("### 💡 Try these queries:")
    st.markdown("- *restaurants in philadelphia*")
    st.markdown("- *hidden gems in philly*")
    st.markdown("- *best places to eat*")
    st.markdown("- *steakhouses*")

# Main content
if not st.session_state.connected:
    st.warning("⚠️ Please connect to the database using the sidebar")
    st.info("👈 Click 'Connect to Database' in the sidebar to get started")
else:
    # Search interface
    st.markdown("---")
    query = st.text_input(
        "💬 What are you looking for?",
        placeholder="e.g., restaurants in philadelphia, hidden gems, best steakhouses...",
        key="search_query"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔎 Search", type="primary", use_container_width=True)
    with col2:
        if st.session_state.current_themes:
            if st.button("🔄 New Search", use_container_width=True):
                st.session_state.current_themes = None
                st.session_state.current_location = None
                st.session_state.llm_response = None
                st.rerun()

    if search_button and query:
        # Extract intent
        if st.session_state.use_llm and st.session_state.models_loaded:
            with st.spinner("🤖 Understanding your query with AI..."):
                location, suggested_theme, llm_response = app.extract_intent_from_query(query)
                st.session_state.current_location = location
                st.session_state.llm_response = llm_response

                # Show what AI understood
                with st.expander("🧠 AI Understanding", expanded=False):
                    st.info(f"**LLM Response:** {llm_response}")
                    st.write(f"**Location detected:** {location}")
                    st.write(f"**Theme suggested:** {suggested_theme if suggested_theme else 'None'}")
        else:
            with st.spinner("🔍 Processing query..."):
                location, suggested_theme = app.simple_extract(query)
                st.session_state.current_location = location
                st.info(
                    f"🎯 Searching for: **{suggested_theme if suggested_theme else 'all categories'}** in **{location}**")

        with st.spinner(f"🔍 Searching in {location}..."):
            filtered_items, themes_dict = app.filter_by_location(location)
            st.session_state.current_themes = themes_dict

            if not themes_dict:
                st.error(f"❌ No places found for '{location}'")
            else:
                st.success(f"✅ Found {len(filtered_items)} places in {location}")

    # Display themes
    if st.session_state.current_themes:
        st.markdown("---")
        st.markdown("## 🎯 Select a Category")

        themes_dict = st.session_state.current_themes

        # Create columns for theme cards
        cols = st.columns(3)
        theme_items = list(themes_dict.items())

        for idx, (theme, places) in enumerate(sorted(theme_items)):
            col_idx = idx % 3
            with cols[col_idx]:
                icon = app.get_theme_icon(theme)

                # Create clickable theme card
                if st.button(
                        f"{icon} {theme.upper()}\n\n{len(places)} places",
                        key=f"theme_{theme}",
                        use_container_width=True
                ):
                    st.session_state.selected_theme = theme

        # Display places for selected theme
        if 'selected_theme' in st.session_state and st.session_state.selected_theme:
            selected_theme = st.session_state.selected_theme
            places = themes_dict[selected_theme]

            st.markdown("---")
            st.markdown(f"## {app.get_theme_icon(selected_theme)} {selected_theme.upper()}")
            st.markdown(f"*{len(places)} places found*")

            # Filter and sort options
            col1, col2, col3 = st.columns(3)
            with col1:
                sentiment_filter = st.selectbox(
                    "Filter by sentiment:",
                    ["All", "great", "good", "mixed", "neutral"]
                )
            with col2:
                sort_by = st.selectbox(
                    "Sort by:",
                    ["Name", "Sentiment", "Location"]
                )

            # Apply filters
            filtered_places = places
            if sentiment_filter != "All":
                filtered_places = [p for p in places if p['sentiment'] == sentiment_filter]

            # Apply sorting
            if sort_by == "Name":
                filtered_places = sorted(filtered_places, key=lambda x: x['place'])
            elif sort_by == "Sentiment":
                sentiment_order = {'great': 0, 'good': 1, 'mixed': 2, 'neutral': 3}
                filtered_places = sorted(filtered_places, key=lambda x: sentiment_order.get(x['sentiment'], 4))
            elif sort_by == "Location":
                filtered_places = sorted(filtered_places, key=lambda x: x['location'])

            st.markdown(f"*Showing {len(filtered_places)} places*")

            # Display places
            for idx, place in enumerate(filtered_places, 1):
                with st.container():
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"### {idx}. {place['place']} {app.get_sentiment_icon(place['sentiment'])}")
                    with col2:
                        sentiment_class = f"sentiment-{place['sentiment']}"
                        st.markdown(f"<p class='{sentiment_class}'>{place['sentiment'].upper()}</p>",
                                    unsafe_allow_html=True)

                    st.markdown(f"📍 **Location:** {place['location']}")
                    st.markdown(f"📝 **Notes:** {place['notes']}")

                    st.markdown("---")

# Footer
st.markdown("---")
st.markdown("*Powered by ChromaDB Cloud, FLAN-T5, and Streamlit*")