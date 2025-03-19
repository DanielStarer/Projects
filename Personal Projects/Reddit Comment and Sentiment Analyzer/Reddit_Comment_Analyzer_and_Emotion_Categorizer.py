import asyncio
from playwright.async_api import async_playwright
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter, defaultdict
import time
import re
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#Downloads necessary NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("vader_lexicon", quiet=True)

#Function to extract comments (change max scroll attempts to a higher number if inspecting massive thread)
async def fetch_comments(url, max_scroll_attempts=5):
    print("Using the new enhanced comment scraper!")
    """Enhanced version with better debugging and updated selectors"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            viewport={"width": 1920, "height": 1080}  #Usse a 1920 x 1080 viewport resolution
        )
        page = await context.new_page()
        
        try:
            print(f"Navigating to {url}")
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            
            #Debugging: Saves a screenshot
            await page.screenshot(path="reddit_page.png")
            print("Screenshot saved as reddit_page.png")
            
            #Debugging: Saves page content
            html_content = await page.content()
            with open("reddit_page.html", "w", encoding="utf-8") as f:
                f.write(html_content)
            print("HTML content saved as reddit_page.html")
            
            #Handles potential interruptions
            try:
                #Handles "Accept all cookies" prompt
                cookie_button = await page.query_selector("button:has-text('Accept all')")
                if cookie_button:
                    print("Found cookie consent button, clicking...")
                    await cookie_button.click()
                    await page.wait_for_timeout(30000)
                
                #Handles "Continue" button or "See more" button
                for button_text in ["Continue", "See more", "View More Comments", "Continue this thread"]:
                    continue_button = await page.query_selector(f"button:has-text('{button_text}')")
                    if continue_button:
                        print(f"Found '{button_text}' button, clicking...")
                        await continue_button.click()
                        await page.wait_for_timeout(30000)
            except Exception as e:
                print(f"Handled button interaction error: {e}")
            
            #Waits for the page to stabilize
            await page.wait_for_timeout(30000)
            
            #Tries to detect the Reddit version for the URL provided with more robustness
            print("Detecting Reddit version...")
            is_new_reddit = True  #Defaults to the new version Reddit
            try:
                old_reddit_indicator = await page.query_selector("body.listing-page, div.commentarea")
                if old_reddit_indicator:
                    is_new_reddit = False
                    print("Detected old Reddit design")
                else:
                    print("Detected new Reddit design")
            except Exception:
                print("Could not detect Reddit version, assuming new Reddit")
            
            #Comprehensive list of selectors to try
            #Checks each one and uses the first that works
            print("Trying multiple selector strategies...")
            all_selectors = [
                #New Reddit Selectors
                "shreddit-comment",  #Web component selector
                "div[data-testid='comment']",  #Reaction/comment test ID
                "div[data-test-id='comment-top-meta']",  #Another possible test ID
                "div.Comment",  #Class-based selector
                "faceplate-partial[elementtiming='bfcache-comment']",  #Web component
                
                #Old Reddit Selectors
                "div.thing.comment",  #Classic old.reddit
                "div.entry div.usertext-body",  #Old Reddit comment body
                
                #General Selectors
                "div[id^='t1_']",  #Comment ID format both designs use
                ".md"  #Markdown content class
            ]
            
            #Tests each selector and see which ones find elements
            working_selectors = []
            for selector in all_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    count = len(elements)
                    print(f"Selector '{selector}': found {count} elements")
                    if count > 0:
                        working_selectors.append((selector, count))
                except Exception as e:
                    print(f"Error with selector '{selector}': {e}")
            
            #Chooses the best selector (the one that found the most elements)
            if working_selectors:
                working_selectors.sort(key=lambda x: x[1], reverse=True)
                comment_selector = working_selectors[0][0]
                print(f"Using selector '{comment_selector}' which found {working_selectors[0][1]} elements")
            else:
                print("No working selectors found!")
                return []
            
            #Gets comments using our best selector
            print(f"Extracting comments with selector: {comment_selector}")
            
            comment_texts = []
            scroll_attempts = 0
            previous_count = 0
            
            #Checks if we're dealing with web components
            is_web_component = comment_selector == "shreddit-comment" or comment_selector == "faceplate-partial"
            
            while scroll_attempts < max_scroll_attempts:
                try:
                    if is_web_component:
                        #Special handling for Shadow DOM / Web Components
                        comments = await page.evaluate(f"""
                            () => {{
                                const comments = Array.from(document.querySelectorAll('{comment_selector}'));
                                return comments.map(comment => {{
                                    // Try to get text from shadow DOM first
                                    if (comment.shadowRoot) {{
                                        const paragraphs = Array.from(comment.shadowRoot.querySelectorAll('p'));
                                        if (paragraphs.length) {{
                                            return paragraphs.map(p => p.textContent).join(' ');
                                        }}
                                    }}
                                    
                                    // Fallback to regular DOM
                                    const paragraphs = Array.from(comment.querySelectorAll('p'));
                                    return paragraphs.map(p => p.textContent).join(' ');
                                }}).filter(text => text && text.trim().length > 0);
                            }}
                        """)
                    else:
                        #Standard DOM elements
                        elements = await page.query_selector_all(f"{comment_selector} p, {comment_selector} div.md p")
                        if not elements:
                            #Tries without the p tag
                            elements = await page.query_selector_all(comment_selector)
                        
                        comments = []
                        for element in elements:
                            text = await element.inner_text()
                            if text.strip():
                                comments.append(text)
                    
                    #Adds new unique comments
                    for comment in comments:
                        if comment not in comment_texts:
                            comment_texts.append(comment)
                    
                    current_count = len(comment_texts)
                    print(f"Found {current_count} unique comments so far")
                    
                    #Checks if new comments have been found
                    if current_count > previous_count:
                        previous_count = current_count
                        scroll_attempts = 0  #Resets the counter if we found new comments
                    else:
                        scroll_attempts += 1
                    
                    #Tries to expand comments
                    await expand_comments(page)
                    
                    #Scrolls to load more
                    await page.evaluate("window.scrollBy(0, 300)")
                    await page.wait_for_timeout(100000)  # Short wait between scrolls
                
                except Exception as e:
                    print(f"Error during comment extraction: {e}")
                    scroll_attempts += 1
            
            print(f"Finished extracting comments, found {len(comment_texts)} unique comments")
            return comment_texts
            
        except Exception as e:
            print(f"Critical error during scraping: {e}")
            return []
        finally:
            await browser.close()

async def expand_comments(page):
    """Try to expand collapsed comments and load more comments"""
    try:
        #Clicks the "more comments" buttons
        for button_text in ['more comments', 'View More Comments', 'Continue this thread', 'show more', 'load more']:
            buttons = await page.query_selector_all(f"button:has-text('{button_text}'), a:has-text('{button_text}')")
            if buttons:
                print(f"Found {len(buttons)} '{button_text}' buttons to click")
                for button in buttons[:3]:  #Limits to the first 3 to avoid endless clicking
                    try:
                        await button.scroll_into_view_if_needed()
                        await button.click()
                        await page.wait_for_timeout(30000)
                        print(f"Clicked a '{button_text}' button")
                    except Exception as e:
                        print(f"Couldn't click a '{button_text}' button: {e}")
        
        #Handles Reddit-specific "more replies" buttons
        more_buttons = await page.query_selector_all("button._1YCqQVO-9r-Up6QPB9H6_4")
        if more_buttons:
            print(f"Found {len(more_buttons)} 'more replies' buttons")
            for button in more_buttons[:3]:
                try:
                    await button.scroll_into_view_if_needed()
                    await button.click()
                    await page.wait_for_timeout(30000)
                except Exception:
                    pass
        
        #Expands the more comment buttons (+ signs)
        expanders = await page.query_selector_all("[data-testid='expand-button'], .expand")
        if expanders:
            print(f"Found {len(expanders)} comment expanders")
            for expander in expanders[:5]:
                try:
                    await expander.scroll_into_view_if_needed()
                    await expander.click()
                    await page.wait_for_timeout(10000)
                except Exception:
                    pass
    except Exception as e:
        print(f"Error while expanding comments: {e}")

#Emotion Analysis Class
class RedditEmotionAnalyzer:
    def __init__(self, comments):
        self.comments = comments
        self.sia = SentimentIntensityAnalyzer()
        
        #Emotion lexicons - these are simplified for demonstration
        self.emotion_lexicons = {
            'joy': ['happy', 'exciting', 'glad', 'joy', 'delighted', 'pleased', 'love', 'awesome', 
                   'amazing', 'wonderful', 'great', 'lol', 'haha', 'lmao', ':)', '❤️'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'upset', 'hate', 
                     'disgusting', 'terrible', 'worst', 'stupid', 'idiot', 'wtf', 'bs'],
            'sadness': ['sad', 'unhappy', 'depressed', 'disappointed', 'unfortunate', 'miserable', 
                       'sorry', 'regret', 'miss', 'lonely', 'cry', 'tears', ':('],
            'fear': ['afraid', 'scared', 'frightened', 'worried', 'anxious', 'concerned', 'panic', 
                    'terror', 'horror', 'nervous', 'yikes'],
            'surprise': ['surprised', 'shocked', 'amazed', 'unexpected', 'wow', 'whoa', 'omg', 
                        'what', 'unbelievable', 'incredible'],
            'disgust': ['disgusting', 'gross', 'nasty', 'repulsive', 'ew', 'yuck', 'distasteful', 'eww'],
            'confusion': ['confused', 'unsure', 'uncertain', 'puzzled', 'baffled', 'not sure', 
                         'perplexed', 'dont understand', 'huh', 'wtf']
        }
        
        #Compiles regular expressions for faster matching
        self.emotion_patterns = {}
        for emotion, words in self.emotion_lexicons.items():
            #Escape special characters and ensure proper regex formatting
            escaped_words = [re.escape(word) for word in words]
            pattern_string = r'\b(' + '|'.join(escaped_words) + r')\b'
            self.emotion_patterns[emotion] = re.compile(pattern_string, re.IGNORECASE)

    def analyze_vader_sentiment(self):
        """Uses VADER for sentiment analysis"""
        sentiments = []
        for comment in self.comments:
            score = self.sia.polarity_scores(comment)
            #Determines the overall sentiment based on compound score
            if score['compound'] >= 0.05:
                sentiment = 'positive'
            elif score['compound'] <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
                
            sentiments.append({
                'text': comment[:100] + "..." if len(comment) > 100 else comment,
                'sentiment': sentiment,
                'compound': score['compound'],
                'pos': score['pos'],
                'neu': score['neu'],
                'neg': score['neg']
            })
        return sentiments
    
    def analyze_textblob_sentiment(self):
        """Uses TextBlob for sentiment and subjectivity analysis"""
        results = []
        for comment in self.comments:
            analysis = TextBlob(comment)
            polarity = analysis.sentiment.polarity
            subjectivity = analysis.sentiment.subjectivity
            
            #Determines sentiment category
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
                
            #Determines subjectivity category
            if subjectivity > 0.6:
                subj_category = 'highly subjective'
            elif subjectivity > 0.3:
                subj_category = 'somewhat subjective'
            else:
                subj_category = 'objective'
                
            results.append({
                'text': comment[:100] + "..." if len(comment) > 100 else comment,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': sentiment,
                'subjectivity_category': subj_category
            })
        return results
    
    def detect_emotions(self):
        """Detects specific emotions using lexicon-based approach"""
        results = []
        
        for comment in self.comments:
            emotion_counts = {emotion: 0 for emotion in self.emotion_lexicons.keys()}
            
            #Checks each emotion pattern
            for emotion, pattern in self.emotion_patterns.items():
                matches = pattern.findall(comment.lower())
                emotion_counts[emotion] = len(matches)
            
            #Determines primary emotion (if any)
            total_emotions = sum(emotion_counts.values())
            primary_emotion = max(emotion_counts.items(), key=lambda x: x[1]) if total_emotions > 0 else ('none', 0)
            
            #Adds the sentiments to the results
            results.append({
                'text': comment[:100] + "..." if len(comment) > 100 else comment,
                'emotions': emotion_counts,
                'primary_emotion': primary_emotion[0] if primary_emotion[1] > 0 else 'neutral',
                'emotion_intensity': sum(emotion_counts.values()) / len(comment.split()) if comment.split() else 0
            })
            
        return results
    
    def perform_topic_modeling(self, num_topics=3, num_words=10):
        """Identifies key topics in the comments using LDA"""
        #Vectorizes the text
        vectorizer = CountVectorizer(
            max_df=0.95, min_df=2, 
            stop_words='english', 
            token_pattern=r'\b[a-zA-Z]{3,}\b'  #Only words with 3+ characters
        )
        
        try:
            dtm = vectorizer.fit_transform(self.comments)
            
            #Creates and fits the LDA model
            lda = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42,
                max_iter=10
            )
            lda.fit(dtm)
            
            #Gets feature names
            feature_names = vectorizer.get_feature_names_out()
            
            #Collects topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-num_words-1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weight': topic.sum()
                })
                
            #Finds the dominant topic for each comment
            comment_topics = []
            doc_topics = lda.transform(dtm)
            for i, comment in enumerate(self.comments):
                dominant_topic = np.argmax(doc_topics[i])
                comment_topics.append({
                    'text': comment[:100] + "..." if len(comment) > 100 else comment,
                    'dominant_topic': dominant_topic,
                    'topic_words': topics[dominant_topic]['words'][:5],  # Top 5 words
                    'topic_score': doc_topics[i][dominant_topic]
                })
                
            return {
                'topics': topics,
                'comment_topics': comment_topics
            }
        except ValueError as e:
            print(f"Error in topic modeling: {e}")
            return {
                'topics': [],
                'comment_topics': []
            }
    
    def analyze_all(self):
        """Performs all types of analysis and returns comprehensive results"""
        print("Analyzing sentiment with VADER...")
        vader_results = self.analyze_vader_sentiment()
        
        print("Analyzing sentiment with TextBlob...")
        textblob_results = self.analyze_textblob_sentiment()
        
        print("Detecting specific emotions...")
        emotion_results = self.detect_emotions()
        
        print("Performing topic modeling...")
        topic_results = self.perform_topic_modeling()
        
        #Compiles the overall sentiment statistics
        vader_sentiments = [r['sentiment'] for r in vader_results]
        textblob_sentiments = [r['sentiment'] for r in textblob_results]
        primary_emotions = [r['primary_emotion'] for r in emotion_results]
        
        sentiment_stats = {
            'vader_sentiment_counts': Counter(vader_sentiments),
            'textblob_sentiment_counts': Counter(textblob_sentiments),
            'primary_emotion_counts': Counter(primary_emotions)
        }
        
        #Calculates the emotion intensity across all comments
        all_emotions = defaultdict(int)
        for result in emotion_results:
            for emotion, count in result['emotions'].items():
                all_emotions[emotion] += count
        
        return {
            'vader_results': vader_results,
            'textblob_results': textblob_results,
            'emotion_results': emotion_results,
            'topics': topic_results,
            'overall_stats': sentiment_stats,
            'emotion_totals': dict(all_emotions)
        }
    
    def generate_summary(self, analysis_results):
        """Generates a human-readable summary of the emotion analysis"""
        #Gets the overall statistics
        stats = analysis_results['overall_stats']
        emotion_totals = analysis_results['emotion_totals']
        
        #Calculates percentages
        total_comments = len(self.comments)
        vader_percentages = {k: (v/total_comments)*100 for k, v in stats['vader_sentiment_counts'].items()}
        
        #Gets the most common emotions
        sorted_emotions = sorted(emotion_totals.items(), key=lambda x: x[1], reverse=True)
        
        #Gets the most relevant topics
        topics = analysis_results['topics'].get('topics', [])
        sorted_topics = sorted(topics, key=lambda x: x['weight'], reverse=True) if topics else []
        
        #Builds the summary
        summary = "# Emotional Analysis Summary\n\n"
        
        #Figures out the overall sentiment
        summary += "## Overall Sentiment\n"
        summary += f"Based on {total_comments} comments:\n"
        summary += f"- Positive: {vader_percentages.get('positive', 0):.1f}%\n"
        summary += f"- Neutral: {vader_percentages.get('neutral', 0):.1f}%\n"
        summary += f"- Negative: {vader_percentages.get('negative', 0):.1f}%\n\n"
        
        #Dominant emotions
        summary += "## Dominant Emotions\n"
        for emotion, count in sorted_emotions[:5]:  #Top 5 emotions
            if count > 0:
                summary += f"- {emotion.capitalize()}: {count} instances\n"
        summary += "\n"
        
        #Main topics
        if sorted_topics:
            summary += "## Main Discussion Topics\n"
            for i, topic in enumerate(sorted_topics[:3]):  #Top 3 topics
                summary += f"Topic {i+1}: {', '.join(topic['words'][:5])}\n"
        
        #Provides a sample summary of the emotional comments
        summary += "\n## Sample Emotional Comments\n"
        
        #Finds a positive comment
        positive_comments = [r for r in analysis_results['vader_results'] 
                            if r['sentiment'] == 'positive' and r['compound'] > 0.5]
        if positive_comments:
            comment = max(positive_comments, key=lambda x: x['compound'])
            summary += f"**Positive**: \"{comment['text']}\"\n"
        
        #Finds a negative comment
        negative_comments = [r for r in analysis_results['vader_results'] 
                            if r['sentiment'] == 'negative' and r['compound'] < -0.5]
        if negative_comments:
            comment = min(negative_comments, key=lambda x: x['compound'])
            summary += f"**Negative**: \"{comment['text']}\"\n"
            
        #Finds a comment with a strong emotion
        emotional_comments = [r for r in analysis_results['emotion_results'] 
                             if r['emotion_intensity'] > 0.1]
        if emotional_comments:
            comment = max(emotional_comments, key=lambda x: x['emotion_intensity'])
            summary += f"**Emotional ({comment['primary_emotion']})**:  \"{comment['text']}\"\n"
        
        return summary

#Visualization functions
def visualize_sentiment(analysis_results, output_file="sentiment_analysis.png"):
    """Creates visualizations of sentiment analysis results"""
    stats = analysis_results['overall_stats']
    
    #Sets up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    #Plots the VADER sentiment distribution
    vader_counts = stats['vader_sentiment_counts']
    labels = list(vader_counts.keys())
    values = list(vader_counts.values())
    colors = ['green', 'gray', 'red']
    
    ax1.bar(labels, values, color=colors)
    ax1.set_title('VADER Sentiment Distribution')
    ax1.set_ylabel('Number of Comments')
    
    #Plots emotion distribution
    emotion_totals = analysis_results['emotion_totals']
    emotions = list(emotion_totals.keys())
    counts = list(emotion_totals.values())
    
    #Sorts by count
    sorted_data = sorted(zip(emotions, counts), key=lambda x: x[1], reverse=True)
    emotions = [item[0] for item in sorted_data]
    counts = [item[1] for item in sorted_data]
    
    #Takes the top 7 emotions for readability
    emotions = emotions[:7]
    counts = counts[:7]
    
    #Emotion colors
    emotion_colors = {
        'joy': 'gold',
        'sadness': 'steelblue',
        'anger': 'firebrick',
        'fear': 'darkviolet',
        'surprise': 'darkorange',
        'disgust': 'darkgreen',
        'confusion': 'slategray',
        'neutral': 'lightgray'
    }
    
    colors = [emotion_colors.get(emotion, 'gray') for emotion in emotions]
    
    ax2.bar(emotions, counts, color=colors)
    ax2.set_title('Emotion Distribution')
    ax2.set_ylabel('Number of Instances')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")

#Main function to orchestrate the process
async def main():
    url = "https://www.reddit.com/r/clevercomebacks/comments/1jb87ed/asked_and_answered_i_guess/" # Replace this URL with the URL of your choosing for the program to pull data from
    
    print("Starting Reddit comment extraction...")
    start_time = time.time()
    
    comments = await fetch_comments(url)
    
    if comments:
        print(f"\nExtracted {len(comments)} comments in {time.time() - start_time:.2f} seconds")
        
        #Prints some sample comments
        if len(comments) > 0:
            print("\nSample comments:")
            for i, comment in enumerate(comments[:3]):
                print(f"{i+1}. {comment[:100]}..." if len(comment) > 100 else f"{i+1}. {comment}")
        
        #Creates an emotion analyzer
        print("\nInitializing emotion analysis...")
        analyzer = RedditEmotionAnalyzer(comments)
        
        #Performs an analysis on the results
        analysis_results = analyzer.analyze_all()
        
        #Generates and prints a summary of the results
        print("\n" + "="*80)
        summary = analyzer.generate_summary(analysis_results)
        print(summary)
        
        #Saves the summary to a file (located in same directory where the file is located)
        with open("reddit_emotion_summary.md", "w", encoding="utf-8") as f:
            f.write(summary)
        print("Summary saved to reddit_emotion_summary.md")
        
        #Creates visualizations of the results
        try:
            visualize_sentiment(analysis_results)
        except Exception as e:
            print(f"Error creating visualizations: {e}")
        
        #Saves a detailed analysis to CSV
        try:
            #VADER sentiment results
            vader_df = pd.DataFrame(analysis_results['vader_results'])
            vader_df.to_csv("vader_sentiment_results.csv", index=False)
            
            #Emotion detection results
            emotion_results = analysis_results['emotion_results']
            #Flatten emotion dictionaries
            for result in emotion_results:
                emotions = result.pop('emotions')
                for emotion, count in emotions.items():
                    result[f"emotion_{emotion}"] = count
            
            emotion_df = pd.DataFrame(emotion_results)
            emotion_df.to_csv("emotion_detection_results.csv", index=False)
            
            print("Detailed analysis results saved to CSV files")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")
        
        #Gets insights about specific emotions
        joy_comments = [r for r in analysis_results['emotion_results'] 
                       if r['primary_emotion'] == 'joy' and any(r['emotions'].values())]
        
        if joy_comments:
            print("\nSample Joyful Comments:")
            for comment in sorted(joy_comments, 
                                 key=lambda x: sum(x['emotions'].values()), 
                                 reverse=True)[:3]:
                print(f"- {comment['text']}")
        
        anger_comments = [r for r in analysis_results['emotion_results'] 
                         if r['primary_emotion'] == 'anger' and any(r['emotions'].values())]
        
        if anger_comments:
            print("\nSample Angry Comments:")
            for comment in sorted(anger_comments, 
                                 key=lambda x: sum(x['emotions'].values()), 
                                 reverse=True)[:3]:
                print(f"- {comment['text']}")
                
        #Analyzes the comment themes by sentiment
        print("\nAnalyzing themes by sentiment...")
        positive_comments = [r['text'] for r in analysis_results['vader_results'] if r['sentiment'] == 'positive']
        negative_comments = [r['text'] for r in analysis_results['vader_results'] if r['sentiment'] == 'negative']
        
        #Gets keywords for positive comments
        if positive_comments:
            positive_text = ' '.join(positive_comments)
            positive_words = [word.lower() for word in nltk.word_tokenize(positive_text) 
                             if word.isalpha() and word.lower() not in stopwords.words('english')]
            positive_keywords = Counter(positive_words).most_common(10)
            
            print("\nPositive Comment Keywords:")
            for word, count in positive_keywords:
                print(f"- '{word}': {count} occurrences")
                
        #Gets keywords for negative comments
        if negative_comments:
            negative_text = ' '.join(negative_comments)
            negative_words = [word.lower() for word in nltk.word_tokenize(negative_text) 
                             if word.isalpha() and word.lower() not in stopwords.words('english')]
            negative_keywords = Counter(negative_words).most_common(10)
            
            print("\nNegative Comment Keywords:")
            for word, count in negative_keywords:
                print(f"- '{word}': {count} occurrences")
                
    else:
        print("No comments were extracted. Please check the URL or try adjusting the selectors.")

#Creates a function to analyze a specific subreddit over multiple posts
async def analyze_subreddit(subreddit_name, num_posts=5):
    #URL pattern for getting the top posts from a subreddit
    url = f"https://www.reddit.com/r/{subreddit_name}/top/?t=month"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = await context.new_page()
        
        try:
            print(f"Navigating to r/{subreddit_name} to collect top posts...")
            await page.goto(url, wait_until="domcontentloaded", timeout=500000)
            
            #Waits for posts to load
            await page.wait_for_timeout(5000)
            
            #Extracts the post URLs
            post_links = await page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[data-click-id="body"][data-testid="post-title"]'));
                    return links.slice(0, arguments[0]).map(link => link.href);
                }
            """, num_posts)
            
            print(f"Found {len(post_links)} posts to analyze")
            
            #Processes each post
            all_comments = []
            for i, post_link in enumerate(post_links):
                print(f"\nAnalyzing post {i+1}/{len(post_links)}: {post_link}")
                comments = await fetch_comments(post_link)
                print(f"  - Extracted {len(comments)} comments")
                all_comments.extend(comments)
            
            print(f"\nTotal comments collected: {len(all_comments)}")
            
            #Analyzes all comments together
            if all_comments:
                analyzer = RedditEmotionAnalyzer(all_comments)
                analysis_results = analyzer.analyze_all()
                
                #Generates a summary
                summary = analyzer.generate_summary(analysis_results)
                
                #Saves the results
                with open(f"r_{subreddit_name}_emotion_analysis.md", "w", encoding="utf-8") as f:
                    f.write(f"# Emotional Analysis of r/{subreddit_name}\n\n")
                    f.write(f"Analysis based on {len(all_comments)} comments from {len(post_links)} top posts.\n\n")
                    f.write(summary)
                
                print(f"\nAnalysis complete! Results saved to r_{subreddit_name}_emotion_analysis.md")
                
                # Visualize
                try:
                    visualize_sentiment(analysis_results, output_file=f"r_{subreddit_name}_sentiment.png")
                except Exception as e:
                    print(f"Error creating visualizations: {e}")
            else:
                print("No comments were collected. Please check the subreddit or try adjusting the selectors.")
        
        except Exception as e:
            print(f"Error during subreddit analysis: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    # For single post analysis
    asyncio.run(main())
    
    # For subreddit analysis, uncomment the following:
    # asyncio.run(analyze_subreddit("AskReddit", num_posts=3))
