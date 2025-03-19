import asyncio
from playwright.async_api import async_playwright
import nltk
from nltk.corpus import stopwords
from collections import Counter
import time
import re

#Downloads the necessary NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

#Function to extract comments with improved reliability
async def fetch_comments(url, max_scroll_attempts=5):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = await context.new_page()
        
        try:
            #Navigates to the Reddit thread with timeout handling
            print(f"Navigating to {url}")
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            
            #Waits for the page to stabilize and bypass potential "Continue" buttons or consent dialogs
            try:
                continue_button = await page.query_selector("button:has-text('Continue')")
                if continue_button:
                    await continue_button.click()
                    await page.wait_for_timeout(2000)
            except Exception as e:
                print(f"No continue button found or error: {e}")
            
            #Detects if we're on the new or old Reddit page design
            is_new_reddit = await page.evaluate("window.location.href.includes('new.reddit') || !window.location.href.includes('old.reddit')")
            
            #Different selectors (inspect element) based on Reddit design
            comment_selector = "div[data-testid='comment']" if is_new_reddit else "div.thing.comment"
            
            #Waits for comments to appear
            try:
                print("Waiting for comments to load...")
                await page.wait_for_selector(comment_selector, timeout=20000)
            except Exception as e:
                print(f"Timeout waiting for comments: {e}")
                # Try an alternative selector before giving up
                alternative_selectors = [
                    "div[id^='t1_']", 
                    ".Comment", 
                    "shreddit-comment",
                    "div.entry div.usertext-body"
                ]
                
                for selector in alternative_selectors:
                    try:
                        print(f"Trying alternative selector: {selector}")
                        await page.wait_for_selector(selector, timeout=10000)
                        comment_selector = selector
                        print(f"Found comments with selector: {selector}")
                        break
                    except Exception:
                        continue
            
            #Progressively scrolls through the Reddit thread's webpage to load more comments
            comment_texts = []
            previous_comment_count = 0
            scroll_attempts = 0
            
            while scroll_attempts < max_scroll_attempts:
                #Extracts the currently visible comments
                if comment_selector == "shreddit-comment":
                    #For the new Reddit redesign with Web Components
                    comments = await page.evaluate("""
                        () => {
                            const comments = Array.from(document.querySelectorAll('shreddit-comment'));
                            return comments.map(comment => {
                                const paragraphs = comment.shadowRoot ? 
                                    Array.from(comment.shadowRoot.querySelectorAll('p')) : 
                                    Array.from(comment.querySelectorAll('p'));
                                return paragraphs.map(p => p.textContent).join(' ');
                            }).filter(text => text.trim().length > 0);
                        }
                    """)
                else:
                    #For standard HTML elements
                    comment_elements = await page.query_selector_all(f"{comment_selector} p, {comment_selector} div.md p")
                    comments = [await comment.inner_text() for comment in comment_elements]
                    comments = [text for text in comments if text.strip()]
                
                #Adds new comments to our collection
                comment_texts.extend(comments)
                
                #Removes duplicate comments from result
                comment_texts = list(dict.fromkeys(comment_texts))
                
                #Checks if new comments have been found while producing result
                if len(comment_texts) > previous_comment_count:
                    previous_comment_count = len(comment_texts)
                    print(f"Found {len(comment_texts)} comments so far")
                    scroll_attempts = 0  #Resets the counter if we found new comments
                else:
                    scroll_attempts += 1
                
                #Scrolls down to load more comments
                await page.evaluate("window.scrollBy(0, 1000)")
                await page.wait_for_timeout(2000)  #Waits for content to load
                
                #Tries to click "load more comments" buttons
                try:
                    more_buttons = await page.query_selector_all("button:has-text('more replies'), a.morecomments, button:has-text('Continue this thread')")
                    if more_buttons:
                        for button in more_buttons[:3]:  #Limits to first few to avoid endless clicking
                            try:
                                await button.click()
                                await page.wait_for_timeout(2000)
                            except Exception:
                                pass  #Ignores errors with individual buttons
                except Exception:
                    pass  #Continues if there's an issue with more buttons
            
            return comment_texts
            
        except Exception as e:
            print(f"Error during scraping: {e}")
            return []
        finally:
            await browser.close()

#Improved summarization using frequency analysis and key phrases
def improved_summarize(texts, num_sentences=3):
    if not texts:
        return "No comments available to summarize."
        
    #Combines all comments and cleans the text
    full_text = " ".join(texts)
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    
    #Tokenizes the comments into sentences
    sentences = nltk.sent_tokenize(full_text)
    
    if not sentences:
        return "No complete sentences found in comments."
    
    #If there are very few sentences, returns them all
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)
    
    #Gets stopwords
    stop_words = set(stopwords.words('english'))
    
    #Calculates word frequencies, ignoring stopwords
    word_frequencies = Counter()
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence.lower()):
            if word not in stop_words and word.isalnum():
                word_frequencies[word] += 1
    
    # Normalize frequencies
    max_frequency = max(word_frequencies.values()) if word_frequencies else 1
    normalized_frequencies = {word: freq/max_frequency for word, freq in word_frequencies.items()}
    
    #Scores sentences based on word frequencies
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in nltk.word_tokenize(sentence.lower()):
            if word in normalized_frequencies:
                if i not in sentence_scores:
                    sentence_scores[i] = 0
                sentence_scores[i] += normalized_frequencies[word]
    
    #Gets the top sentences
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
    top_sentences = sorted(top_sentences, key=lambda x: x[0])  # Sort by original position
    
    #Combines the top sentences
    summary = ' '.join([sentences[i] for i, _ in top_sentences])
    
    return summary

#Mains function
async def main():
    url = "https://www.reddit.com/r/clevercomebacks/comments/1jb87ed/asked_and_answered_i_guess/" #Replace the URL with the URL of the website you want to scrape and summarize.
    
    print("Starting Reddit comment extraction...")
    start_time = time.time()
    
    comments = await fetch_comments(url)
    
    if comments:
        print(f"\nExtracted {len(comments)} comments in {time.time() - start_time:.2f} seconds")
        
        #Prints sample comments
        if len(comments) > 0:
            print("\nSample comments:")
            for i, comment in enumerate(comments[:3]):
                print(f"{i+1}. {comment[:100]}..." if len(comment) > 100 else f"{i+1}. {comment}")
        
        #Generates the Reddit thread summary
        print("\nGenerating summary...")
        summary = improved_summarize(comments, num_sentences=3)
        print(f"\nSummary of Comments:\n{summary}")
        
        #Extracts the key themes
        words = []
        for comment in comments:
            words.extend([w.lower() for w in nltk.word_tokenize(comment) if w.isalpha() and w.lower() not in stopwords.words('english')])
        
        common_words = Counter(words).most_common(10)
        print("\nKey themes in comments:")
        for word, count in common_words:
            print(f"- '{word}' mentioned {count} times")
    else:
        print("No comments were extracted. Please check the URL or try adjusting the selectors.")

if __name__ == "__main__":
    asyncio.run(main())
