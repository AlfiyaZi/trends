"""Visualizing Twitter Sentiment Across America"""

from data import word_sentiments, load_tweets
from datetime import datetime
from geo import us_states, geo_distance, make_position, longitude, latitude
from maps import draw_state, draw_name, draw_dot, wait
from string import ascii_letters
from ucb import main, trace, interact, log_current_line


###################################
# Phase 1: The Feelings in Tweets #
###################################

# The tweet abstract data type, implemented as a dictionary.

def make_tweet(text, time, lat, lon):
	"""Return a tweet, represented as a Python dictionary.

	text  -- A string; the text of the tweet, all in lowercase
	time  -- A datetime object; the time that the tweet was posted
	lat   -- A number; the latitude of the tweet's location
	lon   -- A number; the longitude of the tweet's location

	>>> t = make_tweet("just ate lunch", datetime(2012, 9, 24, 13), 38, 74)
	>>> tweet_text(t)
	'just ate lunch'
	>>> tweet_time(t)
	datetime.datetime(2012, 9, 24, 13, 0)
	>>> p = tweet_location(t)
	>>> latitude(p)
	38
	>>> tweet_string(t)
	'"just ate lunch" @ (38, 74)'
	"""
	return {'text': text, 'time': time, 'latitude': lat, 'longitude': lon}

def tweet_text(tweet):
	"""Return a string, the words in the text of a tweet."""
	return tweet['text'] #returns value of key 'text'

def tweet_time(tweet):
	"""Return the datetime representing when a tweet was posted."""
	return tweet['time'] #returns value of key 'time'

def tweet_location(tweet):
	"""Return a position representing a tweet's location."""
	return [tweet['latitude'],tweet['longitude']] #returns a list of the latitude and longitude, both values from the dictionary

# The tweet abstract data type, implemented as a function.

def make_tweet_fn(text, time, lat, lon):
	"""An alternate implementation of make_tweet: a tweet is a function.

	>>> t = make_tweet_fn("just ate lunch", datetime(2012, 9, 24, 13), 38, 74)
	>>> tweet_text_fn(t)
	'just ate lunch'
	>>> tweet_time_fn(t)
	datetime.datetime(2012, 9, 24, 13, 0)
	>>> latitude(tweet_location_fn(t))
	38
	"""
	def tweet(str_arg): #takes a string argument and returns the corresponding value
		if str_arg=='text':
			return text
		if str_arg=='time':
			return time
		if str_arg=='lat':
			return lat
		if str_arg=='lon':
			return lon
	return tweet #returns value of child function tweet

def tweet_text_fn(tweet):
	"""Return a string, the words in the text of a functional tweet."""
	return tweet('text') #returns the text of a tweet function taken as a parameter

def tweet_time_fn(tweet):
	"""Return the datetime representing when a functional tweet was posted."""
	return tweet('time') #returns the time of a tweet function taken as a parameter

def tweet_location_fn(tweet):
	"""Return a position representing a functional tweet's location."""
	return make_position(tweet('lat'), tweet('lon')) #returns the position object of a tweet function taken as a parameter

### === +++ ABSTRACTION BARRIER +++ === ###

def tweet_words(tweet):
	"""Return the words in a tweet."""
	return extract_words(tweet_text(tweet)) #calls extract_words which processes the text of the tweet

def tweet_string(tweet):
	"""Return a string representing a functional tweet."""
	location = tweet_location(tweet) #get the location as a position object
	point = (latitude(location), longitude(location))
	return '"{0}" @ {1}'.format(tweet_text(tweet), point) #returns the complete tweet

def extract_words(text):
	"""Return the words in a tweet, not including punctuation.

	>>> extract_words('anything else.....not my job')
	['anything', 'else', 'not', 'my', 'job']
	>>> extract_words('i love my job. #winning')
	['i', 'love', 'my', 'job', 'winning']
	>>> extract_words('make justin # 1 by tweeting #vma #justinbieber :)')
	['make', 'justin', 'by', 'tweeting', 'vma', 'justinbieber']
	>>> extract_words("paperclips! they're so awesome, cool, & useful!")
	['paperclips', 'they', 're', 'so', 'awesome', 'cool', 'useful']
	>>> extract_words('@(cat$.on^#$my&@keyboard***@#*')
	['cat', 'on', 'my', 'keyboard']
	"""
	from string import ascii_letters 
	start = 0
	words=[] #stores words as strings
	for i in range(len(text)):
		if i == 0 and text[i] not in ascii_letters: #if the first character of the string is not an ascii letter change the starting position
			start = 1
		if i > 0 and text[i] not in ascii_letters and text[i-1] in ascii_letters: #reached the end of the word
			words += [text[start:i]] #add the word to the list
		elif i > 0 and text[i] in ascii_letters and text[i-1] not in ascii_letters: #reached the beginning of a new word
			start=i #reset the starting position
		if i == len(text)-1 and text[i] in ascii_letters: #if the last letter of the string is an ascii letter
			words += [text[start:]] #add the final word to the list
	return words #return the list of words

def make_sentiment(value):
	"""Return a sentiment, which represents a value that may not exist.

	>>> positive = make_sentiment(0.2)
	>>> neutral = make_sentiment(0)
	>>> unknown = make_sentiment(None)
	>>> has_sentiment(positive)
	True
	>>> has_sentiment(neutral)
	True
	>>> has_sentiment(unknown)
	False
	>>> sentiment_value(positive)
	0.2
	>>> sentiment_value(neutral)
	0
	"""
	assert value is None or (value >= -1 and value <= 1), 'Illegal value'
	return value #simply returns the value

def has_sentiment(s):
	"""Return whether sentiment s has a value."""
	if s is None:
		return False #s does not have a sentiment
	return True #s has a sentiment

def sentiment_value(s):
	"""Return the value of a sentiment s."""
	assert has_sentiment(s), 'No sentiment value'
	return s #simply returns the value

def get_word_sentiment(word):
	"""Return a sentiment representing the degree of positive or negative
	feeling in the given word.

	>>> sentiment_value(get_word_sentiment('good'))
	0.875
	>>> sentiment_value(get_word_sentiment('bad'))
	-0.625
	>>> sentiment_value(get_word_sentiment('winning'))
	0.5
	>>> has_sentiment(get_word_sentiment('Berkeley'))
	False
	"""
	# Learn more: http://docs.python.org/3/library/stdtypes.html#dict.get
	return make_sentiment(word_sentiments.get(word))

def analyze_tweet_sentiment(tweet):
	""" Return a sentiment representing the degree of positive or negative
	sentiment in the given tweet, averaging over all the words in the tweet
	that have a sentiment value.

	If no words in the tweet have a sentiment value, return
	make_sentiment(None).

	>>> positive = make_tweet('i love my job. #winning', None, 0, 0)
	>>> round(sentiment_value(analyze_tweet_sentiment(positive)), 5)
	0.29167
	>>> negative = make_tweet("saying, 'i hate my job'", None, 0, 0)
	>>> sentiment_value(analyze_tweet_sentiment(negative))
	-0.25
	>>> no_sentiment = make_tweet("berkeley golden bears!", None, 0, 0)
	>>> has_sentiment(analyze_tweet_sentiment(no_sentiment))
	False
	"""
	from functools import reduce
	#if the number of words in the tweet that have sentiments equals 0 then return a sentiment of None
	if (sum(list(1 for a in tweet_words(tweet) if has_sentiment(get_word_sentiment(a))))) == 0: return make_sentiment(None)
	
	#returns the average, calculated using a generator function to generate the sum of the sentiment 
	#values of the words in the list, divided by the number of sentiment words in the tweet
	return make_sentiment((sum(list(sentiment_value(get_word_sentiment(a)) for a in tweet_words(tweet) if has_sentiment(get_word_sentiment(a))))) / (sum(list(1 for a in tweet_words(tweet) if has_sentiment(get_word_sentiment(a))))))


#################################
# Phase 2: The Geometry of Maps #
#################################

def find_centroid(polygon):
    """Find the centroid of a polygon.

    http://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon

    polygon -- A list of positions, in which the first and last are the same

    Returns: 3 numbers; centroid latitude, centroid longitude, and polygon area

    Hint: If a polygon has 0 area, use the latitude and longitude of its first
    position as its centroid.

    >>> p1, p2, p3 = make_position(1, 2), make_position(3, 4), make_position(5, 0)
    >>> triangle = [p1, p2, p3, p1]  # First vertex is also the last vertex
    >>> round5 = lambda x: round(x, 5) # Rounds floats to 5 digits
    >>> tuple(map(round5, find_centroid(triangle)))
    (3.0, 2.0, 6.0)
    >>> tuple(map(round5, find_centroid([p1, p3, p2, p1])))
    (3.0, 2.0, 6.0)
    >>> tuple(map(float, find_centroid([p1, p2, p1])))  # A zero-area polygon
    (1.0, 2.0, 0.0)
    """
    #Implements formula for calculating centroid of polygon with 2 steps: summation and then division
    center_x, center_y, area  = 0, 0, 0 #initialize
    #summation portion
    for entry in range(len(polygon)-1): 
      commonFactor = latitude(polygon[entry])*longitude(polygon[entry+1]) - latitude(polygon[entry+1])*longitude(polygon[entry]) #common factor used in all 3 formulas
      center_x += (latitude(polygon[entry]) + latitude(polygon[entry+1]))*commonFactor
      center_y += (longitude(polygon[entry]) + longitude(polygon[entry+1]))*commonFactor
      area += commonFactor
    if area == 0: return (latitude(polygon[0]), longitude(polygon[0]), 0.0) #edge case, if not real polygon
    #division portion
    area /= 2
    center_x /= 6*area
    center_y /= 6*area
    return (center_x, center_y, abs(area)) #abs prevents negative area

def find_state_center(polygons):
    """Compute the geographic center of a state, averaged over its polygons.

    The center is the average position of centroids of the polygons in polygons,
    weighted by the area of those polygons.

    Arguments:
    polygons -- a list of polygons

    >>> ca = find_state_center(us_states['CA'])  # California
    >>> round(latitude(ca), 5)
    37.25389
    >>> round(longitude(ca), 5)
    -119.61439

    >>> hi = find_state_center(us_states['HI'])  # Hawaii
    >>> round(latitude(hi), 5)
    20.1489
    >>> round(longitude(hi), 5)
    -156.21763
    """
    #implements area of polygon formula with 2 steps: summation and then division
    center_x, center_y, area_total = 0, 0, 0 #initialize
    #summation portion for component polygons
    for polygon in polygons:
      (c_x, c_y, area) = find_centroid(polygon)
      center_x += c_x*area
      center_y += c_y*area
      area_total += area
    #division portion
    center_x /= area_total
    center_y /= area_total
    return make_position(center_x, center_y)


###################################
# Phase 3: The Mood of the Nation #
###################################

def group_tweets_by_state(tweets):
    """Return a dictionary that aggregates tweets by their nearest state center.

    The keys of the returned dictionary are state names, and the values are
    lists of tweets that appear closer to that state center than any other.

    tweets -- a sequence of tweet abstract data types

    >>> sf = make_tweet("welcome to san francisco", None, 38, -122)
    >>> ny = make_tweet("welcome to new york", None, 41, -74)
    >>> two_tweets_by_state = group_tweets_by_state([sf, ny])
    >>> len(two_tweets_by_state)
    2
    >>> california_tweets = two_tweets_by_state['CA']
    >>> len(california_tweets)
    1
    >>> tweet_string(california_tweets[0])
    '"welcome to san francisco" @ (38, -122)'
    """
    tweets_by_state = {}
    states_centers = {state: find_state_center(us_states[state]) for state in us_states.keys()} #generates dictionary with states and their center positions   
    for tweet in tweets:
      closest = 999999999999 #initialize to very large distance value
      name = '' #initialize closest state name
      for state in states_centers:
        distance = geo_distance(tweet_location(tweet), states_centers[state]) #calculates distance to  all state centers 
        if distance < closest:
          closest = distance #saves closest distance and state name if new state is closer than previous best
          name = state
      #add tweet to appropriate entry or create new entry if nonexistent:
      if name not in tweets_by_state:
        tweets_by_state[name] = [tweet]
      elif name in tweets_by_state:
        tweets_by_state[name].append(tweet)
    return tweets_by_state

def average_sentiments(tweets_by_state):
	averaged_state_sentiments = {} #initialize dictionary with average sentiment for state
	for key in tweets_by_state.keys():
		list_of_tweets = tweets_by_state[key]
		sentiment_count = 0
		for i in range(len(list_of_tweets)): #checks each tweet's sentiment and adds each sentiment to list corresponding to state
			if has_sentiment(analyze_tweet_sentiment(list_of_tweets[i])) == False:
				list_of_tweets[i] = 0
			else:
				sentiment_count += 1 #count number of tweets with sentiments (needed in order to get average sentiment)
				list_of_tweets[i] = sentiment_value(analyze_tweet_sentiment(list_of_tweets[i]))
		if sentiment_count != 0:
			averaged_state_sentiments[key] = sum(list_of_tweets) / float(sentiment_count) #calculate average sentiment
	return averaged_state_sentiments

##########################
# Command Line Interface #
##########################

def print_sentiment(text='Are you virtuous or verminous?'):
    """Print the words in text, annotated by their sentiment scores."""
    words = extract_words(text.lower())
    layout = '{0:>' + str(len(max(words, key=len))) + '}: {1:+}'
    for word in words:
        s = get_word_sentiment(word)
        if has_sentiment(s):
            print(layout.format(word, sentiment_value(s)))

def draw_centered_map(center_state='TX', n=10):
    """Draw the n states closest to center_state."""
    us_centers = {n: find_state_center(s) for n, s in us_states.items()}
    center = us_centers[center_state.upper()]
    dist_from_center = lambda name: geo_distance(center, us_centers[name])
    for name in sorted(us_states.keys(), key=dist_from_center)[:int(n)]:
        draw_state(us_states[name])
        draw_name(name, us_centers[name])
    draw_dot(center, 1, 10)  # Mark the center state with a red dot
    wait()

def draw_state_sentiments(state_sentiments):
    """Draw all U.S. states in colors corresponding to their sentiment value.

    Unknown state names are ignored; states without values are colored grey.

    state_sentiments -- A dictionary from state strings to sentiment values
    """
    for name, shapes in us_states.items():
        sentiment = state_sentiments.get(name, None)
        draw_state(shapes, sentiment)
    for name, shapes in us_states.items():
        center = find_state_center(shapes)
        if center is not None:
            draw_name(name, center)

def draw_map_for_query(term='my job'):
    """Draw the sentiment map corresponding to the tweets that contain term.

    Some term suggestions:
    New York, Texas, sandwich, my life, justinbieber
    """
    tweets = load_tweets(make_tweet, term)
    tweets_by_state = group_tweets_by_state(tweets)
    state_sentiments = average_sentiments(tweets_by_state)
    draw_state_sentiments(state_sentiments)
    for tweet in tweets:
        s = analyze_tweet_sentiment(tweet)
        if has_sentiment(s):
            draw_dot(tweet_location(tweet), sentiment_value(s))
    wait()

def swap_tweet_representation(other=[make_tweet_fn, tweet_text_fn,
                                     tweet_time_fn, tweet_location_fn]):
    """Swap to another representation of tweets. Call again to swap back."""
    global make_tweet, tweet_text, tweet_time, tweet_location
    swap_to = tuple(other)
    other[:] = [make_tweet, tweet_text, tweet_time, tweet_location]
    make_tweet, tweet_text, tweet_time, tweet_location = swap_to


@main
def run(*args):
    """Read command-line arguments and calls corresponding functions."""
    import argparse
    parser = argparse.ArgumentParser(description="Run Trends")
    parser.add_argument('--print_sentiment', '-p', action='store_true')
    parser.add_argument('--draw_centered_map', '-d', action='store_true')
    parser.add_argument('--draw_map_for_query', '-m', action='store_true')
    parser.add_argument('--use_functional_tweets', '-f', action='store_true')
    parser.add_argument('text', metavar='T', type=str, nargs='*',
                        help='Text to process')
    args = parser.parse_args()
    if args.use_functional_tweets:
        swap_tweet_representation()
        print("Now using a functional representation of tweets!")
        args.use_functional_tweets = False
    for name, execute in args.__dict__.items():
        if name != 'text' and execute:
            globals()[name](' '.join(args.text))
