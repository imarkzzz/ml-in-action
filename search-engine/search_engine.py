class Bloomfilter(object):
  """
  A bloom filter is a probabilistic data-structure that trades space for accuracy when determining if a value is in a set. It can tell you if a value was possibly added, or if definitely not added, but it can't tell you for certain that it was added.
  """

  def __init__(self, size):
    """Setup the BF with the appropriate size"""
    self.values = [False]*size
    self.size = size

  def hash_value(self, value):
    """Hash the value provided and scale it to fit the BF size"""
    return hash(value) % self.size

  def add_value(self, value):
    """Add a value to the BF"""
    h = self.hash_value(value)
    self.values[h] = True

  def might_contain(self, value):
    """Check if the value might be in the BF"""
    h = self.hash_value(value)
    return self.values[h]
  
  def print_contents(self):
    "Dump the contents of the BF for debugging purpose"
    print(self.values)


def major_segments(s):
  """
  Perform major segmenting on a string. Split the string by all of the major breaks, and return the set of everything found. The breaks in this implementation are single characters, but in Splunk proper they can be multiple characters. A set is used because ordering doesn;'t matter, and duplicates are bad.
  """
  major_breaks = ' '


  last = -1
  results = set()

  for idx, ch in enumerate(s):
    if ch in major_breaks:
      segment = s[last+1:idx]
      results.add(segment)
      
      last = idx
  
  # The last character may not be a break so always capture
  segment = s[last+1:]
  results.add(segment)

  return results


def minor_segments(s):
  """
  Perform minor segmenting on a string. This is like major segmenting, except it also captures from the start of the input to each break.
  """
  minor_breaks = '_.'
  last = -1
  results = set()

  for idx, ch in enumerate(s):
    if ch in minor_breaks:
      segment = s[last+1:idx]
      results.add(segment)

      segment = s[:idx]
      results.add(segment)

      last = idx
  
  segment = s[last+1:]
  results.add(segment)
  results.add(s)

  return results


def segments(event):
  """Simple wrapper around major_segments / minor_segments"""
  results = set()
  for major in major_segments(event):
    # print(major)
    for minor in minor_segments(major):
      results.add(minor)
  return results

class Splunk(object):
  def __init__(self):
    self.bf = Bloomfilter(64)
    self.terms = {} # Dictionary of term to set of events
    self.events = []
  
  def add_event(self, event):
    """Adds an event to this object"""

    # Generate a unique ID for the event, and save it
    event_id = len(self.events)
    self.events.append(event)

    # Add each term to the bloomfilter, and track the event by each term
    for term in segments(event):
      self.bf.add_value(term)

      if term not in self.terms:
        self.terms[term] = set()
      self.terms[term].add(event_id)
  

  def search(self, term):
    """Search for a single term, and yield all the events that contain it"""

    # In Splunk this runs in O(1), and is likely to be in filesystem cache (memory)
    if not self.bf.might_contain(term):
      return
    
    # In Splunk this probably runs in O(log N) where N is the number of terms in the tsidx
    if term not in self.terms:
      return
    
    for event_id in sorted(self.terms[term]):
      yield self.events[event_id]

class SplunkX(object):
  def __init__(self):
    self.bf = Bloomfilter(64)
    self.terms = {} # Dictionary of term to set of events
    self.events = []
  
  def add_event(self, event):
    pass
  
  def search_all(self, terms):
    pass
  
  def search_any(self, terms):
    pass