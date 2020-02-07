"""
Read configurations from config.json.
"""
# Python Libraries
import os
import json
import logging

class DictClass(object):
    """
    Turns a dictionary into a class
    """
 
    def __init__(self, dictionary):
        """Constructor"""
        self.current_path = os.path.dirname(os.path.dirname(__file__))
        for key in dictionary:
            if (isinstance(dictionary[key], str)) and \
               ('/' in dictionary[key] or '\\' in dictionary[key]):
                dictionary[key] = os.path.join(self.current_path, dictionary[key])
                
                if ("dir" in key and not os.path.exists(dictionary[key])):
                    os.mkdir(dictionary[key]) 
            
            setattr(self, key, dictionary[key])
 
    def __repr__(self):
        """"""
        return "<DictClass: {}>".format(self.__dict__)
    
def loadConfig(file_name='config.json'):
    with open(file_name, "r") as fp:
        config = json.load(fp)
    
    return DictClass(config) 

log_Level = \
{
    "INFO":     logging.INFO,
    "WARNING":  logging.WARNING,
    "ERROR":    logging.ERROR
} 

if __name__ == '__main__':
    test = loadConfig()