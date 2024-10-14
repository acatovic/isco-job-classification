import json
import requests
import pandas as pd
from tqdm import tqdm
from time import sleep
import networkx as nx
from typing import Dict, Any, List


def get_esco_data(uri: str, endpoint: str) -> dict:
    """
    Fetches ESCO data from a specified API endpoint.

    Args:
        uri (str): The URI for the resource.
        endpoint (str): The API endpoint to fetch data from.

    Returns:
        dict: The JSON response as a dictionary.
    """
    params = {
        "uri": uri, 
        "language": "en",  # Desired reference language
        "selectedVersion": "v1.2.0"  # Optional, defaults to latest
    }

    # Define headers
    headers = {
        "Accept": "application/json,application/json;charset=UTF-8",
        "Content-Type": "application/json"
    }

    url = f"https://ec.europa.eu/esco/api/resource/{endpoint}"

    # Make the GET request
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Exception when calling the API: {e} for {uri}\n")
        return 'error'
    
class Vividict(dict):
    """
    A subclass of dict that returns another instance of itself when accessing
    a missing key, allowing creating nested dictionaries on the fly

    Example:
        v = Vividict()
        v['a']['b'] = 1  # Automatically creates nested dictionaries.
    """

    def __missing__(self, key: str) -> 'Vividict':
        """
        Handles missing keys by returning a new instance of Vividict.

        Args:
            key (str): The missing key.

        Returns:
            Vividict: A new instance for the missing key.
        """
        value = self.__class__()
        self[key] = value
        return value


def makde_dict(raw_data: Dict[str, Any], dict_type: type) -> Vividict:
    """
    Processes raw data into a custom Vividict structure.

    Args:
        raw_data (dict): The raw data containing fields such as 'title', 'uri', 
                         and optionally 'description'.
        dict_type (type): The type of dictionary to use (e.g., Vividict).

    Returns:
        Vividict: A nested dictionary structure with selected fields.
    """
    keep_dict = Vividict()
    keep_dict['code'] = code
    keep_fields = ['title','uri']
    for field in keep_fields:
        keep_dict[field] = raw_data[field]
    if raw_data.get('description'):
        keep_dict['description'] = raw_data['description']['en']['literal']
        for language, desc_dict in raw_data['description'].items():
            keep_dict['languages'][language]['description'] = desc_dict['literal']
    if raw_data.get('preferredLabel'):
        for language, label in raw_data['preferredLabel'].items():
            keep_dict['languages'][language]['preferredLabel'] = label
    if raw_data.get('alternativeLabel'):
        for language, label_list in raw_data['alternativeLabel'].items():
            keep_dict['languages'][language]['alternativeLabel'] = label_list
    if dict_type == 'occupations':
        for skill_type in ['hasEssentialSkill','hasOptionalSkill']:
            if raw_data['_links'].get(skill_type):
                keep_dict[skill_type] = [skill['title'] for skill in raw_data['_links'][skill_type]]
        if raw_data['_embedded']['ancestors']:
            for ancestor in raw_data['_embedded']['ancestors']:
                keep_dict['ancestors'][ancestor['title']] = ancestor['_links']['self']['uri']
        
    return keep_dict        


def split_nodes(code: str) -> List[str]:
    """
    Splits a string by periods and flattens the first segment into individual characters.

    Args:
        x (str): A string in the form of "1111.12.1".

    Returns:
        List[str]: A list of characters from the first segment and the rest of the split segments, 
        ie ["1", "1", "1", "12", "1"].
    """
    x = code.split('.')
    yo = list(x[0])
    yo.extend(x[1:])
    return yo


def get_parent(ancestors: str) -> str:
    """
    Retrieves the parent node by trimming the last segment from the split representation of the string.

    Args:
        x (str): A string in the form of a node identifier.

    Returns:
        str: The identifier for the parent node.
    """
    parent = ancestors[:-1]

    base = ''.join(parent[:4])
    if len(parent) > 4:
        base += '.' + '.'.join(parent[4:])

    return base


if __name__ == '__main__':
    # Define the path to the ESCO dataset directory
    ESCO_PATH = './ESCO_dataset'

    # Load ISCO group data (codes and URIs) from the CSV file
    df_isco = pd.read_csv(f'{ESCO_PATH}/ISCOGroups_en.csv', dtype={'code': str})
    isco_codes = df_isco['code'].tolist()          
    isco_uris = df_isco['conceptUri'].tolist()     
    isco_data = [x for x in zip(isco_codes, isco_uris)]  

    # Load ESCO occupations data (codes and URIs) from the CSV file
    df_esco = pd.read_csv(f'{ESCO_PATH}/occupations_en.csv', dtype={'iscoGroup': str})
    esco_codes = df_esco['code'].tolist()          
    esco_uris = df_esco['conceptUri'].tolist()     
    esco_data = [x for x in zip(esco_codes, esco_uris)]  # Pair each code with its URI

    # Combine ISCO and ESCO data and sort the list
    to_scrape = sorted(isco_data + esco_data)

    errors = []             # List to keep track of scraping errors
    occupation_dict = {}    # Dictionary to store processed occupation data

    # Scrape data from the ESCO API
    for code, uri in tqdm(to_scrape):
        data = get_esco_data(uri, 'occupation') 
        if data != 'error':                       
            occupation_dict[code] = makde_dict(data, 'occupations')  
        else:
            errors.append(code)
        sleep(.25)

    # Build a directed graph to track parent-child relationships among nodes (occupations)
    G = nx.DiGraph()
    for code in occupation_dict.keys():
        ancestors = split_nodes(code)
        if len(ancestors) > 1: 
            parent = get_parent(ancestors)  
            G.add_edge(parent, code)

    # Annotate each occupation with its level (depth) and whether it's a leaf node
    for id in occupation_dict.keys():
        occupation_dict[id]['level'] = len(split_nodes(id))
        occupation_dict[id]['is_leaf'] = G.out_degree(id) == 0

    # Write the processed occupation data to a JSON file
    with open(f'../data/occupations_data.json', 'w') as f:
        json.dump(occupation_dict, f, indent=4)
    
    # Write any errors to a separate JSON file
    with open('../data/errors.json', 'w') as f:
        json.dump(errors, f, indent=4)