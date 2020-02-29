# Name: Menghe Dou
# UW NetID: md4492
# Homework 3: Election prediction

import csv
import os
import time

def read_csv(path):
    """
    Reads the CSV file at path, and returns a list of rows. Each row is a
    dictionary that maps a column name to a value in that column, as a string.
    """
    output = []
    for row in csv.DictReader(open(path)):
        output.append(row)
    return output

################################################################################
# Problem 1: State edges
################################################################################

def row_to_edge(row):
    """
    Given an election result row or poll data row, returns the Democratic edge
    in that state.
    """
    return float(row["Dem"]) - float(row["Rep"])  

def state_edges(election_result_rows):
    """
    Given a list of election result rows, returns state edges.
    The input list does has no duplicate states;
    that is, each state is represented at most once in the input list.
    """
    #create a dictionary to save result
    stage_edges_dict= {}
    
    for row in election_result_rows:
        # make values in new dict equal edges
        value = row_to_edge(row)
        # iterate rows to update values in stage_edges_dict
        stage_edges_dict.update({row['State']:value})
    # output the result    
    return stage_edges_dict

"""
Report: In this question, we need to create a new dictionary to save the states
and thier edges, and make the key equal the state, make the value equal to edges.
For dictionary, use update to add items in it.
"""
    
################################################################################
# Problem 2: Find the most recent poll row
################################################################################

def earlier_date(date1, date2):
    """
    Given two dates as strings (formatted like "Oct 06 2012"), returns True if 
    date1 is after date2.
    """
    return (time.strptime(date1, "%b %d %Y") < time.strptime(date2, "%b %d %Y"))

earlier_date("Jan 01 1900", "Jan 01 2010")

def most_recent_poll_row(poll_rows, pollster, state):
    """
    Given a list of poll data rows, returns the most recent row with the
    specified pollster and state. If no such row exists, returns None.
    """
    # create a very early date to do the first comparision
    a = "Jan 01 2000"
    # make result equal none at first
    result = None
    for rows in poll_rows:
        # ensure in the row with same state and same pollster
        if rows['State'] == state and rows['Pollster'] == pollster:
            # to compare their date
            if earlier_date(a, rows['Date']) == True:
                # then make "a" equal the nearst date
                a = rows['Date']
                # output the rows with the nearst date
                result = rows
        
    return result

"""
Report: We need to find the nearest date under the same state and pollster, and
return the whole row when input a given pollster and state. To achieve it, two
if conditions are needed: first match the state and pollster, and second use 
earlier_date function to select the most recent date.
"""
 
################################################################################
# Problem 3: Pollster predictions
################################################################################

def unique_column_values(rows, column_name):
    """
    Given a list of rows and the name of a column (a string), returns a set
    containing all values in that column.
    """
    # create a dictionary to save column values
    column_dict = {}
    for i in rows:
        # select values in the give column and make values be the key
        keys = i[column_name]
        # iterate rows to update dictionary
        column_dict.update({keys:()})
    # return the keys of column_dict
    return column_dict.keys()
  
"""
Report: This question needs to ensure the output be exclusive, which is same to
the attribute of the key, so I use key to save the values of column, and it doesn't
matter what values in new dictionary, so I just make it empty.
"""

def pollster_predictions(poll_rows):
    """
    Given a list of poll data rows, returns pollster predictions.
    """
    # create a list to select pollster and state in the give data
    most_recent_rows = []
    # save a list of pollsters and states in given data
    poll = unique_column_values(poll_rows,"Pollster")
    state = unique_column_values(poll_rows,"State")

    for p in poll:
        for s in state:
            # select the row with most recent date in given pollsters and states
            most_recent_rows.append(most_recent_poll_row(poll_rows, p, s))
    print(most_recent_rows)
    # defaultdict can achive some functions of dictionary
    from collections import defaultdict
    
    # create a dictionary to save result
    prediction = defaultdict(dict)
   
    for i in most_recent_rows:
        # pass none value in most_recent_rows
        if i == None:
            pass
        else:
            # make pollster be the key in the outside dictionary 
            key = i["Pollster"]
            # make state be the key of inner dictionary 
            value = i["State"]  
            # calculate edge within given state 
            edge = state_edges([i])[value]
            # input edge to the value of inner dictionary
            prediction[key][value] = edge
        
    return prediction

"""
Report: In this question, I save the pollsters and states of the given data in two
lists by using the function "unique_column_values", then find the most recent date
of selected pollster and state and put them in another list, at the end use defaultdict 
module to create a nested dictionary. 
"""   
     
################################################################################
# Problem 4: Pollster errors
################################################################################

def average_error(state_edges_predicted, state_edges_actual):
    """
    Given predicted state edges and actual state edges, returns
    the average error of the prediction.
    """
    # define value as a number to iterate the calculation
    values = 0 
    for state in state_edges_predicted:
        # the given formula in homework
        values += abs(state_edges_predicted[state] - state_edges_actual[state])
        # use lenth of "predicted" for the denominator 
        average_error = values/len(state_edges_predicted)
        # to ensure dividing by the states in which a pollster made a prediction
    
    return average_error

def pollster_errors(pollster_predictions, state_edges_actual):
    """
    Given pollster predictions and actual state edges, retuns pollster errors.
    """
    p_errors = {}
    # use one for loop to ensure
    for pollster in pollster_predictions:
        # the pollster is sanme in the prediction and actual data 
        value = average_error(pollster_predictions[pollster], state_edges_actual)
        # use pollster as the key and average_error as the value
        p_errors.update({pollster:value})
        
    return p_errors
"""
Report: Because the prediction and actual data share the same index, we can calculate
average_error by using one for loop. Then make pollster be the key and pollster_errors 
be the value in the result dictionary, in which value can be calculated in average_error.
"""

################################################################################
# Problem 5: Pivot a nested dictionary
################################################################################

def pivot_nested_dict(nested_dict):
    """
    Pivots a nested dictionary, producing a different nested dictionary
    containing the same values.
    The input is a dictionary d1 that maps from keys k1 to dictionaries d2,
    where d2 maps from keys k2 to values v.
    The output is a dictionary d3 that maps from keys k2 to dictionaries d4,
    where d4 maps from keys k1 to values v.
    For example:
      input = { "a" : { "x": 1, "y": 2 },
                "b" : { "x": 3, "z": 4 } }
      output = {'y': {'a': 2},
                'x': {'a': 1, 'b': 3},
                'z': {'b': 4} }
    """
    
    new_dict = {}
    # divide keys and values in input dictionary
    for k1, v1 in nested_dict.items():
        for k2, v2 in v1.items():
            # rearrange keya and values
            new_dict.setdefault(k2,{})[k1]=v2
    
    return new_dict
"""
Report: This is a question about how to rearrange a dictionary, so first we need 
to divide all keys and values in the nested_dict, and then give the new set of 
keys and values according to the output.
"""

################################################################################
# Problem 6: Average the edges in a single state
################################################################################

def average_error_to_weight(error):
    """
    Given the average error of a pollster, returns that pollster's weight.
    The error must be a positive number.
    """
    return error ** (-2)

# The default average error of a pollster who did no polling in the
# previous election.
DEFAULT_AVERAGE_ERROR = 5.0

def pollster_to_weight(pollster, pollster_errors):
    """"
    Given a pollster and a pollster errors, return the given pollster's weight.
    """
    if pollster not in pollster_errors:
        weight = average_error_to_weight(DEFAULT_AVERAGE_ERROR)
    else:
        weight = average_error_to_weight(pollster_errors[pollster])
    return weight

def weighted_average(items, weights):
    """
    Returns the weighted average of a list of items.
    
    Arguments:
    items is a list of numbers.
    weights is a list of numbers, whose sum is nonzero.
    
    Each weight in weights corresponds to the item in items at the same index.
    items and weights must be the same length.
    """
    assert len(items) > 0
    assert len(items) == len(weights)
    
    total = 0
    # make total equal to the weighted value
    for i in range(0, len(items)):
        total += items[i]*weights[i]
    # calculate weighted_average according to the given formula
    w_average = total/sum(weights)
    return w_average

weighted_average([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])

def average_edge(pollster_edges, pollster_errors):
    """
    Given pollster edges and pollster errors, returns the average of these edges
    weighted by their respective pollster errors.
    """   
    weight = []
    # create a list to save values of pollster_edges
    items = []
    # make sure the p1 = p2
    for p1 in pollster_edges:
        items.append(pollster_edges[p1])
    # so that the lenth of items and weights are the same
    for p2 in pollster_edges:
         weight.append(pollster_to_weight(p2, pollster_errors))
    # calculate average_edge by using weighted_average  
    avg_edge = weighted_average(items, weight)
    
    # This a way to return integer which is same as its round value 
    if round(avg_edge, 10) == int(avg_edge):
        avg_edge = int(avg_edge)
        
    return avg_edge

"""
Report: This question is mainly about math and calculation, but we need to be careful
to make the lenth of items and weights be the same, to achieve it, I use same "pollster"
in pollster_edges as the index of edges and errors are the same.
"""

################################################################################
# Problem 7: Predict the 2012 election
################################################################################

def predict_state_edges(pollster_predictions, pollster_errors):
    """
    Given pollster predictions from a current election and pollster errors from
    a past election, returns the predicted state edges of the current election.
    """
    # change the key position of the pollster and state in the given nested dictionary
    pollster_dict = pivot_nested_dict(pollster_predictions)
    # create a new dictionary to save predict_results of each state
    predict_results = {}
    
    for state in pollster_dict:
        # do average edge of each state as the value of new dictionary
        value = average_edge(pollster_dict[state],pollster_errors)
        predict_results.update({state: value})
    
    return predict_results

"""
Report: This question is also to rearrange a nested dictionary, so we can use 
the former function pivot_nested_dict to get the eage of each state that all 
pollsters predicted. Then use average_edge to calculate predicted edges and put 
it as value.
"""

################################################################################
# Electoral College, Main Function, etc.
################################################################################

def electoral_college_outcome(ec_rows, state_edges):
    """
    Given electoral college rows and state edges, returns the outcome of
    the Electoral College, as a map from "Dem" or "Rep" to a number of
    electoral votes won.  If a state has an edge of exactly 0.0, its votes
    are evenly divided between both parties.
    """
    ec_votes = {}               # maps from state to number of electoral votes
    for row in ec_rows:
        ec_votes[row["State"]] = float(row["Electors"])

    outcome = {"Dem": 0, "Rep": 0}
    for state in state_edges:
        votes = ec_votes[state]
        if state_edges[state] > 0:
            outcome["Dem"] += votes
        elif state_edges[state] < 0:
            outcome["Rep"] += votes
        else:
            outcome["Dem"] += votes/2.0
            outcome["Rep"] += votes/2.0
    return outcome

def print_dict(dictionary):
    """
    Given a dictionary, prints its contents in sorted order by key.
    Rounds float values to 8 decimal places.
    """
    for key in sorted(dictionary.keys()):
        value = dictionary[key]
        if type(value) == float:
            value = round(value, 8)
        print(key, value)


def main():
    """
    Main function, which is executed when election.py is run as a Python script.
    """
    # Read state edges from the 2008 election
    edges_2008 = state_edges(read_csv("data/2008-results.csv"))
    
    # Read pollster predictions from the 2008 and 2012 election
    polls_2008 = pollster_predictions(read_csv("data/2008-polls.csv"))
    polls_2012 = pollster_predictions(read_csv("data/2012-polls.csv"))
    
    # Compute pollster errors for the 2008 election
    error_2008 = pollster_errors(polls_2008, edges_2008)
    
    # Predict the 2012 state edges
    prediction_2012 = predict_state_edges(polls_2012, error_2008)
    
    # Obtain the 2012 Electoral College outcome
    ec_2012 = electoral_college_outcome(read_csv("data/2012-electoral-college.csv"),
                                        prediction_2012)
    
    print ("Predicted 2012 election results:")
    print_dict(prediction_2012)
    print
    
    print ("Predicted 2012 Electoral College outcome:")
    print_dict(ec_2012)
    print    


# If this file, election.py, is run as a Python script (such as by typing
# "python election.py" at the command shell), then run the main() function.
if __name__ == "__main__":
    main()
