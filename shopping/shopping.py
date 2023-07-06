import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data('shopping.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)
    r_count=0
    for x in range(4932):
        if y_test[x]==predictions[x]:
            r_count+=1
    print (r_count)
    w_count=0
    for x in range(4932):
        if y_test[x]!=predictions[x]:
            w_count+=1
    print (w_count)
    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        data = []
        Evidence=[]

        for row in reader:
            for x in range(0,5,2):
                Evidence.append(int(row[x]))
            
            for x in range(1,6,2):
                Evidence.append(float(row[x]))
            
            for x in row[6:10]:
                Evidence.append(float(x))
                
            if row[10]=='Jan':
                Evidence.append(0)
            elif row[10]=='Feb':
                Evidence.append(1)
            elif row[10]=='Mar':
                Evidence.append(2)
            elif row[10]=='Apr':
                Evidence.append(3)
            elif row[10]=='May':
                Evidence.append(4)
            elif row[10]=='June':
                Evidence.append(5)
            elif row[10]=='Jul':
                Evidence.append(6)
            elif row[10]=='Aug':
                Evidence.append(7)
            elif row[10]=='Sep':
                Evidence.append(8)
            elif row[10]=='Oct':
                Evidence.append(9)
            elif row[10]=='Nov':
                Evidence.append(10)
            elif row[10]=='Dec':
                Evidence.append(11)

            for x in row[11:15]:
                Evidence.append(int(x))
            
            if row[15]=='Returning_Visitor':
                Evidence.append(1)
            else:
                Evidence.append(0)

            if row[16]=='TRUE':
                Evidence.append(1)
            else:
                Evidence.append(0)               

            data.append({
                "evidence":Evidence,
                "label": [1] if row[17] == 'TRUE' else [0]
            })
            Evidence=[]


        Evidence = [row["evidence"] for row in data]
        Labels = [row["label"] for row in data]
        return(Evidence,Labels)
            


        
        



def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model=KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence,labels)
    return(model)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    false_count=0
    spec_count=0
    true_count=0
    sens_count=0
    pre_count=0
    for x in range(4932):
        

        if labels[x]==[1]:
            true_count+=1
            if predictions[x]==1:
                sens_count+=1
        
        if labels[x]==[0]:
            false_count+=1
            if predictions[x]==0:
                spec_count+=1

    print(sens_count,true_count,spec_count,false_count)

    return(((sens_count/true_count),(spec_count/false_count)))


if __name__ == "__main__":
    main()
