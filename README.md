# Review-Sorter
A NLP program that uses K nearest neighbors to differentiate if a review of a product is positive or not

This project uses Numpy, Pandas, and NLP.

The program inputs a reviews and outputs +1/-1 determining if the review given is positive or negative.
The program determines whether a review is positive or negative by vectorizing words based off its influence of making a review negative or positive and then using the KNN model to compare it to the words in the review.
The program will then determine if the review is positive or negative based off KNN clustering.

The data was trained by inputing reviews with the +1/-1 notation, showing whether the review is postive. 
The test to be inputted is a list of strings of reviews.
