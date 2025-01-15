//in progress creating template for the project (movie dataset reading, masking, KNN for K=1)
import java.io.*;

public class Tests {
    // Use we use 'static' for all methods to keep things simple, so we can call those methods main

    static void Assert (boolean res) // We use this to test our results - don't delete or modify!
    {
        if(!res)	{
            System.out.print("Something went wrong.");
            System.exit(0);
        }
    }
    static int NumberOfFeatures = 13; // Moved NumberOfFeatures to the top
    static double highestIMDB = 0;
    static double highestRating = 0; // Only to be used when testing against IMDB
    static double highestRuntime = 0;
    static double highestBoxOffice = 0;
    static double highestBudget = 0;


    static int posCount = 0, negCount = 0; // Counts of positive (Like) and negative (Dislike) datapoints.
    static double [] FeatureCountsPos = new double [NumberOfFeatures];  // to count #{x AND pos}, where x can be cough, fever or sneezing. pos = having a flu
    static double [] FeatureCountsNeg = new double [NumberOfFeatures]; // to count #{x AND ~pos}

    static class NaiveBayesModel {
        public NaiveBayesModel() {
        }

        double estimate(double[] features) {
            double s = Math.log((double) posCount / (double) negCount);
            for (int i = 0; i < NumberOfFeatures; i++) {
                if (features[i] > 0) {
                    double p_feature_cond_pos = FeatureCountsPos[i] / posCount; // P(x|C) = #{x AND pos} / #{pos}
                    if (p_feature_cond_pos == 0)
                        p_feature_cond_pos = .01; // We make each estimated probability to be at least 0.01 to avoid division by 0 later.
                    // This is called "smoothing."
                    double p_feature_cond_neg = FeatureCountsNeg[i] / negCount;
                    if (p_feature_cond_neg == 0)
                        p_feature_cond_neg = .01;

                    double feature_strength = p_feature_cond_pos / p_feature_cond_neg; //

                    s = s + Math.log(feature_strength);
                }
            }
            return 1 / (1 + Math.exp(-s)); // Convert back from log O(C|X) to P(C|X)
        }


        public void update(double[][] features, int label[]) {
            for (int i = 0; i < features[i].length; i++) {
                for (int j = 0; j < NumberOfFeatures; j++) {
                    if (label[i] > 0) {
                        if (features[i][j] > 0) {
                            FeatureCountsPos[i]++;
                        }
                    } else {
                        if (features[i][j] > 0) {
                            FeatureCountsNeg[i]++;
                        }
                    }

                }
                // Update the counts of #{pos} and #{~pos}:
                if (label[i] > 0) {
                    posCount++;
                } else {
                    negCount++;
                }
            }
        }
    }

        // Dot Function
        static double dot(double[] U, double[] V) { // dot product of two vectors
            // add your code
            Assert(U.length == V.length);
            double result = 0;
            for (int i = 0; i < U.length; i++)
                result += U[i] * V[i];
            return result;
        }

        // Cosine Similarity
        static double cosineSimilarity(double[] U, double[] V) {
            double dotProduct = 0.0;
            double normA = 0.0;
            double normB = 0.0;
            for (int i = 0; i < U.length; i++) {
                dotProduct += U[i] * V[i];
                normA += Math.pow(U[i], 2);
                normB += Math.pow(V[i], 2);
            }
            return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
        }

        static double[] toFeatureVector(double id, String genre, double runtime, double year, double imdb, double rt, double budget, double boxOffice, int model) {

            double[] feature = new double[NumberOfFeatures];
            System.out.println("Model: " + model);
            switch (genre) { // We also use represent each movie genre as an integer number:
                case "Action":
                    feature[0] = 1;
                    break;
                case "Fantasy":
                    feature[1] = 1;
                    break;
                case "Romance":
                    feature[2] = 1;
                    break;
                case "Sci-Fi":
                    feature[3] = 1;
                    break;
                case "Adventure":
                    feature[4] = 1;
                    break;
                case "Horror":
                    feature[5] = 1;
                    break;
                case "Comedy":
                    feature[6] = 1;
                    break;
                case "Thriller":
                    feature[7] = 1;
                    break;
                default:
                    Assert(false);

            }

            // This will add IMDB rating as a feature to be tested
            if (model > 1) {
                for (int i = 0; i < 101; i++) {
                    if (highestIMDB < imdb) {
                        highestIMDB = imdb;
                    }
                }
                highestIMDB -= 7.2844; // Normalised highestIMDB
                feature[8] = highestIMDB;

                // This will add runtime as a feature to be tested
                if (model > 2) {
                    for (int i = 0; i < 101; i++) {
                        if (highestRuntime < runtime) {
                            highestRuntime = runtime;
                        }
                    }
                    // highestRuntime -= 109.6; // Normalised highestRuntime
                    feature[9] = highestRuntime;

                    // This will add budget as a feature to be tested
                    if (model > 3) {
                        for (int i = 0; i < 101; i++) {
                            if (highestBudget < budget) {
                                highestBudget = budget;
                            }
                        }
                        // highestBudget -= 97.91; // Normalised highestBudget
                        feature[10] = highestBudget;

                        // This will add Box Office Revenue as a feature to be tested
                        if (model > 4) {
                            for (int i = 0; i < 101; i++) {
                                if (highestBoxOffice < boxOffice) {
                                    highestBoxOffice = boxOffice;
                                }
                            }
                            // highestBoxOffice -= 149.55; // Normalised highestBoxOffice
                            feature[11] = highestBoxOffice;

                            // This will add year as a feature to be tested
                            if (model > 5) {
                                if (year >= 2022) { // Normalised year
                                    feature[12] = year;
                                }
                            }
                        }
                    }
                }
            }

            return feature;
        }


        // We are using the dot product to determine similarity:
        static double similarity(double[] u, double[] v) {
            return cosineSimilarity(u, v);
        }

        // Updated KNN classifier supporting any K - Please message Discord if not implemented correctly
        static int knnClassify(double[][] trainingData, int[] trainingLabels, double[] testFeature, int K) {
            int n = trainingData.length;           // Assign length of data to n
            double[] similarities = new double[n]; // Similarity score of all samples
            int[] indices = new int[n];            // Corresponding indices - used for sorting

            for (int i = 0; i < n; i++) {
                similarities[i] = similarity(testFeature, trainingData[i]); // Similarity of all samples
                indices[i] = i;
            }

            // Sorting of similarities (Selection sort) - high to low
            for (int i = 0; i < n - 1; i++) {
                for (int j = i + 1; j < n; j++) {
                    if (similarities[i] < similarities[j]) {

                        // If similarity is greater, swap placement
                        double temporarySimilarity = similarities[i];
                        similarities[i] = similarities[j];
                        similarities[j] = temporarySimilarity;

                        // Swap indices
                        int temporaryIndex = indices[i];
                        indices[i] = indices[j];
                        indices[j] = temporaryIndex;
                    }
                }
            }

            // Store neighbouring labels for K most similar
            int[] neighbouringLabels = new int[K];
            for (int i = 0; i < K; i++) {
                neighbouringLabels[i] = trainingLabels[indices[i]]; // Assigns label of i nearest to neighbour
            }

            // Majority voting
            int[] labelVotes = new int[2];          // Counts 0s and 1s and stores in labelVotes
            for (int label : neighbouringLabels) {  // For each label in neighbouringLabels
                labelVotes[label]++;                // Increment votes if 0 or 1
            }

            // Return label based on majority of k closest neighbours
            if (labelVotes[0] > labelVotes[1]) {
                return 0;
            } else {
                return 1;   // Returns 1 if the labels are equal - is there a way to fix this??
            }
        }

        static void loadData(String filePath, double[][] dataFeatures, int[] dataLabels, int model) throws IOException {
            try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
                String line;
                int idx = 0;
                br.readLine(); // skip header line
                while ((line = br.readLine()) != null) {
                    String[] values = line.split(",");
                    // Assuming csv format: MovieID,Title,Genre,Runtime,Year,Lead Actor,Director,IMDB,RT(%),Budget,Box Office Revenue (in million $),Like it
                    double id = Double.parseDouble(values[0]);
                    String genre = values[2];
                    double runtime = Double.parseDouble(values[10]);
                    double year = Double.parseDouble(values[3]);
                    double imdb = Double.parseDouble(values[7]);
                    double rt = Double.parseDouble(values[6]);
                    double budget = Double.parseDouble(values[9]);
                    double boxOffice = Double.parseDouble(values[8]);

                    dataFeatures[idx] = toFeatureVector(id, genre, runtime, year, imdb, rt, budget, boxOffice, model);
                    dataLabels[idx] = Integer.parseInt(values[11]); // Assuming the label is the last column and is numeric
                    idx++;
                }
            }
        }


        public static void main(String[] args) {

            double[][] trainingData = new double[100][NumberOfFeatures];
            int[] trainingLabels = new int[100];
            double[][] testingData = new double[100][NumberOfFeatures];
            int[] testingLabels = new int[100];
            // Compute accuracy on the testing set
            int correctPredictions = 0;
            int modelChoice = 2; // Change to 1 for KNN, 2 for Naive Bayes

            try {
                // You may need to change the path:
                loadData("/Users/jackallones/Downloads/training-set.csv", trainingData, trainingLabels, 6);
                loadData("/Users/jackallones/Downloads/testing-set.csv", testingData, testingLabels, 6);
            } catch (IOException e) {
                System.out.println("Error reading data files: " + e.getMessage());
                return;
            }

            // Choose between either KNN or Naive Bayes Model
            if (modelChoice == 1) {
                System.out.println("KNN Classifier");
                for (int i = 0; i < trainingData.length; i++) {
                    int predictedLabel = knnClassify(trainingData, trainingLabels, testingData[i], 1);
                    if (predictedLabel == testingLabels[i]) {
                        correctPredictions++;
                    }
                }
                // Test Accuracy
                double accuracy = (double) correctPredictions / testingData.length * 100;
                System.out.printf("A: %.2f%%\n", accuracy);
            } else if (modelChoice == 2) {
                NaiveBayesModel naiveBayesModel = new NaiveBayesModel(); // Create a new naive baiyes model


                // Initialising feature counts to 0s:
                for (int i = 0; i < NumberOfFeatures; i++) {
                    FeatureCountsPos[i] = 0;
                    FeatureCountsNeg[i] = 0;
                }
                // Update Tables
                naiveBayesModel.update(trainingData, trainingLabels);

                // Test Accuracy
                for (int k = 0; k < testingData.length; k++) {
                    int prediction;
                    double probability = naiveBayesModel.estimate(testingData[k]);
                    if (probability >= .5) // We apply .5 probability threshold to predict the class
                        prediction = 1;
                    else
                        prediction = 0;

                    if (prediction == testingLabels[k])
                        correctPredictions++; // Replace '??' with the line of code that changes 'number_correct_predictions' accordingly
                }
                System.out.println("Naive Bayes Model");
                double accuracy = (double) correctPredictions / testingData.length * 100;
                System.out.printf("A: %.2f%%\n", accuracy);
            }
        }
    }
