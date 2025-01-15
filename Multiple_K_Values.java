// This file implements K > 1, needs checked to see if works correctly

import java.io.*;

public class Test_Changes {


    // Assert method for testing results
    static void Assert(boolean res) {
        if (!res) {
            System.out.print("Something went wrong.");
            System.exit(0);
        }
    }

    static int NumberOfFeatures = 8;

    // Function to convert movie attributes to a feature vector
    static double[] toFeatureVector(double id, String genre, double runtime, double year, double imdb, double rt, double budget, double boxOffice) {
        double[] feature = new double[NumberOfFeatures];

        // Represent genre as an integer (one-hot encoding could be used for better scalability)
        switch (genre) {
            case "Action": feature[1] = 1; break;
            case "Fantasy": feature[1] = 1; break;
            case "Romance": feature[1] = 1; break;
            case "Sci-Fi": feature[1] = 1; break;
            case "Adventure": feature[1] = 1; break;
            case "Horror": feature[1] = 1; break;
            case "Comedy": feature[1] = 1; break;
            case "Thriller": feature[1] = 1; break;
            default: Assert(false); // Error if the genre is unknown
        }

        // Other numerical features
        feature[2] = runtime;
        feature[3] = year;
        feature[4] = imdb;
        feature[5] = rt;
        feature[6] = budget;
        feature[7] = boxOffice;

        return feature;
    }

    // Cosine similarity function for similarity measurement
    static double similarity(double[] u, double[] v) {
        return cosineSimilarity(u, v);
    }

    private static double cosineSimilarity(double[] u, double[] v) {
        double dotProduct = 0.0;
        double normU = 0.0;
        double normV = 0.0;

        for (int i = 0; i < u.length; i++) {
            dotProduct += u[i] * v[i];
            normU += u[i] * u[i];
            normV += v[i] * v[i];
        }

        return dotProduct / (Math.sqrt(normU) * Math.sqrt(normV));
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
        }
        else {
            return 1;   // Returns 1 if the labels are equal - is there a way to fix this??
        }
    }

    // Load data from CSV file
    static void loadData(String filePath, double[][] dataFeatures, int[] dataLabels) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            int idx = 0;
            br.readLine(); // Skip header line
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                try {
                    double id = Double.parseDouble(values[0]);
                    String genre = values[2];
                    double runtime = Double.parseDouble(values[10]);
                    double year = Double.parseDouble(values[3]);
                    double imdb = Double.parseDouble(values[7]);
                    double rt = Double.parseDouble(values[6]);
                    double budget = Double.parseDouble(values[9]);
                    double boxOffice = Double.parseDouble(values[8]);

                    dataFeatures[idx] = toFeatureVector(id, genre, runtime, year, imdb, rt, budget, boxOffice);
                    dataLabels[idx] = Integer.parseInt(values[11]); // Assuming the label is numeric
                } catch (NumberFormatException | ArrayIndexOutOfBoundsException e) {
                    System.out.println("Skipping line due to data issue: " + line);
                    continue; // Skip this line if there's an error
                }
                idx++;
            }
        }
    }

    // Main method for executing the program
    public static void main(String[] args) {

        // Arrays to hold the features and labels for training and testing sets
        double[][] trainingData = new double[100][];
        int[] trainingLabels = new int[100];
        double[][] testingData = new double[100][];
        int[] testingLabels = new int[100];

        try {
            // Load the data from CSV files (paths need to be modified based on your setup)
            loadData("C:\\Users\\short\\OneDrive\\Uni Work\\2nd Year\\CS259\\Final\\src\\training-set.csv", trainingData, trainingLabels);
            loadData("C:\\Users\\short\\OneDrive\\Uni Work\\2nd Year\\CS259\\Final\\src\\testing-set.csv", testingData, testingLabels);
        } catch (IOException e) {
            System.out.println("Error reading data files: " + e.getMessage());
            return;
        }

        // Compute accuracy on the testing set
        int correctPredictions = 0;

        // Classify each test sample and check if it's correct
        int K = 3;  // // Value of K to be used - Change where necessary
        for (int i = 0; i < testingData.length; i++) {
            int predictedLabel = knnClassify(trainingData, trainingLabels, testingData[i], K);
            if (predictedLabel == testingLabels[i]) {
                correctPredictions++;
            }
        }

        // Calculate and print the accuracy
        double accuracy = (double) correctPredictions / testingData.length * 100;
        System.out.printf("Accuracy: %.2f%%\n", accuracy);
    }
}
