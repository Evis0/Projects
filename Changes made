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
            case "Action":    feature[0] = 1; break;
            case "Fantasy":   feature[1] = 1; break;
            case "Romance":   feature[2] = 1; break;
            case "Sci-Fi":    feature[3] = 1; break;
            case "Adventure": feature[4] = 1; break;
            case "Horror":    feature[5] = 1; break;
            case "Comedy":    feature[6] = 1; break;
            case "Thriller":  feature[7] = 1; break;
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



    // KNN Classifier (K = 1 in this case)
    static int knnClassify(double[][] trainingData, int[] trainingLabels, double[] testFeature) {
        int bestMatch = -1;
        double bestSimilarity = -Double.MAX_VALUE; // Start with worst similarity

        for (int i = 0; i < trainingData.length; i++) {
            double currentSimilarity = similarity(testFeature, trainingData[i]);
            if (currentSimilarity > bestSimilarity) {
                bestSimilarity = currentSimilarity;
                bestMatch = i;
            }
        }

        return trainingLabels[bestMatch]; // Return the label of the closest match
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
        for (int i = 0; i < testingData.length; i++) {
            int predictedLabel = knnClassify(trainingData, trainingLabels, testingData[i]);
            if (predictedLabel == testingLabels[i]) {
                correctPredictions++;
            }
        }

        // Calculate and print the accuracy
        double accuracy = (double) correctPredictions / testingData.length * 100;
        System.out.printf("Accuracy: %.2f%%\n", accuracy);
    }
}
