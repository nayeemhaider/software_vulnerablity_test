/* This program classify the attacks on software development using
 * a machine learning algorithm. The approach is to classifying the
 * vulnerable data set and record the accuracy, precision, recall and F-measure (F1 score)
 * of the evaluated model. WEKA API for java has been used in this program.
 */

package codingtest;


// Importing here all the libraries needed
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
//import java.math.RoundingMode;
import java.text.DecimalFormat;

public class codingtest {

	private static final DecimalFormat df = new DecimalFormat("0.00");	// Declaring a constant value to round up the decimal point within this class //
	
	public static void main(String[] args) {
		
		/* try and catch statement has been used in this program to make sure the program runs smoothly and 
		throws an exception if any error occurs */
		try {			
			
			// Taking the vulnerable data set as input in ARFF format
			String dataFileName = "C:\\Users\\User\\Documents\\Java Programs\\codingtest\\vulnerable-methods-dataset.arff";	// use appropriate path
			
			Instances data = (new DataSource(dataFileName)).getDataSet();	// Defining the instances of the data set //
			data.setClassIndex(data.numAttributes() - 1);	// Defining the target class // 
			
			data.randomize(new java.util.Random());	// randomize instance order before splitting data set //
	        
			// Splitting the data set into training set and testing set //
			Instances trainData = data.trainCV(70, 10);
			Instances testData = data.testCV(30, 10);
			
			// Creating a new J48 classifier //
			J48 j48Classifier = new J48();
			j48Classifier.buildClassifier(trainData);
			
			// Training the j48 classifier using training set
			System.out.println("\n\nModel Evaluation of Training Data");
			Evaluation evaluation = new Evaluation(trainData);
			evaluation.evaluateModel(j48Classifier, trainData);
			
			// Printing the model evaluation and accuracy of training data //
			System.out.println(evaluation.toSummaryString("\nResults",false));
			System.out.println("Accuracy:  " + df.format(100*(evaluation.pctCorrect()/100)) + " %");
			
			// Testing the j48 classifier using test data
			System.out.println("\n\nModel Evaluation of Testing Data");
			evaluation.evaluateModel(j48Classifier, testData);
			
			// Printing out the model evaluation and matrices (Accuracy, Precision, Recall, F1-score) on test data //
			System.out.println(evaluation.toSummaryString("\nResults",false));
			System.out.println("Accuracy:  " + df.format(100*(evaluation.pctCorrect()/100)) +" %");
			System.out.println("Precision: " + df.format(100*(evaluation.precision(1))) + " %");
			System.out.println("Recall:    " + df.format(100*(evaluation.recall(1))) + " %");
			System.out.println("F-measure: " + df.format(100*(evaluation.fMeasure(1))) + " %");
			
		}
		
		// this piece of code print "Error Occured!!!!" message if there is any error in the process //
		catch (Exception e) {
            		System.out.println("Error Occured!!!! \n" + e.getMessage());
        }
	}
}