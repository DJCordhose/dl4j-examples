import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

import static org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class Util {
    static final String MODEL_PATH = "src/main/resources/models/";
    static final int SEED = 12345;

    private static final Logger log = LoggerFactory.getLogger(Util.class);
    static int outputNum = 10; // Number of possible outcomes (e.g. labels 0 through 9).
    static int batchSize = 128; // How many examples to fetch with each step.
    static int numEpochs = 10; // An epoch is a complete pass through a given dataset.

    public static void printStats(DataSetIterator mnistTest, MultiLayerNetwork net) {
        //Perform evaluation (distributed)
        Evaluation evaluation = net.evaluate(mnistTest);
        log.info("***** Evaluation *****");
        log.info(evaluation.stats());

        log.info("***** Example Complete *****");
    }

    public static void train(MultiLayerNetwork model,
                             DataSetIterator train,
                             DataSetIterator test,
                             String modelName) throws IOException {
        log.info("Train model....");
        model.setListeners(new ScoreIterationListener(1));
        for (int i = 0; i < numEpochs; i++) {
            model.fit(train);
            log.info("*** Completed epoch {} ***", i);

            log.info("Evaluate model....");
            Evaluation eval = new Evaluation(outputNum);
            while (test.hasNext()) {
                DataSet ds = test.next();
                INDArray output = model.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);
            }

            saveModel(model, modelName);
            log.info(eval.stats());
            test.reset();
        }
    }

    private static void saveModel(MultiLayerNetwork model, String modelName) throws IOException {
        //Where to save the network. Note: the file is in .zip format - can be opened externally
        File locationToSave = new File(MODEL_PATH + modelName + ".zip");
        //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc.
        //Save this if you want to train your network more in the future
        boolean saveUpdater = true;
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);
    }
}
