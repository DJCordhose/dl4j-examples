import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

import static org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

// derived from
// https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/mnist/MLPMnistSingleLayerExample.java
public class FeedForward1Hidden {
    private static final Logger log = LoggerFactory.getLogger(FeedForward1Hidden.class);

    public static void main(String[] args) throws IOException {
        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(Util.batchSize, true, Util.SEED);
        DataSetIterator mnistTest = new MnistDataSetIterator(Util.batchSize, false, Util.SEED);

        //----------------------------------
        //Create network configuration and conduct network training
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(Util.SEED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(28 * 28) // Number of input datapoints.
                        .nOut(1000) // Number of output datapoints.
                        .activation(Activation.RELU) // Activation function.
                        .weightInit(WeightInit.XAVIER) // Weight initialization.
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(1000)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork mlpNet = new MultiLayerNetwork(conf);
        mlpNet.init();

        Util.train(mlpNet, mnistTrain, mnistTest, "ff1");
        Util.printStats(mnistTest, mlpNet);
    }

}
