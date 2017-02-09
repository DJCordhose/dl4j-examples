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
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

// derived from
// https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/mnist/MLPMnistTwoLayerExample.java
public class FeedForward2Hidden {
    private static final Logger log = LoggerFactory.getLogger(FeedForward2Hidden.class);

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
                .activation(Activation.LEAKYRELU)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.02)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(28 * 28).nOut(500).build())
                .layer(1, new DenseLayer.Builder().nIn(500).nOut(100).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX).nIn(100).nOut(10).build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork mlpNet = new MultiLayerNetwork(conf);
        mlpNet.init();

        Util.train(mlpNet, mnistTrain, mnistTest, "ff2");
        Util.printStats(mnistTest, mlpNet);
    }
}
