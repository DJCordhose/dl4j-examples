import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Predictor {
    private static Logger log = LoggerFactory.getLogger(Predictor.class);

    public static void main(String[] args) throws IOException, InterruptedException {
        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();

        final List<String> modelNames;
        if (args.length == 0 ) {
            modelNames = new ArrayList<>();
            modelNames.add("conv");
            modelNames.add("ff1");
            modelNames.add("ff2");
        } else {
            modelNames = Arrays.asList(args);
        }
        final List<MultiLayerNetwork> models = new ArrayList<>();
        modelNames.forEach(modelName -> {
            try {
                MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File(classLoader.getResource("models/" + modelName + ".zip").getFile()));
                models.add(model);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });

        RecordReader recordReader = new ImageRecordReader(28, 28, 1);
        final File rootDir = new File(classLoader.getResource("./samples").getFile());
        recordReader.initialize(new FileSplit(rootDir));

        DataSetIterator dataSets = new RecordReaderDataSetIterator(recordReader, 1);

        int index = 0;
        while (dataSets.hasNext()) {
            log.info(rootDir.list()[index++]);
            final INDArray indArray = convertToBinary(dataSets.next().getFeatureMatrix());

            for (int i=0; i < models.size(); i++) {
                MultiLayerNetwork model = models.get(i);
                String name = modelNames.get(i);
                INDArray output = model.output(indArray);
                log.info(output + " <- " + name);
            }

            log.info("   ^     ^     ^     ^     ^     ^     ^     ^     ^     ^");
            log.info("   0     1     2     3     4     5     6     7     8     9");
        }
    }

    // convert the data so that it has the 0/1 form of the training data
    private static INDArray convertToBinary(final INDArray indArray) {
        final int threshold = 127;
        final float[] f = indArray.data().asFloat();
        final float[] t = new float[f.length];
        for (int i = 0; i < f.length; i++) {
            if (f[i] < threshold) {
                t[i] = 1;

            } else {
                t[i] = 0;
            }
        }
        return Nd4j.create(t);
    }
}
