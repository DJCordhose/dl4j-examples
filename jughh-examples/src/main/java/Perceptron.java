
public class Perceptron {
    double w0 = 3, w1 = -4, w2 = 2;
    double forward(double x1, double x2) {
        double sum = w0 + w1 * x1 + w2 * x2;
        return activation(sum);
    }

    double activation(double z) {
        // in this case a sigmoid function (alt.: tanh, linear, relu)
        return 1 / (1 + Math.exp(z * -1));
    }

    public static void main(String[] args) {
        final Perceptron perceptron = new Perceptron();
        final double forwarded = perceptron.forward(1, 2);
        System.out.println(forwarded);
    }
}
