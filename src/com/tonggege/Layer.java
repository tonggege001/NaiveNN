package com.tonggege;

public class Layer {
    //该层神经元
    double [] inputdata;
    Neuron [] neurons;
    Layer nextLayer;
    Layer priorLayer;
    boolean is_OutputLayer;//判断是否是输出层
    int index;//层的编号
    /**
     * 构造函数
     */
    public Layer(int index){
        this.nextLayer = null;
        this.priorLayer = null;
        this.neurons = null;
        this.inputdata = null;
        this.is_OutputLayer = false;
        this.index = index;
    }
    double [] generateOutput(){
        double[] output = new double[neurons.length];
        for(int i = 0;i<neurons.length;i++){
            output[i] = neurons[i].output;
        }
        if(this.is_OutputLayer) return softmax(output);
        else return output;
    }

    double [] softmax(double[] out){
        if(this.is_OutputLayer){
            double total = 0.0;
            for(int i = 0;i<neurons.length;i++){
                out[i] = Math.exp(out[i]);
                total = total+out[i];
            }
            for(int i = 0;i<neurons.length;i++){
                out[i] = out[i]/total;
            }
            return out;
        }
        else
            return null;
    }
}
