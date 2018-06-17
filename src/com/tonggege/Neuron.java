package com.tonggege;

import java.util.Random;

public class Neuron {
    Layer layer;
    double LearningRT;//学习率
    int index;//该神经元在该层的第几个,下标从0开始
    double bias;
    double w[];//weight from last layer，输入数据时直接在数据的最后一项设置为1
    double z;
    double output;
    double DcIDz;//交叉熵对Z的偏微分

    public  Neuron(){
        this.layer = null;
    }
    /**
     * 计算神经元的z = w*x+b值
     * @param input 上一层lay产生的输出
     * @return z值
     */
    double figureZ(double []input){
        if(input.length!=w.length) return 0.0;
        double z = 0.0;
        for(int i = 0;i<input.length;i++){
            z = z+w[i]*input[i];
        }
        this.z = z + bias;
        return z;
    }

    /**
     * 计算该神经元的输出，在前向传播的时候调用
     */
    void figureOutput(){
        if(!layer.is_OutputLayer){
            if(layer.priorLayer!=null)
                this.output = 1/(Math.exp(figureZ(layer.priorLayer.generateOutput()))+1.0);
            else
                this.output = 1/(Math.exp(figureZ(layer.inputdata))+1.0);
        }
        else
            this.output = figureZ(layer.priorLayer.generateOutput());
    }
    /**
     * 计算DC/DZ的值，注意该函数只能够从后向前调用，也就是先计算LayerN，LayerN-1...Layer1
     */
    void figureDcIDz(){
        //如果是输出层，则直接可以计算Dc/Dz
        if(layer.is_OutputLayer){
            double c = 0.0;
            double e = 0.0;
            for(int i = 0;i<layer.neurons.length;i++){
                c = c + Math.exp(layer.neurons[i].output);
            }
            e = Math.exp(this.output);
            this.DcIDz = -(c-e)/c;
        }
        //如果不是输出层，则需要从后向前递归调用（理论上是递归，实际上是直接调用之前计算好的
        //该手段类似于动态规划，保存好已经计算后的结果
        else{
            double dcidz = 0.0;
            for(int i = 0;i<layer.nextLayer.neurons.length;i++){
                dcidz += layer.nextLayer.neurons[i].w[this.index] * layer.nextLayer.neurons[i].DcIDz;
            }
            this.DcIDz = this.z*dcidz;
        }
    }
    void updatePara(){
        //如果是第一层的话,直接使用inoutData
        if(layer.priorLayer==null){
            for(int i = 0;i<this.w.length;i++){
                this.w[i] = this.w[i]-LearningRT*(this.DcIDz*layer.inputdata[i]);
            }
        }
        //否则乘以前一层的输出
        else{
            for(int i = 0;i<this.w.length;i++){
                this.w[i] = this.w[i]-LearningRT*(this.DcIDz*layer.priorLayer.neurons[i].output);
            }
        }

        //更新bias
        this.bias = this.bias - LearningRT*(this.DcIDz);
    }

}
