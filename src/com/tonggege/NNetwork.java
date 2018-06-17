package com.tonggege;

public class NNetwork {
    public static int laynumber = 2;
    public static int [] neuronnumber = {3,3};
    public static double learningrate = 0.01;
    Layer firstLayer;

    double dataSet[][];
    public NNetwork(double [][]dataSet){
        this.dataSet = dataSet;
    }
    void SetupLayer(){
        //建立层结构
        Layer p,q;
        p = null;q = null;
        for(int i = 0;i<laynumber;i++){
            //处理头结点
            if(i==0){
                this.firstLayer = new Layer(i);
                p = this.firstLayer;
            }
            else{
                p.nextLayer = new Layer(i);
                p.is_OutputLayer = (i-1==laynumber);//添加是否是最后一层
                q = p;
                p = p.nextLayer;
                if(p!=null) p.priorLayer = q;
            }
        }
    }

    void SetupNeurons(){
        Layer p = null;
        int i = 0;
        for(p = this.firstLayer;p!=null;p = p.nextLayer,i++){
            p.neurons = new Neuron[neuronnumber[i]];
            for(int j = 0;j<neuronnumber[i];j++){
                p.neurons[j].layer = p;
                p.neurons[j].index = j;
                p.neurons[j].LearningRT = learningrate;
                if(p.priorLayer==null)
                    p.neurons[j].w = new double[dataSet[0].length];
                else
                    p.neurons[j].w = new double[p.priorLayer.neurons.length];
                for(int k = 0;k<p.neurons[j].w.length;k++){
                    p.neurons[j].w[k] = Math.random()*2;
                }
            }
        }
    }

    void SetupStructure(){
        SetupLayer();
        SetupNeurons();

        for(int i = 0;i<dataSet.length;i++){
            /*  1. 前馈过程  */
            this.firstLayer.inputdata = dataSet[i];//传送初始数据
            //对于每一层
            for(Layer lay = this.firstLayer;lay!=null;lay = lay.nextLayer){
                for(int j = 0;j<lay.neurons.length;j++){
                    lay.neurons[i].figureOutput();
                }
            }

            /*  2.反向传播过程  */
            Layer lay;
            for(lay = this.firstLayer;lay.nextLayer!=null;lay = lay.nextLayer) ;
            for(;lay.priorLayer!=null;lay = lay.priorLayer){
                for(int j = 0;j<lay.neurons.length;j++){
                    lay.neurons[j].figureDcIDz();//计算Dc/Dz
                    lay.neurons[j].updatePara();//更新参数
                }
            }

        }
        Layer lay;
        for(lay = this.firstLayer;lay.nextLayer!=null;lay = lay.nextLayer) ;
        double []outcome = lay.generateOutput();
        for(int i = 0;i<lay.neurons.length;i++) System.out.print(outcome[i]);

    }
}
