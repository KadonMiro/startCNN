﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using CNN.v01.CNN.FCN;

namespace CNN.v01.CNN
{
    class Net
    {
        public Layer[] layers;
        public FCLayer[] fclayers;
        public MiddleLayer middlelayer;
        public int[] poolingCoreSize;
        public int[] convolutionCoreSize;
        private float[] finalOutput;
        bool firstStart;


        public Net(ref int[] deepLayers, ref int[] convolutionCoreSize, ref int[] poolingCoreSize, int fullConnectedLayer, ref bool firstStart) // глубина слоев в массиве
        {
            this.firstStart = firstStart;
            int answer = 3; // колличество выходов 
            layers = new Layer[deepLayers.Length];
            int sizeW;
            for (int i = 0; i < layers.Length; i++)
            {
                if (i == 0) sizeW = 3;
                else sizeW = deepLayers[i - 1];
                layers[i] = new Layer(deepLayers[i], sizeW, convolutionCoreSize[i], poolingCoreSize[i]);
            }

            this.poolingCoreSize = poolingCoreSize;
            this.convolutionCoreSize = convolutionCoreSize;

            fclayers = new FCLayer[fullConnectedLayer];
            fclayers[fclayers.Length - 1] = new FCLayer(answer);
        }

        public void ConnectConvLayers(int i)
        {
            for (int j = 0; j < layers[i].featureMap.Length; j++)
                layers[i + 1].Input[j] = layers[i].featureMap[j].Output;
        }

        public void ConnectConvAndFCLayers()
        {
            /*
             * интеграция с полносвязными слоями
             * для каждого нейрона в слое формируется отдельный массив входов нв него,
             * это как-то неправильно должно использовать память системы, тк слишком много копий одинакового сигнала
             * хз как это работать должно
             */
            /*int k = 0;
            for (int j = 0; j < layers[layers.Length - 1].featureMap.Length; j++)
                for (int g = 0; g < layers[layers.Length - 1].featureMap[j].Output.X.GetLength(0); g++)
                    for (int h = 0; h < layers[layers.Length - 1].featureMap[j].Output.X.GetLength(1); h++)
                    {
                        for(int n = 0; n < fclayers[0].neurons.Length; n++)
                            fclayers[0].neurons[n].X[k].X = layers[layers.Length - 1].featureMap[j].Output.X[g, h];
                        k++;
                    }
            */
            int n = 0;
            for (int j = 0; j < layers[layers.Length - 1].featureMap.Length; j++)
                for (int g = 0; g < layers[layers.Length - 1].featureMap[j].Output.X.GetLength(0); g++)
                    for (int h = 0; h < layers[layers.Length - 1].featureMap[j].Output.X.GetLength(1); h++)
                    {
                        middlelayer.Input[n].X = layers[layers.Length - 1].featureMap[j].Output.X[g, h];
                        n++;
                    }

            middlelayer.Calculate();
            for (int i = 0; i < fclayers[0].neurons.Length; i++)
                for (int j = 0; j < fclayers[0].neurons.Length; j++)
                    fclayers[0].neurons[j].X[i] = middlelayer.Output[i];

        }

        public void ConnectFCLayers(int i)
        {
            /*
             * важной особенностью при написании этой части кода служит то, что каждый объект
             * класса нейрона имеет много входов и один выход!!!
             */
            for (int j = 0; j < fclayers[i].neurons.Length; j++)
                for (int h = 0; h < fclayers[i + 1].neurons.Length; h++)
                    fclayers[i + 1].neurons[h].X[j] = fclayers[i].neurons[j].Output;
        }

        public float[] Calculate(byte[,,] signals)
        {
            if(firstStart)//том можно перенести в конструктор
            {
                layers[0].CreateInput(signals.GetLength(0), signals.GetLength(1), signals.GetLength(2));
                /* 
                 * задание рандомных весов
                 */
                Random weight = new Random();

                for (int i = 0; i < layers.Length; i++)
                    for (int j = 0; j < layers[i].featureMap.Length; j++)
                    {
                        for (int k = 1; k < layers[i].featureMap[j].W.Length; k++)
                            for (int g = 0; g < layers[i].featureMap[j].W[k].GetLength(0); g++)
                                for (int h = 0; h < layers[i].featureMap[j].W[k].GetLength(1); h++)
                                {
                                    layers[i].featureMap[j].W[k][g, h] = weight.Next(-5, 5);
                                    layers[i].featureMap[j].W[k][g, h] /= 10;
                                }

                        //for (int k = 1; k < layers[i].featureMap[j].W.Length; k++)
                          //  layers[i].featureMap[j].W[k] = layers[i].featureMap[j].W[0];
                    }
            }
            /*
             * подаем на вход первого слоя сигналы
            */
            for (int n = 0; n < signals.GetLength(0); n++)
                for (int i = 0; i < signals.GetLength(1); i++)
                    for (int j = 0; j < signals.GetLength(2); j++)
                    {
                        layers[0].Input[n].X[i, j] = signals[n, i, j];// потом нужно будет изменить 0 !!!!!
                        layers[0].Input[n].X[i, j] /= 255;
                    }
            /*
             * блок вычисления сверточных слоев
             */
            for (int i = 0; i < layers.Length - 1; i++)
            {
                layers[i].Calculate();
                if (firstStart)
                    layers[i + 1].CreateInput(layers[i].featureMap.Length, layers[i].featureMap[0].Output.X.GetLength(0), layers[i].featureMap[0].Output.X.GetLength(1));
                ConnectConvLayers(i);
            }
            /*
             * определяю количествво нейронов в полносвязных слоях, кроме выходного
             */
            layers[layers.Length - 1].Calculate();
            int neuronsSize = 0;
            for (int j = 0; j < layers[layers.Length - 1].featureMap.Length; j++)
                for (int g = 0; g < layers[layers.Length - 1].featureMap[j].Output.X.GetLength(0); g++)
                    for (int h = 0; h < layers[layers.Length - 1].featureMap[j].Output.X.GetLength(1); h++)
                        neuronsSize++;
            if (firstStart)
            {
                middlelayer = new MiddleLayer(neuronsSize);
                for (int i = 0; i < fclayers.Length - 1; i++)
                {
                    fclayers[i] = new FCLayer(neuronsSize);
                    fclayers[i].CreateInput(neuronsSize);
                }
                fclayers[fclayers.Length - 1].CreateInput(neuronsSize);
            
                Random weight = new Random();
                for (int i = 0; i < fclayers.Length; i++)
                    for (int j = 0; j < fclayers[i].neurons.Length; j++)
                        for (int k = 0; k < fclayers[i].neurons[j].W.Length; k++)
                        {
                                fclayers[i].neurons[j].W[k] = weight.Next(-5, 5);
                                fclayers[i].neurons[j].W[k] /= 10;
                        }
            }

            ConnectConvAndFCLayers();
            /*
            * блок вычисления полносвязных слоев
            */
            for (int i = 0; i < fclayers.Length - 1; i++)
            {
                fclayers[i].Calculate();
                ConnectFCLayers(i);
            }
            /*
            * блок вычисления выходного слоя
            */
            if (firstStart)
                finalOutput = new float[fclayers[fclayers.Length - 1].neurons.Length];
            fclayers[fclayers.Length - 1].Calculate();

            for (int i = 0; i < finalOutput.Length; i++)
                finalOutput[i] = fclayers[fclayers.Length - 1].neurons[i].Output.X;

            return finalOutput;
        }

        public float DeepLerning(float[] trueOutput, float stepLerning, float miu)
        {
            float error = 0; // среднеквадратичная ошибка
            float GRAD = 0, dW;
            int inkr = 0; // нужна для связи олноссяхных слоев со сверточными
            //float[,] GRADConv;

            // вычисление error; сделать с помощью локальной функции
            for (int i = 0; i < finalOutput.Length; i++)
                error += (float)Math.Pow((trueOutput[i] - finalOutput[i]), 2) / 2;
            /*
             * расчет дельты для выходного слоя
             */
            for (int i = 0; i < finalOutput.Length; i++)
                fclayers[fclayers.Length - 1].neurons[i].delta = (trueOutput[i] - finalOutput[i]) * DerSigmoid(finalOutput[i]);
            /*
             * изменение весов для внутренних полносвязных слоев
             */
            for (int i = fclayers.Length - 2; i > -1 ; i--)
                for (int k = 0; k < fclayers[i].neurons.Length; k++)
                {
                    for (int j = 0; j < fclayers[i + 1].neurons.Length; j++)
                    {
                        fclayers[i].neurons[k].delta += fclayers[i + 1].neurons[j].W[k] * fclayers[i + 1].neurons[j].delta;

                        GRAD = fclayers[i + 1].neurons[j].delta * fclayers[i].neurons[k].Output.X;
                        dW = stepLerning * GRAD + miu * fclayers[i + 1].neurons[j].dWLast[k];
                        fclayers[i + 1].neurons[j].W[k] -= dW;
                        fclayers[i + 1].neurons[j].dWLast[k] = dW;
                    }
                    fclayers[i].neurons[k].delta *= DerSigmoid(fclayers[i].neurons[k].Output.X);
                }
            for (int i = 0; i < fclayers[0].neurons.Length; i++)
            {
                for (int j = 0; j < fclayers[0].neurons.Length; j++)
                {
                    middlelayer.delta[i] += fclayers[0].neurons[j].W[i] * fclayers[0].neurons[j].delta;

                    GRAD = fclayers[0].neurons[j].delta * middlelayer.Output[i].X;//fclayers[0].neurons[i].X[i].X;
                    dW = stepLerning * GRAD + miu * fclayers[0].neurons[j].dWLast[i];
                    fclayers[0].neurons[j].W[i] -= dW;
                    fclayers[0].neurons[j].dWLast[i] = dW;
                }
                //middlelayer.delta[i] *= DerSigmoid(middlelayer.Output[i].X); 
            }
            /*
             * проход по всем сверточным слоям
             */
            for (int o = layers.Length  - 1; o >= 0; o--)
            {
                for (int p = 0; p < layers[o].featureMap.Length; p++)
                {
                    /*
                        * вычисление дельты для слоя пуллинга
                        * в условии заложена связь мужду полносвязными и сверточными слоями
                        */
                    if (o == layers.Length - 1)
                    {
                        int tempi = 0, tempj = 0;

                        for (int h = 0; h < layers[o].featureMap[p].Output.X.GetLength(0); tempi += poolingCoreSize[o] - 1, tempj = 0, h++)
                            for (int g = 0; g < layers[o].featureMap[p].Output.X.GetLength(1); tempj += poolingCoreSize[o] - 1, g++)
                                for (int i = 0; i < poolingCoreSize[o]; i++)
                                    for (int j = 0; j < poolingCoreSize[o]; j++)
                                        if (layers[o].featureMap[p].deltaPool[tempi + i, tempj + j] == 1)
                                        {
                                            layers[o].featureMap[p].deltaPool[tempi + i, tempj + j] = middlelayer.delta[inkr];//fclayers[0].neurons[inkr].delta;
                                            inkr++;
                                        }
                    }
                    else
                    {
                        int tempi = 0, tempj = 0;
                        // номер карты и и номер матрицы свертки равен матрице пулинга предыдущего слоя
                        for (int h = 0; h < layers[o + 1].featureMap[p].deltaConv[p].GetLength(0); tempi += poolingCoreSize[o] - 1, tempj = 0, h++)
                            for (int g = 0; g < layers[o + 1].featureMap[p].deltaConv[p].GetLength(1); tempj += poolingCoreSize[o] - 1, g++)
                                for (int i = 0; i < poolingCoreSize[o]; i++)
                                    for (int j = 0; j < poolingCoreSize[o]; j++)
                                        if (layers[o].featureMap[p].deltaPool[tempi + i, tempj + j] == 1)
                                            layers[o].featureMap[p].deltaPool[tempi + i, tempj + j] = layers[o + 1].featureMap[p].deltaConv[p][h, g];
                    }
                    /*
                    * вычисление дельты для сверточного 
                    * проверить правилноть зерро падинга, воможно нужно увеличивать его! good
                    */
                    int zero = convolutionCoreSize[o] - 1; //zero padding; преобразование конструктора
                    float[,] temp = new float[layers[o].featureMap[p].deltaPool.GetLength(0) + 2 * zero, layers[o].featureMap[p].deltaPool.GetLength(1) + 2 * zero];

                    for (int i = 0; i < temp.GetLength(0); i++)
                        for (int j = 0; j < temp.GetLength(1); j++)
                        {
                            if ((i > zero - 1 && j > zero - 1) && (i < (temp.GetLength(0) - zero) && (j < (temp.GetLength(1) - zero))))
                                temp[i, j] = layers[o].featureMap[p].deltaPool[i - zero, j - zero];
                            else temp[i, j] = 0;
                        }
                    /*
                     * создаем экземпляры матриц дельт для сверточного слоя
                     */
                    if (firstStart)
                    {
                        layers[o].featureMap[p].deltaConv = new float[layers[o].featureMap[p].W.Length][,];
                        for (int i = 0; i < layers[o].featureMap[p].deltaConv.Length; i++)
                            layers[o].featureMap[p].deltaConv[i] = new float[temp.GetLength(0) - layers[o].featureMap[p].W[0].GetLength(0) + 1, temp.GetLength(1) - layers[o].featureMap[p].W[0].GetLength(1) + 1];
                    }
                    for (int cv = 0; cv < layers[o].featureMap[p].deltaConv.Length; cv++) // cv - число матриц свертки в каждой карте признаков
                    {
                        layers[o].featureMap[p].deltaConv[cv] = Convolution(temp, Rot180(layers[o].featureMap[p].W[cv]));

                        for (int i = 0; i < layers[o].featureMap[p].deltaConv[cv].GetLength(0); i++)
                            for (int j = 0; j < layers[o].featureMap[p].deltaConv[cv].GetLength(1); j++)
                                layers[o].featureMap[p].deltaConv[cv][i, j] *= DerReLu(layers[o].Input[cv].X[i, j]);
                    }
                    /*
                    * вычисление градиента и изменение весов
                    */
                    float[,] GRADConv = new float[layers[o].featureMap[p].W[0].GetLength(0), layers[o].featureMap[p].W[0].GetLength(1)];
                    for (int cv = 0; cv < layers[o].Input.Length; cv++)
                    {
                        GRADConv = Rot180(Convolution(layers[o].Input[cv].X, Rot180(layers[o].featureMap[p].deltaPool)));
                        for (int i = 0; i < GRADConv.GetLength(0); i++)
                            for (int j = 0; j < GRADConv.GetLength(1); j++)
                            {
                                dW = stepLerning * GRADConv[i, j] + miu * layers[o].featureMap[p].dWLast[cv][i, j];
                                layers[o].featureMap[p].W[cv][i, j] -= dW;
                                layers[o].featureMap[p].dWLast[cv][i, j] = dW;
                            }
                    }
                }
                /*
                 * для каждого входа находим общую матрицу ошибки
                 */
                for(int s = 0; s < layers[o].Input.Length; s++)
                    for (int fm = 0; fm < layers[o].featureMap.Length; fm++)
                        for (int i =0; i < layers[o].featureMap[fm].deltaConv[s].GetLength(0); i++)
                            for (int j = 0; j < layers[o].featureMap[fm].deltaConv[s].GetLength(1); j++)
                                layers[o].featureMap[s].deltaConv[s][i, j] += layers[o].featureMap[fm].deltaConv[s][i, j];
            }
            firstStart = false;
            return error;
        }
        
        float[,] Rot180(float[,] temp)
        {
            float[,] buf = new float[temp.GetLength(0), temp.GetLength(1)];
            for (int i = 0, h = temp.GetLength(0) - 1; i < temp.GetLength(0) - 1; i++, h--)
                for (int j = 0, g = temp.GetLength(1) -  1; j < temp.GetLength(1) - 1; j++, g--)
                    buf[h, g] = temp[i, j];
            return buf;
        }

        float[,] Convolution(float[,] A , float[,] B)
        {           
            float summa;
            float[,] C = new float[A.GetLength(0) - B.GetLength(0) + 1, A.GetLength(1) - B.GetLength(0) +1];

            for (int g = 0; g < A.GetLength(0) - B.GetLength(0) + 1; g++)
                for (int h = 0; h < A.GetLength(1) - B.GetLength(1) + 1; h++)
                {
                    summa = 0;
                    for (int i = 0; i < B.GetLength(0); i++)
                        for (int j = 0; j < B.GetLength(1); j++)
                            summa += A[g + i, h + j] * B[i, j];

                    C[g, h] = summa;
                }
            return C;
        }

        float DerReLu(float i)
        {
            return 1 / (1 + (float)Math.Exp(-i));
             
            //return i *= 1 - i;
        }

        float DerSigmoid(float i)
        {
            //i = i == 0 ? 0.01f : i;
            //i = i == 1 ? 0.99f : i;
            return i *= 1 - i;
        }

    }
}
