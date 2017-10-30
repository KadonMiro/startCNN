using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.v01.CNN
{
    class FeatureMap
    {
        public float[][,] W;
        public float[][,] dWLast;
        private int convolutionCoreSize;
        private int poolingCoreSize;
        private float summa;
        private Signal buffer;
        public Signal Output;
        public float[,] deltaPool;
        public float[][,] deltaConv;

        public FeatureMap(int convolutionCoreSize, int poolingCoreSize, int sizeW)
        {
            W = new float[sizeW][,];
            for (int i = 0; i < sizeW; i++)
                W[i] = new float[convolutionCoreSize, convolutionCoreSize];

            dWLast = new float[sizeW][,];
            for (int i = 0; i < sizeW; i++)
                dWLast[i] = new float[convolutionCoreSize, convolutionCoreSize];

            this.convolutionCoreSize = convolutionCoreSize;
            this.poolingCoreSize = poolingCoreSize;
        }

        public void Calculate(ref Signal[] X)
        {
            Convolution(ref X);
            Pooling();
        }

        public void Convolution(ref Signal[] X)
        {

            //int edgEffect = convolutionCoreSize / 2; //проверить инт
            buffer = new Signal(X[0].X.GetLength(0) - convolutionCoreSize + 1, X[0].X.GetLength(1) - convolutionCoreSize + 1);
            for (int g = 0; g < buffer.X.GetLength(0); g++)
                for (int h = 0; h < buffer.X.GetLength(1); h++)
                    {
                        summa = 0;
                        for (int o = 0; o < X.Length; o++)
                            for (int i = 0; i < convolutionCoreSize; i++)
                                for (int j = 0; j < convolutionCoreSize; j++)
                                    summa += X[o].X[g + i, h + j] * W[o][i, j];
                        buffer.X[g, h] = ReLu();
                    }            
        }

        public void Pooling()
        {
            deltaPool = new float[buffer.X.GetLength(0), buffer.X.GetLength(1)]; // здаем размеры массива
            Output = new Signal(deltaPool.GetLength(0) / 2, deltaPool.GetLength(1) / 2);
            Array.Clear(deltaPool, 0, deltaPool.GetLength(0)* deltaPool.GetLength(1));

            int outI = 0, outJ = 0;
            for (int g = 0; g < deltaPool.GetLength(0); g += poolingCoreSize, outJ = 0, outI++)
                for (int h = 0; h < deltaPool.GetLength(1); h += poolingCoreSize, outJ++)
                {
                    float buf = 0;
                    int o = 0, p = 0;
                    for (int i = 0; i < poolingCoreSize; i++)
                        for (int j = 0; j < poolingCoreSize; j++)
                        {
                            if (buffer.X[g + i, h + j] > buf)
                            {
                                buf = buffer.X[g + i, h + j];
                                o = i;
                                p = j;
                            }
                        }

                    deltaPool[o + g, p + h] = 1;                
                    Output.X[outI, outJ] = buf;
                }
        }

        public float ReLu()
        {
            //return 1 / (1 + (float)Math.Exp(-summa));
            return Math.Max(0, summa);
        }
    }
}
